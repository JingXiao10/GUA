import os
import sys
import time
from argparse import ArgumentParser
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import tqdm

CELEBA_ROOT = Path(__file__).resolve().parent
MTL_ROOT = CELEBA_ROOT.parent
PROJECT_ROOT = MTL_ROOT.parents[1]
for path in (MTL_ROOT, CELEBA_ROOT, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data import CelebaDataset  # noqa: E402
from conflict_monitor import MTLConflictMonitor  # noqa: E402
from gua_adapter import MTLGUAAdapter  # noqa: E402
from metrics import CelebaMetrics  # noqa: E402
from models import Network  # noqa: E402
from methods.config_method import WeightMethods  # noqa: E402
from utils import (  # noqa: E402
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_seed,
)

def make_loaders(
    data_path,
    batch_size,
    num_tasks,
    num_workers,
):
    train_set = CelebaDataset(data_dir=data_path, split="train", num_tasks=num_tasks)
    val_set = CelebaDataset(data_dir=data_path, split="val", num_tasks=num_tasks)
    test_set = CelebaDataset(data_dir=data_path, split="test", num_tasks=num_tasks)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def parse_seed_list(value):
    return [int(seed.strip()) for seed in value.split(",") if seed.strip()]


def resolve_run_seeds(args):
    if args.seeds is not None:
        seed_pool = parse_seed_list(args.seeds)
    elif args.num_run == 1 and args.seed_offset == 0:
        seed_pool = [args.seed]
    else:
        seed_pool = list(range(args.seed, args.seed + args.seed_offset + args.num_run))

    if args.seed_offset < 0:
        raise ValueError("--seed-offset must be non-negative")
    if args.num_run < 1:
        raise ValueError("--num-run must be at least 1")

    selected = seed_pool[args.seed_offset : args.seed_offset + args.num_run]
    if len(selected) != args.num_run:
        raise ValueError(
            f"Need {args.num_run} seeds from offset {args.seed_offset}, "
            f"but only got {len(selected)} from the seed source."
        )
    return selected


def evaluate(model, loader, device, metric):
    metric.reset()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            metric.incr(model(x), y)
    return metric.result()


def main(args):
    set_seed(args.seed)
    device = get_device(gpus=args.gpu)
    os.makedirs(args.save_dir, exist_ok=True)

    model = Network(num_tasks=args.num_tasks).to(device)
    train_loader, val_loader, test_loader = make_loaders(
        args.data_path,
        args.batch_size,
        args.num_tasks,
        args.num_workers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_f1_metrics = np.zeros([args.n_epochs, args.num_tasks], dtype=np.float32)
    test_f1_metrics = np.zeros([args.n_epochs, args.num_tasks], dtype=np.float32)
    metric = CelebaMetrics()
    loss_fn = torch.nn.BCELoss()

    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method,
        n_tasks=args.num_tasks,
        device=device,
        **weight_methods_parameters[args.method],
    )
    gua_adapter = MTLGUAAdapter.from_args(args)
    conflict_monitor = MTLConflictMonitor.from_args(args)
    global_step = 0
    best_val_f1 = 0.0
    best_f1_epoch = None
    for epoch in range(args.n_epochs):
        print(f"{epoch}/{args.n_epochs}")
        model.train()
        t0 = time.time()
        for x, y in tqdm.tqdm(train_loader):
            x = x.to(device)
            y = [y_.to(device) for y_ in y]
            y_pred = model(x)
            losses = torch.stack(
                [
                    loss_fn(y_task_pred, y_task)
                    for y_task_pred, y_task in zip(y_pred, y)
                ]
            )
            if not torch.isfinite(losses).all():
                raise FloatingPointError("CelebA task losses must be finite.")
            optimizer.zero_grad()
            shared_parameters = list(model.shared_parameters())
            task_specific_parameters = list(model.task_specific_parameters())
            last_shared_parameters = list(model.last_shared_parameters())
            conflict_trace = conflict_monitor.before_backward(losses, shared_parameters)
            if gua_adapter.enabled:
                params_before = conflict_monitor.params_before_step(shared_parameters)
                _, extra = gua_adapter.step(
                    optimizer=optimizer,
                    weight_method=weight_method,
                    losses=losses,
                    shared_parameters=shared_parameters,
                    task_specific_parameters=task_specific_parameters,
                    last_shared_parameters=last_shared_parameters,
                    return_direction_vectors=conflict_monitor.enabled,
                )
                conflict_monitor.record_step(
                    shared_parameters=shared_parameters,
                    optimizer=optimizer,
                    params_before=params_before,
                    trace=conflict_trace,
                    before_optimizer_direction=(
                        extra or {}
                    ).get("gua_constructed_grad_vector"),
                    raw_optimizer_direction=(
                        extra or {}
                    ).get("gua_optimizer_direction_vector"),
                    post_projection_direction=(
                        extra or {}
                    ).get("gua_update_direction_vector"),
                    global_step=global_step,
                )
            else:
                weight_method.backward(
                    losses=losses,
                    shared_parameters=shared_parameters,
                    task_specific_parameters=task_specific_parameters,
                    last_shared_parameters=last_shared_parameters,
                )
                if any(
                    param.grad is not None and not torch.isfinite(param.grad).all()
                    for param in model.parameters()
                ):
                    raise FloatingPointError(
                        "Gradient surgery returned a non-finite gradient."
                    )
                before_optimizer_direction = None
                if conflict_monitor.enabled:
                    before_optimizer_direction = torch.cat(
                        [
                            (
                                torch.zeros_like(param.data).view(-1)
                                if param.grad is None
                                else param.grad.detach().view(-1)
                            )
                            for param in shared_parameters
                        ]
                    )
                params_before = conflict_monitor.params_before_step(shared_parameters)
                optimizer.step()
                conflict_monitor.record_step(
                    shared_parameters=shared_parameters,
                    optimizer=optimizer,
                    params_before=params_before,
                    trace=conflict_trace,
                    before_optimizer_direction=before_optimizer_direction,
                    global_step=global_step,
                )
            global_step += 1
        t1 = time.time()
        train_speed = t1 - t0

        model.eval()
        val_result = evaluate(model, val_loader, device, metric)
        val_f1 = val_result["f1"]
        val_f1_metrics[epoch] = val_f1
        if val_f1.mean() > best_val_f1:
            best_val_f1 = val_f1.mean()
            best_f1_epoch = epoch

        test_result = evaluate(model, test_loader, device, metric)
        test_f1 = test_result["f1"]
        test_f1_metrics[epoch] = test_f1

        t2 = time.time()
        print(
            f"[info] epoch {epoch + 1} | train takes {(t1 - t0) / 60:.1f} min "
            f"| test takes {(t2 - t1) / 60:.1f} min "
            f"| val F1 {val_f1.mean():.4f} "
            f"| test F1 {test_f1.mean():.4f}"
        )

        method_name = f"{args.method}_gua" if gua_adapter.enabled else args.method
        name = f"{method_name}_sd{args.seed}_nt{args.num_tasks}"
        stats_payload = {
            "metric": test_f1_metrics,
            "val_f1": val_f1_metrics,
            "test_f1": test_f1_metrics,
            "best_epoch": best_f1_epoch,
            "best_f1_epoch": best_f1_epoch,
            "seed": args.seed,
            "num_tasks": args.num_tasks,
            "train_speed": t1 - t0,
            "conflict": conflict_monitor.summary(),
        }
        torch.save(stats_payload, Path(args.save_dir) / f"{name}.stats")

def run_many(args):
    seeds = resolve_run_seeds(args)
    print(f"[info] CelebA seeds: {seeds}")
    for run_idx, seed in enumerate(seeds, start=1):
        run_args = Namespace(**vars(args))
        run_args.seed = seed
        print(f"[info] Starting CelebA run {run_idx}/{len(seeds)} with seed {seed}")
        main(run_args)


if __name__ == "__main__":
    common_parser.add_argument(
        "--num-tasks", "--num_tasks", dest="num_tasks", type=int,
        choices=[2, 3, 5, 10, 20, 30, 40], default=40,
    )
    common_parser.add_argument("--num-workers", type=int, default=2)
    common_parser.add_argument("--save-dir", type=Path, default=Path("./save/celeba"))
    common_parser.add_argument("--num-run", type=int, default=1)
    common_parser.add_argument("--seed-offset", type=int, default=0)
    common_parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seed list. Example: --seeds 0,1,2,3,4",
    )
    MTLGUAAdapter.add_args(common_parser)
    MTLConflictMonitor.add_args(common_parser)

    parser = ArgumentParser("CelebA", parents=[common_parser])
    parser.set_defaults(
        data_path=CELEBA_ROOT / "dataset",
        lr=3e-4,
        n_epochs=15,
        batch_size=256,
        method="config",
    )
    run_many(parser.parse_args())
