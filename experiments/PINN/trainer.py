import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


EQUATIONS = {
    "burgers": 30000,
    "schrodinger": 100000,
    "kovasznay": 100000,
    "beltrami": 100000,
    "heat": 100000,
    "poisson5d": 100000,
}

DEFAULT_STATE_ALIGNMENT_RHO = {
    "beltrami": (0.7, 0.2),
    "burgers": (0.1, 0.03),
    "heat": (0.7, 0.2),
    "kovasznay": (0.7, 0.2),
    "poisson5d": (0.5, 0.15),
    "schrodinger": (0.1, 0.03),
}

DEFAULT_LEARNING_RATE_OVERRIDES = {
    ("heat", 3): 1e-4,
}

METHODS = ["config", "pcgrad", "cagrad", "upgrad", "aligned_mtl", "imtlg"]
THREE_LOSS_EQUATIONS = {"burgers", "schrodinger", "heat", "beltrami"}


def load_equation_config(equation):
    if equation == "burgers":
        from lib_pinns.burgers.trainer import BurgersTrainerBasis, run_burgers

        return BurgersTrainerBasis, run_burgers
    if equation == "schrodinger":
        from lib_pinns.schrodinger.trainer import (
            SchrodingerTrainerBasis,
            run_schrodinger,
        )

        return SchrodingerTrainerBasis, run_schrodinger
    if equation == "kovasznay":
        from lib_pinns.kovasznay.trainer import KovasznayTrainerBasis, run_kovasznay

        return KovasznayTrainerBasis, run_kovasznay
    if equation == "beltrami":
        from lib_pinns.beltrami.trainer import BeltramiTrainerBasis, run_beltrami

        return BeltramiTrainerBasis, run_beltrami
    if equation == "heat":
        from lib_pinns.heat.trainer import HeatTrainerBasis, run_heat

        return HeatTrainerBasis, run_heat
    if equation == "poisson5d":
        from lib_pinns.poisson5d.trainer import Poisson5DTrainerBasis, run_poisson5d

        return Poisson5DTrainerBasis, run_poisson5d
    raise ValueError(f"Unknown equation: {equation}")


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def resolve_state_alignment_defaults(args):
    args.optimizer_state_correction = args.optimizer_correction == "gua"
    if not args.optimizer_state_correction:
        args.optimizer_state_correction_rho_m = 0.0
        args.optimizer_state_correction_rho_v = 0.0
        return
    rho_m, rho_v = DEFAULT_STATE_ALIGNMENT_RHO[args.equation]
    args.optimizer_state_correction_rho_m = rho_m
    args.optimizer_state_correction_rho_v = rho_v


def resolve_training_defaults(args):
    if args.lr is None:
        args.lr = DEFAULT_LEARNING_RATE_OVERRIDES.get(
            (args.equation, args.n_losses)
        )


def make_operator(args):
    from conflictfree.grad_operator import (
        AlignedMTLOperator,
        CAGradOperator,
        ConFIGOperator,
        IMTLGOperator,
        PCGradOperator,
        UPGradOperator,
    )

    if args.method == "config":
        return ConFIGOperator()
    if args.method == "pcgrad":
        return PCGradOperator()
    if args.method == "imtlg":
        return IMTLGOperator()
    if args.method == "cagrad":
        return CAGradOperator()
    if args.method == "upgrad":
        return UPGradOperator()
    if args.method == "aligned_mtl":
        return AlignedMTLOperator()
    raise ValueError(f"Unsupported method: {args.method}")


def build_run_kwargs(args):
    kwargs = {
        "n_losses": args.n_losses,
        "device": args.device,
        "seed_offset": args.seed_offset,
        "random_seed": args.random_seed,
        "run_folder_name": args.run_folder_name,
        "record_update_conflict": args.record_update_conflict,
        "conflict_cos_tol": args.conflict_cos_tol,
        "correction_distance_tol": args.correction_distance_tol,
        "conflict_log_epoch_frequency": args.conflict_log_epoch_frequency,
        "optimizer_correction": args.optimizer_correction,
        "optimizer_projection_metric": "adam",
        "optimizer_state_correction": args.optimizer_state_correction,
        "optimizer_state_correction_rho_m": args.optimizer_state_correction_rho_m,
        "optimizer_state_correction_rho_v": args.optimizer_state_correction_rho_v,
    }
    kwargs["optimizer"] = "Adam"
    for key in (
        "lr",
        "final_lr",
        "lr_scheduler",
        "validation_epoch_frequency",
        "save_epoch",
        "final_record_epoch",
    ):
        value = getattr(args, key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def parse_args():
    parser = argparse.ArgumentParser(description="Unified PINN training entry point.")
    parser.add_argument("--equation", choices=sorted(EQUATIONS), default="burgers")
    parser.add_argument("--method", choices=METHODS, default="config")
    parser.add_argument("--name", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--num-run", type=int, default=1)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-losses", type=int, choices=[2, 3], default=2)
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--run-folder-name", default="")
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--update-training-data", type=str2bool, default=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--final-lr", type=float, default=None)
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "linear", "constant", "cosine_constant"],
        default=None,
    )
    parser.add_argument("--validation-epoch-frequency", type=int, default=None)
    parser.add_argument("--save-epoch", type=int, default=None)
    parser.add_argument("--final-record-epoch", type=int, default=None)
    parser.add_argument("--record-update-conflict", type=str2bool, default=True)
    parser.add_argument("--conflict-cos-tol", type=float, default=1e-6)
    parser.add_argument("--correction-distance-tol", type=float, default=1e-8)
    parser.add_argument("--conflict-log-epoch-frequency", type=int, default=100)
    parser.add_argument(
        "--optimizer-correction", choices=["none", "gua"], default="none"
    )
    args = parser.parse_args()
    resolve_training_defaults(args)
    resolve_state_alignment_defaults(args)
    if args.random_seed is not None and args.num_run != 1:
        parser.error("--random-seed requires --num-run 1.")
    if args.run_folder_name and args.num_run != 1:
        parser.error("--run-folder-name requires --num-run 1 to avoid overwriting runs.")
    if args.n_losses == 3 and args.equation not in THREE_LOSS_EQUATIONS:
        parser.error(f"{args.equation} supports only the 2-loss decomposition.")
    if args.conflict_cos_tol < 0:
        parser.error("--conflict-cos-tol must be non-negative.")
    if args.correction_distance_tol < 0:
        parser.error("--correction-distance-tol must be non-negative.")
    return args


def _run(args):
    basis, run_func = load_equation_config(args.equation)
    from lib_pinns.trainer_basis import get_gradvec_trainer

    trainer = get_gradvec_trainer(basis, make_operator(args))
    epochs = args.epochs if args.epochs is not None else EQUATIONS[args.equation]
    name = args.name if args.name is not None else f"{args.method}_{args.n_losses}loss"
    run_func(
        name=name,
        trainer=trainer,
        epochs=epochs,
        num_run=args.num_run,
        save_path=args.save_path,
        config_file=args.config_file,
        update_training_data=args.update_training_data,
        **build_run_kwargs(args),
    )


def main():
    args = parse_args()
    _run(args)


if __name__ == "__main__":
    main()
