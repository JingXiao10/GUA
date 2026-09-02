from typing import Iterable, Optional, Sequence

import torch

from conflictfree.utils import get_optimizer_proposal, task_gradient_conflict_stats
from utils import str2bool


def _as_list(parameters: Iterable[torch.nn.Parameter]) -> list[torch.nn.Parameter]:
    return list(parameters) if not isinstance(parameters, list) else parameters


def _grads_to_vector(
    grads: Sequence[Optional[torch.Tensor]],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    chunks = []
    for grad, param in zip(grads, parameters):
        if grad is None:
            chunks.append(torch.zeros_like(param.data).view(-1))
        else:
            chunks.append(grad.detach().view(-1))
    return torch.cat(chunks)


def _parameter_grad_vector(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    chunks = []
    for param in parameters:
        if param.grad is None:
            chunks.append(torch.zeros_like(param.data).view(-1))
        else:
            chunks.append(param.grad.detach().view(-1))
    return torch.cat(chunks)


def _params_to_vector(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([p.data.view(-1) for p in parameters])


def _task_grad_matrix(
    losses: torch.Tensor,
    shared_parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    task_grads = []
    for loss in losses:
        grads = torch.autograd.grad(
            loss,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        task_grads.append(_grads_to_vector(grads, shared_parameters))
    return torch.stack(task_grads, dim=0)


def _conflict_details(
    direction: torch.Tensor,
    task_grads: torch.Tensor,
    conflict_tol: float,
) -> tuple[bool, float, float, int, float]:
    dots = task_grads @ direction
    direction_norm = direction.norm()
    grad_norms = task_grads.norm(dim=1)
    cosines = dots / (grad_norms * direction_norm + 1e-12)
    min_dot = float(dots.min().detach().item())
    min_cos = float(cosines.min().detach().item())
    violating = cosines < -conflict_tol
    violating_count = int(violating.sum().detach().item())
    violating_fraction = float(violating.float().mean().detach().item())
    return (
        bool(torch.any(violating).item()),
        min_dot,
        min_cos,
        violating_count,
        violating_fraction,
    )


def _find_param_group(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
):
    for group in optimizer.param_groups:
        if any(parameter is p for p in group["params"]):
            return group
    raise ValueError("Parameter is not owned by the optimizer.")


def _adam_metric_diag(
    parameters: Sequence[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.Tensor]:
    """Return D=sqrt(v_hat)+eps used by the Adam metric projection."""
    chunks = []
    with torch.no_grad():
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if not state or "exp_avg_sq" not in state or "step" not in state:
                return None
            group = _find_param_group(optimizer, parameter)
            beta2 = group["betas"][1]
            step = state["step"]
            if isinstance(step, torch.Tensor):
                step = step.item()
            bias_correction2 = 1.0 - beta2 ** step
            if bias_correction2 <= 0.0:
                return None
            v_hat = state["exp_avg_sq"] / bias_correction2
            eps = group.get("eps", 1e-8)
            chunks.append(torch.sqrt(v_hat.clamp_min(0.0)) + eps)
    return torch.cat([chunk.view(-1) for chunk in chunks])


def _relative_distance(
    reference: torch.Tensor,
    corrected: torch.Tensor,
    metric_diag: Optional[torch.Tensor] = None,
) -> float:
    delta = corrected - reference
    if metric_diag is None:
        numerator = delta.norm()
        denominator = reference.norm()
    else:
        numerator = torch.sqrt((metric_diag * delta.square()).sum())
        denominator = torch.sqrt((metric_diag * reference.square()).sum())
    return float((numerator / (denominator + 1e-12)).detach().item())


def _vector_cosine(reference: torch.Tensor, corrected: torch.Tensor) -> float:
    """Cosine between the raw optimizer proposal u and applied update p."""
    denominator = reference.norm() * corrected.norm()
    return float(((reference @ corrected) / (denominator + 1e-12)).detach().item())


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
        "p95": float(torch.quantile(tensor, 0.95).item()),
    }


class MTLConflictMonitor:
    def __init__(
        self,
        enabled: bool = False,
        conflict_tol: float = 1e-6,
        correction_distance_tol: float = 1e-8,
        log_interval: int = 100,
    ):
        self.enabled = enabled
        self.conflict_tol = conflict_tol
        self.correction_distance_tol = correction_distance_tol
        self.log_interval = max(1, int(log_interval))
        self.total_steps = 0
        self.task_gradient_conflict_steps = 0
        self.task_gradient_pair_fraction_sum = 0.0
        self.task_gradient_min_pair_cos_sum = 0.0
        self.before_optimizer_conflict_steps = 0
        self.raw_optimizer_conflict_steps = 0
        self.post_projection_conflict_steps = 0
        self.gua_steps = 0
        self.before_optimizer_min_cos_sum = 0.0
        self.raw_optimizer_min_cos_sum = 0.0
        self.post_projection_min_cos_sum = 0.0
        self.before_optimizer_violating_tasks_sum = 0.0
        self.raw_optimizer_violating_tasks_sum = 0.0
        self.post_projection_violating_tasks_sum = 0.0
        self.before_optimizer_violating_fraction_sum = 0.0
        self.raw_optimizer_violating_fraction_sum = 0.0
        self.post_projection_violating_fraction_sum = 0.0
        self.correction_d2_values: list[float] = []
        self.correction_adam_values: list[float] = []
        self.projection_update_cosine_values: list[float] = []
        self.effective_correction_steps = 0
        self.last = {}

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument("--record-update-conflict", type=str2bool, default=False)
        parser.add_argument("--conflict-cos-tol", type=float, default=1e-6)
        parser.add_argument(
            "--correction-distance-tol",
            type=float,
            default=1e-8,
            help="Relative Euclidean correction threshold for effective intervention rate.",
        )
        parser.add_argument("--conflict-log-interval", type=int, default=100)

    @classmethod
    def from_args(cls, args):
        return cls(
            enabled=args.record_update_conflict,
            conflict_tol=args.conflict_cos_tol,
            correction_distance_tol=args.correction_distance_tol,
            log_interval=args.conflict_log_interval,
        )

    def before_backward(
        self,
        losses: torch.Tensor,
        shared_parameters: Iterable[torch.nn.Parameter],
    ):
        if not self.enabled:
            return None
        shared_parameters = _as_list(shared_parameters)
        return {
            "shared_parameters": shared_parameters,
            "task_grads": _task_grad_matrix(losses, shared_parameters).detach(),
        }

    def params_before_step(self, shared_parameters: Iterable[torch.nn.Parameter]):
        if not self.enabled:
            return None
        return _params_to_vector(_as_list(shared_parameters)).clone()

    def record_step(
        self,
        *,
        shared_parameters: Iterable[torch.nn.Parameter],
        optimizer: torch.optim.Optimizer,
        params_before: torch.Tensor,
        trace,
        before_optimizer_direction: Optional[torch.Tensor] = None,
        raw_optimizer_direction: Optional[torch.Tensor] = None,
        post_projection_direction: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
    ) -> None:
        if not self.enabled or trace is None or params_before is None:
            return

        shared_parameters = _as_list(shared_parameters)
        task_grads = trace["task_grads"]
        if before_optimizer_direction is None:
            before_optimizer_direction = _parameter_grad_vector(shared_parameters)
        gua_direction_available = post_projection_direction is not None
        if raw_optimizer_direction is None:
            lr = optimizer.param_groups[0]["lr"]
            if lr <= 0:
                return
            raw_optimizer_direction = get_optimizer_proposal(
                optimizer,
                params_before,
                parameters=shared_parameters,
            )
        if post_projection_direction is None:
            post_projection_direction = raw_optimizer_direction

        task_has_conflict, task_min_pair_cos, task_pair_fraction = (
            task_gradient_conflict_stats(task_grads, self.conflict_tol)
        )
        (
            before_has_conflict,
            before_min_dot,
            before_min_cos,
            before_violating_tasks,
            before_violating_fraction,
        ) = _conflict_details(
            before_optimizer_direction.detach(),
            task_grads,
            self.conflict_tol,
        )
        (
            raw_has_conflict,
            raw_min_dot,
            raw_min_cos,
            raw_violating_tasks,
            raw_violating_fraction,
        ) = _conflict_details(
            raw_optimizer_direction.detach(),
            task_grads,
            self.conflict_tol,
        )
        (
            post_has_conflict,
            post_min_dot,
            post_min_cos,
            post_violating_tasks,
            post_violating_fraction,
        ) = _conflict_details(
            post_projection_direction.detach(),
            task_grads,
            self.conflict_tol,
        )
        correction_d2 = _relative_distance(
            raw_optimizer_direction.detach(), post_projection_direction.detach()
        )
        correction_adam = _relative_distance(
            raw_optimizer_direction.detach(),
            post_projection_direction.detach(),
            _adam_metric_diag(shared_parameters, optimizer),
        )
        projection_update_cosine = _vector_cosine(
            raw_optimizer_direction.detach(), post_projection_direction.detach()
        )
        if correction_d2 <= self.correction_distance_tol:
            correction_d2 = 0.0
            correction_adam = 0.0
            projection_update_cosine = 1.0
        self.total_steps += 1
        self.task_gradient_conflict_steps += int(task_has_conflict)
        self.task_gradient_pair_fraction_sum += task_pair_fraction
        self.task_gradient_min_pair_cos_sum += task_min_pair_cos
        self.before_optimizer_conflict_steps += int(before_has_conflict)
        self.raw_optimizer_conflict_steps += int(raw_has_conflict)
        self.post_projection_conflict_steps += int(post_has_conflict)
        self.gua_steps += int(gua_direction_available)
        self.before_optimizer_min_cos_sum += before_min_cos
        self.raw_optimizer_min_cos_sum += raw_min_cos
        self.post_projection_min_cos_sum += post_min_cos
        self.before_optimizer_violating_tasks_sum += before_violating_tasks
        self.raw_optimizer_violating_tasks_sum += raw_violating_tasks
        self.post_projection_violating_tasks_sum += post_violating_tasks
        self.before_optimizer_violating_fraction_sum += before_violating_fraction
        self.raw_optimizer_violating_fraction_sum += raw_violating_fraction
        self.post_projection_violating_fraction_sum += post_violating_fraction
        self.correction_d2_values.append(correction_d2)
        self.correction_adam_values.append(correction_adam)
        self.projection_update_cosine_values.append(projection_update_cosine)
        self.effective_correction_steps += int(
            correction_d2 > self.correction_distance_tol
        )
        self.last = {
            "task_gradient_min_pair_cos": task_min_pair_cos,
            "task_gradient_conflicting_pair_fraction": task_pair_fraction,
            "task_gradient_conflict": task_has_conflict,
            "before_optimizer_min_dot": before_min_dot,
            "raw_optimizer_min_dot": raw_min_dot,
            "post_projection_min_dot": post_min_dot,
            "before_optimizer_min_cos": before_min_cos,
            "raw_optimizer_min_cos": raw_min_cos,
            "post_projection_min_cos": post_min_cos,
            "before_optimizer_violating_tasks": before_violating_tasks,
            "raw_optimizer_violating_tasks": raw_violating_tasks,
            "post_projection_violating_tasks": post_violating_tasks,
            "before_optimizer_violating_task_fraction": before_violating_fraction,
            "raw_optimizer_violating_task_fraction": raw_violating_fraction,
            "post_projection_violating_task_fraction": post_violating_fraction,
            "before_optimizer_conflict": before_has_conflict,
            "raw_optimizer_conflict": raw_has_conflict,
            "post_projection_conflict": post_has_conflict,
            "cos_p_u": projection_update_cosine,
        }

        if (
            global_step is not None
            and global_step > 0
            and global_step % self.log_interval == 0
        ):
            summary = self.summary()
            r_p_text = (
                ""
                if summary["r_p"] is None
                else f" r_p:{summary['r_p']:.2%}"
            )
            print(
                "[Conflict] "
                f"step:{global_step} steps:{summary['conflict_total_steps']} "
                f"r_g:{summary['r_g']:.2%} "
                f"r_g_pair:{summary['r_g_pair']:.2%} "
                f"r_a:{summary['r_a']:.2%} "
                f"r_u:{summary['r_u']:.2%}"
                f"{r_p_text} "
                f"correction_d2_median:{summary['correction_d2_median']:.6g} "
                f"cos_p_u_median:{summary['cos_p_u_median']:.6f}"
            )

    def summary(self) -> dict:
        if self.total_steps == 0:
            return {
                "conflict_total_steps": 0,
                "r_g": 0.0,
                "r_g_pair": 0.0,
                "r_a": 0.0,
                "r_u": 0.0,
                "r_p": None,
                "before_optimizer_conflict_rate": 0.0,
                "raw_optimizer_conflict_rate": 0.0,
                "post_projection_conflict_rate": 0.0,
                "before_optimizer_min_cos_avg": 0.0,
                "raw_optimizer_min_cos_avg": 0.0,
                "post_projection_min_cos_avg": 0.0,
                "before_optimizer_mean_violating_tasks": 0.0,
                "raw_optimizer_mean_violating_tasks": 0.0,
                "post_projection_mean_violating_tasks": 0.0,
                "before_optimizer_mean_violating_task_fraction": 0.0,
                "raw_optimizer_mean_violating_task_fraction": 0.0,
                "post_projection_mean_violating_task_fraction": 0.0,
                "effective_correction_rate": 0.0,
                "correction_distance_tol": self.correction_distance_tol,
                "correction_d2_mean": 0.0,
                "correction_d2_median": 0.0,
                "correction_d2_p90": 0.0,
                "correction_d2_p95": 0.0,
                "correction_adam_mean": 0.0,
                "correction_adam_median": 0.0,
                "correction_adam_p90": 0.0,
                "correction_adam_p95": 0.0,
                "cos_p_u_mean": 0.0,
                "cos_p_u_median": 0.0,
                "cos_p_u_p90": 0.0,
                "cos_p_u_p95": 0.0,
                **self.last,
            }
        d2 = _distribution(self.correction_d2_values)
        adam = _distribution(self.correction_adam_values)
        cos_p_u = _distribution(self.projection_update_cosine_values)
        raw_rate = self.raw_optimizer_conflict_steps / self.total_steps
        post_rate = self.post_projection_conflict_steps / self.total_steps
        return {
            "conflict_total_steps": self.total_steps,
            "r_g": self.task_gradient_conflict_steps / self.total_steps,
            "r_g_pair": self.task_gradient_pair_fraction_sum / self.total_steps,
            "r_a": self.before_optimizer_conflict_steps / self.total_steps,
            "r_u": raw_rate,
            "r_p": post_rate if self.gua_steps == self.total_steps else None,
            "task_gradient_min_pair_cos_avg": (
                self.task_gradient_min_pair_cos_sum / self.total_steps
            ),
            "before_optimizer_conflict_rate": (
                self.before_optimizer_conflict_steps / self.total_steps
            ),
            "raw_optimizer_conflict_rate": raw_rate,
            "post_projection_conflict_rate": post_rate,
            "before_optimizer_min_cos_avg": (
                self.before_optimizer_min_cos_sum / self.total_steps
            ),
            "raw_optimizer_min_cos_avg": self.raw_optimizer_min_cos_sum / self.total_steps,
            "post_projection_min_cos_avg": self.post_projection_min_cos_sum / self.total_steps,
            "before_optimizer_mean_violating_tasks": (
                self.before_optimizer_violating_tasks_sum / self.total_steps
            ),
            "raw_optimizer_mean_violating_tasks": (
                self.raw_optimizer_violating_tasks_sum / self.total_steps
            ),
            "post_projection_mean_violating_tasks": (
                self.post_projection_violating_tasks_sum / self.total_steps
            ),
            "before_optimizer_mean_violating_task_fraction": (
                self.before_optimizer_violating_fraction_sum / self.total_steps
            ),
            "raw_optimizer_mean_violating_task_fraction": (
                self.raw_optimizer_violating_fraction_sum / self.total_steps
            ),
            "post_projection_mean_violating_task_fraction": (
                self.post_projection_violating_fraction_sum / self.total_steps
            ),
            "effective_correction_rate": (
                self.effective_correction_steps / self.total_steps
            ),
            "correction_distance_tol": self.correction_distance_tol,
            "correction_d2_mean": d2["mean"],
            "correction_d2_median": d2["median"],
            "correction_d2_p90": d2["p90"],
            "correction_d2_p95": d2["p95"],
            "correction_adam_mean": adam["mean"],
            "correction_adam_median": adam["median"],
            "correction_adam_p90": adam["p90"],
            "correction_adam_p95": adam["p95"],
            "cos_p_u_mean": cos_p_u["mean"],
            "cos_p_u_median": cos_p_u["median"],
            "cos_p_u_p90": cos_p_u["p90"],
            "cos_p_u_p95": cos_p_u["p95"],
            **self.last,
        }
