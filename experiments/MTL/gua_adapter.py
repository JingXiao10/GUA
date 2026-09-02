from typing import Iterable, List, Optional, Sequence

import torch

from conflictfree.utils import get_optimizer_proposal, project


def _as_list(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return list(parameters) if not isinstance(parameters, list) else parameters


def _params_to_vector(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    with torch.no_grad():
        return torch.cat([param.data.reshape(-1) for param in parameters])


def _apply_param_vector(parameters, vector):
    with torch.no_grad():
        start = 0
        for param in parameters:
            end = start + param.numel()
            param.copy_(vector[start:end].view_as(param))
            start = end


def _grads_to_vector(
    grads: Sequence[Optional[torch.Tensor]],
    parameters: Sequence[torch.nn.Parameter],
) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros_like(param).reshape(-1)
            if grad is None
            else grad.detach().reshape(-1)
            for grad, param in zip(grads, parameters)
        ]
    )


def _parameter_grad_vector(parameters):
    return torch.cat(
        [
            torch.zeros_like(param).reshape(-1)
            if param.grad is None
            else param.grad.detach().reshape(-1)
            for param in parameters
        ]
    )


def _task_grad_matrix(losses, shared_parameters):
    task_grads = []
    for loss in losses:
        grads = torch.autograd.grad(
            loss,
            shared_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        task_grads.append(_grads_to_vector(grads, shared_parameters))
    return torch.stack(task_grads)


def _find_param_group(optimizer, parameter):
    for group in optimizer.param_groups:
        if any(parameter is candidate for candidate in group["params"]):
            return group
    raise ValueError("Parameter is not managed by the optimizer.")


def _subset_adam_v_hat(parameters, optimizer):
    chunks = []
    with torch.no_grad():
        for parameter in parameters:
            state = optimizer.state[parameter]
            if "exp_avg_sq" not in state or "step" not in state:
                raise ValueError("CelebA GUA requires Adam optimizer state.")
            group = _find_param_group(optimizer, parameter)
            beta2 = group["betas"][1]
            step = state["step"]
            if torch.is_tensor(step):
                step = step.item()
            chunks.append(
                (state["exp_avg_sq"] / (1.0 - beta2**step)).reshape(-1)
            )
    return torch.cat(chunks)


class MTLGUAAdapter:
    """GUA for CelebA shared parameters."""

    def __init__(
        self,
        optimizer_correction: str = "none",
        conflict_tol: float = 1e-6,
    ):
        if optimizer_correction not in {"none", "gua"}:
            raise ValueError("optimizer_correction must be 'none' or 'gua'.")
        self.optimizer_correction = optimizer_correction
        if conflict_tol < 0:
            raise ValueError("conflict_tol must be non-negative.")
        self.conflict_tol = float(conflict_tol)

    @property
    def enabled(self) -> bool:
        return self.optimizer_correction == "gua"

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument(
            "--optimizer-correction",
            choices=["none", "gua"],
            default="none",
        )

    @classmethod
    def from_args(cls, args):
        return cls(
            optimizer_correction=args.optimizer_correction,
            conflict_tol=args.conflict_cos_tol,
        )

    def step(
        self,
        optimizer: torch.optim.Optimizer,
        weight_method,
        losses: torch.Tensor,
        shared_parameters: Iterable[torch.nn.Parameter],
        task_specific_parameters: Iterable[torch.nn.Parameter],
        last_shared_parameters: Iterable[torch.nn.Parameter],
        return_direction_vectors: bool = False,
    ):
        if not self.enabled:
            raise RuntimeError("MTLGUAAdapter.step requires --optimizer-correction gua.")

        shared_parameters = _as_list(shared_parameters)
        task_specific_parameters = _as_list(task_specific_parameters)
        last_shared_parameters = _as_list(last_shared_parameters)
        task_grads = _task_grad_matrix(losses, shared_parameters)
        loss, extra = weight_method.backward(
            losses=losses,
            shared_parameters=shared_parameters,
            task_specific_parameters=task_specific_parameters,
            last_shared_parameters=last_shared_parameters,
        )
        constructed_gradient = _parameter_grad_vector(shared_parameters)
        if not torch.isfinite(constructed_gradient).all():
            raise FloatingPointError("Gradient surgery returned a non-finite direction.")
        params_before = _params_to_vector(shared_parameters).clone()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        if lr <= 0:
            return loss, extra
        optimizer_direction = get_optimizer_proposal(
            optimizer,
            params_before,
            parameters=shared_parameters,
        )
        if optimizer_direction is None or not torch.isfinite(optimizer_direction).all():
            raise FloatingPointError("Optimizer returned an empty or non-finite proposal.")
        dots = task_grads @ optimizer_direction
        cosines = dots / (
            task_grads.norm(dim=1) * optimizer_direction.norm() + 1e-12
        )
        if torch.any(cosines < -self.conflict_tol):
            corrected_direction = project(
                optimizer_direction,
                task_grads,
                metric="adam",
                v_hat=_subset_adam_v_hat(shared_parameters, optimizer),
            )
        else:
            corrected_direction = optimizer_direction
        _apply_param_vector(
            shared_parameters,
            params_before - lr * corrected_direction,
        )

        extra = dict(extra or {})
        if return_direction_vectors:
            extra["gua_constructed_grad_vector"] = constructed_gradient.clone()
            extra["gua_optimizer_direction_vector"] = optimizer_direction.clone()
            extra["gua_update_direction_vector"] = corrected_direction.clone()
        return loss, extra
