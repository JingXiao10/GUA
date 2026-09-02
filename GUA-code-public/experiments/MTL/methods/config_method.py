import copy
from typing import List, Tuple, Union

import torch

from conflictfree.grad_operator import ConFIGOperator


def _as_param_list(parameters):
    if parameters is None:
        return []
    if isinstance(parameters, torch.Tensor):
        return [parameters]
    return list(parameters)


def _select_param_groups(scope, shared_parameters, task_specific_parameters):
    if scope == "shared":
        return _as_param_list(shared_parameters)
    if scope == "all":
        return _as_param_list(shared_parameters) + _as_param_list(task_specific_parameters)
    raise ValueError("ConFIG parameter scope must be 'shared' or 'all'.")


class ConFIGWeightMethod:
    """ConFIG gradient construction for the shared CelebA representation."""

    def __init__(self, n_tasks: int, device: torch.device, params: str = "shared"):
        del device
        if n_tasks < 1:
            raise ValueError("n_tasks must be positive.")
        if params not in {"shared", "all"}:
            raise ValueError("ConFIG parameter scope must be 'shared' or 'all'.")
        self.n_tasks = n_tasks
        self.params = params
        self.config_operator = ConFIGOperator()

    def backward(
        self,
        losses: torch.Tensor,
        shared_parameters=None,
        task_specific_parameters=None,
        **kwargs,
    ) -> Tuple[None, dict]:
        del kwargs
        shared_parameters = _as_param_list(shared_parameters)
        task_specific_parameters = _as_param_list(task_specific_parameters)
        target_parameters = _select_param_groups(
            self.params, shared_parameters, task_specific_parameters
        )
        if not target_parameters:
            raise ValueError("ConFIG requires at least one target parameter.")

        task_grads = [
            torch.autograd.grad(loss, target_parameters, retain_graph=True)
            for loss in losses
        ]
        template = copy.deepcopy(task_grads[0])
        gradient_matrix = torch.stack(
            [self._gradients_to_vector(grads) for grads in task_grads], dim=0
        )
        merged = self.config_operator.calculate_gradient(gradient_matrix)
        for parameter, gradient in zip(
            target_parameters, self._vector_to_gradients(template, merged)
        ):
            parameter.grad = gradient

        if self.params != "all" and task_specific_parameters:
            task_grads = torch.autograd.grad(losses.sum(), task_specific_parameters)
            for parameter, gradient in zip(task_specific_parameters, task_grads):
                parameter.grad = gradient
        return None, {}

    @staticmethod
    def _gradients_to_vector(gradients):
        return torch.cat([gradient.detach().reshape(-1) for gradient in gradients])

    @staticmethod
    def _vector_to_gradients(template, vector):
        gradients = []
        start = 0
        for gradient in template:
            end = start + gradient.numel()
            gradients.append(vector[start:end].view_as(gradient))
            start = end
        return gradients


class WeightMethods:
    def __init__(self, method: str, n_tasks: int, device: torch.device, **kwargs):
        if method != "config":
            raise ValueError("This release supports only the ConFIG MTL method.")
        self.method = ConFIGWeightMethod(n_tasks=n_tasks, device=device, **kwargs)

    def backward(self, losses, **kwargs):
        return self.method.backward(losses, **kwargs)


METHODS = {"config": ConFIGWeightMethod}
