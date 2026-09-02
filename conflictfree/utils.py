# usr/bin/python3
# -*- coding: UTF-8 -*-
import torch
import itertools
from typing import Optional, Sequence, Union, Literal


def _solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        return torch.linalg.solve(matrix, rhs)
    solution, _ = torch.solve(rhs.unsqueeze(-1), matrix)
    return solution.squeeze(-1)


def _solve_nonnegative_qp(
    gram: torch.Tensor,
    linear: torch.Tensor,
    tol: float,
    ridge: float,
    max_iterations: Optional[int] = None,
) -> torch.Tensor:
    """Solve min 0.5*x^T*gram*x + linear^T*x subject to x >= 0."""
    size = linear.numel()
    solution = torch.zeros_like(linear)
    passive = torch.zeros(size, dtype=torch.bool, device=linear.device)
    max_iterations = max_iterations or max(50, 10 * size)
    scale = max(float(torch.diag(gram).abs().max().item()), 1.0)
    regularization = ridge * scale

    for _ in range(max_iterations):
        gradient = gram @ solution + linear
        candidates = (~passive) & (gradient < -tol)
        if not torch.any(candidates):
            return solution

        masked_gradient = torch.where(
            candidates,
            gradient,
            torch.full_like(gradient, float("inf")),
        )
        passive[torch.argmin(masked_gradient)] = True

        while torch.any(passive):
            indices = torch.where(passive)[0]
            sub_gram = gram[indices][:, indices]
            eye = torch.eye(
                indices.numel(), dtype=gram.dtype, device=gram.device
            )
            try:
                candidate_values = _solve(
                    sub_gram + regularization * eye,
                    -linear[indices],
                )
            except RuntimeError:
                candidate_values = torch.linalg.lstsq(
                    sub_gram + regularization * eye,
                    -linear[indices],
                ).solution

            candidate = torch.zeros_like(solution)
            candidate[indices] = candidate_values
            if torch.all(candidate_values > tol):
                solution = candidate
                break

            nonpositive = passive & (candidate <= tol)
            denominator = solution[nonpositive] - candidate[nonpositive]
            valid = denominator > 0
            if torch.any(valid):
                alpha = torch.min(solution[nonpositive][valid] / denominator[valid])
                solution = solution + alpha * (candidate - solution)
            else:
                solution = candidate.clamp_min(0.0)
            remove = passive & (solution <= tol)
            passive[remove] = False
            solution[remove] = 0.0

    gradient = gram @ solution + linear
    if torch.any((~passive) & (gradient < -10 * tol)):
        raise RuntimeError("Nonnegative QP active-set solver did not converge.")
    return solution


def _repair_projection_feasibility(
    projected: torch.Tensor,
    gradients: torch.Tensor,
    inverse_metric: Optional[torch.Tensor] = None,
    max_iterations: Optional[int] = None,
    feasibility_cos_tol: float = 1e-10,
) -> torch.Tensor:
    """Repair only violations larger than floating-point cosine error."""
    result = projected.clone()
    inverse_metric = (
        torch.ones_like(result) if inverse_metric is None else inverse_metric
    )
    eps = torch.finfo(result.dtype).eps
    numerical_tol = max(float(feasibility_cos_tol), 4.0 * eps)
    max_iterations = max_iterations or max(50, 5 * gradients.shape[0])

    for _ in range(max_iterations):
        dots = gradients @ result
        scales = gradients.norm(dim=1) * result.norm().clamp_min(eps)
        violated = dots < -numerical_tol * scales
        if not torch.any(violated):
            return result

        # Repair one most-negative normalized constraint at a time. This is a
        # tiny metric projection used only to offset float32 dot-product error.
        normalized = dots / scales.clamp_min(eps)
        index = torch.argmin(normalized)
        gradient = gradients[index]
        denominator = torch.dot(gradient * inverse_metric, gradient).clamp_min(eps)
        step = -dots[index] / denominator
        result = result + step * inverse_metric * gradient

    dots = gradients @ result
    scales = gradients.norm(dim=1) * result.norm().clamp_min(eps)
    if torch.any(dots < -numerical_tol * scales):
        raise RuntimeError("Projection feasibility repair did not converge.")
    return result


def get_para_vector(network: torch.nn.Module) -> torch.Tensor:
    """
    Returns the parameter vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.

    Returns:
        torch.Tensor: The parameter vector of the network.
    """
    with torch.no_grad():
        para_vec = None
        for par in network.parameters():
            viewed = par.data.view(-1)
            if para_vec is None:
                para_vec = viewed
            else:
                para_vec = torch.cat((para_vec, viewed))
        return para_vec


def get_gradient_vector(
    network: torch.nn.Module, none_grad_mode: Literal["raise", "zero", "skip"] = "skip"
) -> torch.Tensor:
    """
    Returns the gradient vector of the given network.

    Args:
        network (torch.nn.Module): The network for which to compute the gradient vector.
        none_grad_mode (Literal['raise', 'zero', 'skip']): The mode to handle None gradients. default: 'skip'
            - 'raise': Raise an error when the gradient of a parameter is None.
            - 'zero': Replace the None gradient with a zero tensor.
            - 'skip': Skip the None gradient.
                        The None gradient usually occurs when part of the network is not trainable (e.g., fine-tuning)
            or the weight is not used to calculate the current loss (e.g., different parts of the network calculate different losses).
            If all of your losses are calculated using the same part of the network, you should set none_grad_mode to 'skip'.
            If your losses are calculated using different parts of the network, you should set none_grad_mode to 'zero' to ensure the gradients have the same shape.

    Returns:
        torch.Tensor: The gradient vector of the network.
    """
    with torch.no_grad():
        grad_vec = None
        for par in network.parameters():
            if par.grad is None:
                if none_grad_mode == "raise":
                    raise RuntimeError("None gradient detected.")
                elif none_grad_mode == "zero":
                    viewed = torch.zeros_like(par.data.view(-1))
                elif none_grad_mode == "skip":
                    continue
                else:
                    raise ValueError(f"Invalid none_grad_mode '{none_grad_mode}'.")
            else:
                viewed = par.grad.data.view(-1)
            if grad_vec is None:
                grad_vec = viewed
            else:
                grad_vec = torch.cat((grad_vec, viewed))
        return grad_vec


def apply_gradient_vector_para_based(
    network: torch.nn.Module,
    grad_vec: torch.Tensor,
) -> None:
    """
    Applies a gradient vector to the network's parameters.
    Please only use this function when you are sure that the length of `grad_vec` is the same of your network's parameters.
    This happens when you use `get_gradient_vector` with `none_grad_mode` set to 'zero'.
    Or, the 'none_grad_mode' is 'skip' but all of the parameters in your network is involved in the loss calculation.

    Args:
        network (torch.nn.Module): The network to apply the gradient vector to.
        grad_vec (torch.Tensor): The gradient vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.grad = grad_vec[start:end].view(par.data.shape)
            start = end


def apply_para_vector(network: torch.nn.Module, para_vec: torch.Tensor) -> None:
    """
    Applies a parameter vector to the network's parameters.

    Args:
        network (torch.nn.Module): The network to apply the parameter vector to.
        para_vec (torch.Tensor): The parameter vector to apply.
    """
    with torch.no_grad():
        start = 0
        for par in network.parameters():
            end = start + par.data.view(-1).shape[0]
            par.data = para_vec[start:end].view(par.data.shape)
            start = end


def _optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if not learning_rates:
        raise ValueError("Optimizer has no parameter groups.")
    if any(lr != learning_rates[0] for lr in learning_rates[1:]):
        raise ValueError(
            "Exact optimizer proposals require one shared learning rate because "
            "the PINN correction path applies one global step size."
        )
    return learning_rates[0]


def _parameter_slices(optimizer, params_before, parameters=None):
    group_by_parameter = {
        id(param): group
        for group in optimizer.param_groups
        for param in group["params"]
    }
    if parameters is None:
        parameters = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
        ]
    offset = 0
    for param in parameters:
        group = group_by_parameter.get(id(param))
        if group is None:
            raise ValueError("Requested parameter is not managed by the optimizer.")
        end = offset + param.numel()
        if end > params_before.numel():
            raise ValueError("params_before is shorter than the requested parameter vector.")
        yield group, param, params_before[offset:end].view_as(param)
        offset = end
    if offset != params_before.numel():
        raise ValueError("params_before does not match the optimizer parameter vector.")


def _sgd_proposal(optimizer, params_before, parameters=None):
    flat = []
    for group, param, param_before in _parameter_slices(optimizer, params_before, parameters):
        if param.grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
            continue
        grad = param.grad.detach()
        if group.get("maximize", False):
            grad = -grad
        if group["weight_decay"] != 0:
            grad = grad + group["weight_decay"] * param_before
        if group["momentum"] != 0:
            buffer = optimizer.state[param]["momentum_buffer"]
            if group["nesterov"]:
                grad = grad + group["momentum"] * buffer
            else:
                grad = buffer
        flat.append(grad.reshape(-1))
    return torch.cat(flat)


def _rmsprop_proposal(optimizer, params_before, parameters=None):
    flat = []
    for group, param, param_before in _parameter_slices(optimizer, params_before, parameters):
        if param.grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
            continue
        state = optimizer.state[param]
        if group["momentum"] > 0:
            proposal = state["momentum_buffer"]
        else:
            grad = param.grad.detach()
            if group.get("maximize", False):
                grad = -grad
            if group["weight_decay"] != 0:
                grad = grad + group["weight_decay"] * param_before
            variance = state["square_avg"]
            if group["centered"]:
                variance = variance - state["grad_avg"].square()
            proposal = grad / (variance.sqrt() + group["eps"])
        flat.append(proposal.reshape(-1))
    return torch.cat(flat)


def _adam_proposal(optimizer, params_before, parameters=None):
    flat = []
    is_adamw = isinstance(optimizer, torch.optim.AdamW)
    for group, param, param_before in _parameter_slices(optimizer, params_before, parameters):
        if param.grad is None:
            flat.append(torch.zeros_like(param).reshape(-1))
            continue
        state = optimizer.state[param]
        step_value = state["step"]
        step = float(step_value.item() if torch.is_tensor(step_value) else step_value)
        beta1, beta2 = group["betas"]
        bias_correction1 = 1.0 - float(beta1) ** step
        bias_correction2 = 1.0 - float(beta2) ** step
        variance = (
            state["max_exp_avg_sq"] if group.get("amsgrad", False)
            else state["exp_avg_sq"]
        )
        denominator = (variance / bias_correction2).sqrt() + group["eps"]
        proposal = (state["exp_avg"] / bias_correction1) / denominator
        decoupled = is_adamw or group.get("decoupled_weight_decay", False)
        if decoupled and group["weight_decay"] != 0:
            proposal = proposal + group["weight_decay"] * param_before
        flat.append(proposal.reshape(-1))
    return torch.cat(flat)


def get_optimizer_proposal(
    optimizer: torch.optim.Optimizer,
    params_before: torch.Tensor,
    parameters=None,
) -> Optional[torch.Tensor]:
    """Return the exact direction proposed by the optimizer's latest step.

    The optimizer must already have completed ``step()``. Unlike recovering a
    direction from parameter differences, this uses optimizer state or a custom
    optimizer's proposal-tracking protocol and therefore avoids float32
    cancellation.
    """
    lr = _optimizer_learning_rate(optimizer)
    if lr <= 0:
        return None

    get_last_proposal = getattr(optimizer, "get_last_proposal", None)
    if callable(get_last_proposal):
        proposal = get_last_proposal(parameters)
        if proposal is None:
            raise RuntimeError("Optimizer did not record a proposal during step().")
        if proposal.numel() != params_before.numel():
            raise ValueError("Recorded optimizer proposal has the wrong size.")
        return proposal.to(device=params_before.device, dtype=params_before.dtype)

    if isinstance(optimizer, torch.optim.SGD):
        return _sgd_proposal(optimizer, params_before, parameters)
    if isinstance(optimizer, torch.optim.RMSprop):
        return _rmsprop_proposal(optimizer, params_before, parameters)
    if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return _adam_proposal(optimizer, params_before, parameters)
    raise NotImplementedError(
        f"Exact proposal extraction is not implemented for {type(optimizer).__name__}."
    )


def task_gradient_conflict_stats(
    gradients: torch.Tensor,
    conflict_tol: float = 1e-6,
) -> tuple[bool, float, float]:
    """Return any-pair conflict, minimum pair cosine, and conflict-pair fraction."""
    if gradients.ndim != 2:
        raise ValueError("gradients must have shape [n_tasks, n_parameters].")
    n_tasks = gradients.shape[0]
    if n_tasks < 2:
        return False, 0.0, 0.0

    with torch.no_grad():
        norms = gradients.norm(dim=1)
        cosines = (gradients @ gradients.T) / (
            norms.unsqueeze(1) * norms.unsqueeze(0) + 1e-12
        )
        pair_indices = torch.triu_indices(
            n_tasks,
            n_tasks,
            offset=1,
            device=gradients.device,
        )
        pair_cosines = cosines[pair_indices[0], pair_indices[1]]
        conflicts = pair_cosines < -conflict_tol
        return (
            bool(conflicts.any().item()),
            float(pair_cosines.min().item()),
            float(conflicts.float().mean().item()),
        )


def project(
    u: torch.Tensor,
    grad_list: Union[torch.Tensor, Sequence[torch.Tensor]],
    metric: str = "euclidean",
    *,
    v_hat: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    tol: float = 1e-10,
    ridge: float = 1e-12,
) -> torch.Tensor:
    """Project an optimizer proposal onto the conflict-free cone.

    ``metric='euclidean'`` uses the identity metric. ``metric='adam'`` uses
    Adam's diagonal denominator ``sqrt(v_hat) + eps``. Both choices solve the
    same constrained projection problem:

        min_p ||p - u||_M^2
        s.t.  g_i^T p >= 0

    Small task sets use exact active-constraint enumeration. Larger task sets
    solve the equivalent nonnegative dual QP with an active-set method.
    """
    if isinstance(grad_list, torch.Tensor):
        G = grad_list
    else:
        G = torch.stack(grad_list)

    if u.ndim != 1:
        raise ValueError("u must be a flat proposal vector.")
    if G.ndim != 2 or G.shape[1] != u.numel() or G.shape[0] == 0:
        raise ValueError("gradients must have shape [n_tasks, len(u)] with n_tasks > 0.")
    if not torch.isfinite(u).all() or not torch.isfinite(G).all():
        raise FloatingPointError("Projection inputs must be finite.")
    if tol < 0 or ridge < 0:
        raise ValueError("Projection tolerances must be non-negative.")

    orig_dtype = u.dtype
    u = u.double()
    G = G.double()
    tiny = torch.finfo(u.dtype).eps
    u_scale = torch.linalg.vector_norm(u)
    if u_scale <= tiny:
        return u.to(dtype=orig_dtype).clone()
    gradient_norms = torch.linalg.vector_norm(G, dim=1)
    active_gradients = gradient_norms > tiny
    if not torch.any(active_gradients):
        return u.to(dtype=orig_dtype).clone()
    G = G[active_gradients] / gradient_norms[active_gradients].unsqueeze(1)
    u = u / u_scale
    if metric == "euclidean":
        inverse_metric = torch.ones_like(u)
    elif metric == "adam":
        if v_hat is None:
            raise ValueError("Projection under the Adam metric requires v_hat.")
        v_hat = v_hat.to(device=u.device, dtype=u.dtype)
        if v_hat.shape != u.shape:
            raise ValueError("v_hat must have the same shape as the proposal.")
        if not torch.isfinite(v_hat).all():
            raise FloatingPointError("v_hat must be finite.")
        inverse_metric = 1.0 / (torch.sqrt(v_hat.clamp_min(0.0)) + eps).clamp_min(1e-20)
    else:
        raise ValueError(f"Unsupported projection metric: {metric!r}.")

    m = G.shape[0]
    b = G @ u

    if torch.all(b >= 0):
        return (u * u_scale).to(dtype=orig_dtype).clone()

    K = (G * inverse_metric.unsqueeze(0)) @ G.T
    K = 0.5 * (K + K.T)

    if m > 8:
        best = _solve_nonnegative_qp(K, b, tol=tol, ridge=ridge)
    else:
        best = None
        best_obj = None
        indices = list(range(m))

        for r in range(1, m + 1):
            for active in itertools.combinations(indices, r):
                active = list(active)
                S = torch.tensor(active, device=u.device)
                Kss = K[S][:, S]
                bs = b[S]
                eye = torch.eye(len(active), dtype=u.dtype, device=u.device)
                Kss = Kss + ridge * eye

                try:
                    lam = _solve(Kss, -bs)
                except RuntimeError:
                    continue

                if torch.any(lam < 0):
                    continue

                full_lam = torch.zeros(m, dtype=u.dtype, device=u.device)
                full_lam[S] = lam

                residual = b + K @ full_lam
                if torch.any(residual < -1e-8):
                    continue

                obj = 0.5 * full_lam @ (K @ full_lam) + b @ full_lam
                if best_obj is None or obj < best_obj:
                    best_obj = obj
                    best = full_lam

        if best is None:
            best = _solve_nonnegative_qp(K, b, tol=tol, ridge=ridge)

    p = ((u + inverse_metric * (G.T @ best)) * u_scale).to(dtype=orig_dtype)
    if not torch.isfinite(p).all():
        raise FloatingPointError("Projection solver returned a non-finite update.")
    zero_tol = max(float(tol), 16.0 * torch.finfo(torch.float64).eps)
    if torch.linalg.vector_norm(p.double()) <= zero_tol * u_scale:
        return torch.zeros_like(p)
    return _repair_projection_feasibility(
        p,
        G.to(dtype=orig_dtype),
        inverse_metric=inverse_metric.to(dtype=orig_dtype),
    )


def get_adam_v_hat_vector(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """
    Returns the bias-corrected Adam second-moment vector.

    The returned vector is v_hat in Adam:

        v_hat = v_t / (1 - beta2 ** t)

    This helper expects an Adam-like optimizer whose parameter states contain
    exp_avg_sq and step.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("Adam-metric projection requires Adam or AdamW.")
    v_hat = None
    with torch.no_grad():
        for group in optimizer.param_groups:
            beta2 = group["betas"][1]
            for par in group["params"]:
                state = optimizer.state[par]
                if "exp_avg_sq" not in state or "step" not in state:
                    raise ValueError(
                        "Projection under the Adam metric requires Adam-like optimizer state."
                    )
                step = state["step"]
                if isinstance(step, torch.Tensor):
                    step = step.item()
                bias_correction2 = 1.0 - beta2 ** step
                if bias_correction2 <= 0:
                    raise ValueError("Adam state step must be positive.")
                viewed = (state["exp_avg_sq"] / bias_correction2).view(-1)
                if v_hat is None:
                    v_hat = viewed
                else:
                    v_hat = torch.cat((v_hat, viewed))
    if v_hat is None or not torch.isfinite(v_hat).all():
        raise FloatingPointError("Adam v_hat is empty or non-finite.")
    return v_hat






def correct_adam_consistent_state_from_direction(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target_direction: torch.Tensor,
    rho_m: float,
    rho_v: float,
) -> None:
    """
    Soft-corrects Adam's first and second moments toward the executed direction.

    The corrected direction is converted to a pseudo-gradient under Adam's
    current denominator:

        g_corr = (sqrt(v_hat) + eps) * target_direction.

    Then both bias-corrected states are softly moved toward
    m_hat = g_corr and v_hat = g_corr ** 2. This keeps the optimizer memory
    more consistent with the direction that was actually applied.
    """
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        raise TypeError("GUA state alignment requires Adam or AdamW.")
    rho_m = float(rho_m)
    rho_v = float(rho_v)
    if not 0.0 <= rho_m <= 1.0 or not 0.0 <= rho_v <= 1.0:
        raise ValueError("State-alignment strengths must lie in [0, 1].")
    expected_size = sum(par.numel() for group in optimizer.param_groups for par in group["params"])
    if target_direction.ndim != 1 or target_direction.numel() != expected_size:
        raise ValueError("target_direction does not match the optimizer parameter vector.")
    if not torch.isfinite(target_direction).all():
        raise FloatingPointError("target_direction must be finite.")
    with torch.no_grad():
        start = 0
        for group in optimizer.param_groups:
            beta1, beta2 = group["betas"]
            eps = group.get("eps", 1e-8)
            for par in group["params"]:
                state = optimizer.state[par]
                if (
                    "exp_avg" not in state
                    or "exp_avg_sq" not in state
                    or "step" not in state
                ):
                    raise ValueError(
                        "GUA state alignment requires Adam states with "
                        "exp_avg, exp_avg_sq, and step."
                    )

                step = state["step"]
                if isinstance(step, torch.Tensor):
                    step = step.item()

                end = start + par.data.view(-1).shape[0]
                direction_i = target_direction[start:end].view_as(par.data)
                start = end

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                if bias_correction1 <= 0 or bias_correction2 <= 0:
                    raise ValueError("Adam state step must be positive before alignment.")

                m_hat = state["exp_avg"] / bias_correction1
                v_hat = state["exp_avg_sq"] / bias_correction2
                pseudo_grad = (torch.sqrt(v_hat.clamp_min(0.0)) + eps) * direction_i

                m_hat_new = (1.0 - rho_m) * m_hat + rho_m * pseudo_grad
                v_hat_target = pseudo_grad.square()
                v_hat_new = (1.0 - rho_v) * v_hat + rho_v * v_hat_target

                state["exp_avg"].copy_(m_hat_new * bias_correction1)
                state["exp_avg_sq"].copy_(v_hat_new.clamp_min(0.0) * bias_correction2)
        if start != target_direction.numel():
            raise ValueError("State alignment did not consume the full target direction.")






def correct_optimizer_state_from_direction(
    network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    target_direction: torch.Tensor,
    rho_m: float,
    rho_v: float,
) -> None:
    """Align Adam first- and second-moment state with the executed direction."""
    correct_adam_consistent_state_from_direction(
        network=network,
        optimizer=optimizer,
        target_direction=target_direction,
        rho_m=rho_m,
        rho_v=rho_v,
    )


def get_cos_similarity(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
    """
    Calculates the cosine angle between two vectors.

    Args:
        vector1 (torch.Tensor): The first vector.
        vector2 (torch.Tensor): The second vector.

    Returns:
        torch.Tensor: The cosine angle between the two vectors.
    """
    with torch.no_grad():
        return torch.dot(vector1, vector2) / vector1.norm() / vector2.norm()


def unit_vector(vector: torch.Tensor, warn_zero: bool = False) -> torch.Tensor:
    """
    Compute the unit vector of a given tensor.

    Parameters:
        vector (torch.Tensor): The input tensor.
        warn_zero (bool): Whether to print a warning when the input tensor is zero. default: False

    Returns:
        torch.Tensor: The unit vector of the input tensor.
    """
    with torch.no_grad():
        if vector.norm() == 0:
            if warn_zero:
                print("Detected zero vector when doing normalization.")
            return torch.zeros_like(vector)
        else:
            return vector / vector.norm()


def transfer_coef_double(
    weights: torch.tensor,
    unit_vec_1: torch.tensor,
    unit_vec_2: torch.tensor,
    or_unit_vec_1: torch.tensor,
    or_unit_vec_2: torch.tensor,
) -> tuple:
    """
    Transfer the angle weights to a length coefficient for ConFIG method with two vectors.

    This function will return a coefficient matrix [c_1,c_2] so that
    $$
    \frac{(c_1 o_1+c_2 o_2)\dot \mathbf{v}_1}{(c_1 o_1+c_2 o_2) \dot \mathbf{v}_2}
    =\frac{w_1}{w_2}
    $$
    where w_1 and w_2 are the angle weights,
    v_1 and v_2 are unit vectors of the two gradients,
    and o_1 and o_2 are the unit orthogonal components of the two gradients.

    Args:
        weights (torch.tensor): Angle weights for the two vectors. It should have shape (2,).
        unit_vecs (torch.tensor): Unit vectors of the orthogonal components. It should have shape (2,k) where k is the length of the vector.

    Returns:
        tuple: The coefficients for the two vectors.

    Raises:
        ValueError: If the number of weights or unit vectors is not equal to 2.
    """

    return (
        torch.dot(or_unit_vec_2, unit_vec_1)
        / (weights[0] / weights[1] * torch.dot(or_unit_vec_1, unit_vec_2)),
        1,
    )
