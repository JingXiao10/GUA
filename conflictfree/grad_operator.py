# usr/bin/python3
# -*- coding: UTF-8 -*-
import torch
import inspect
from typing import Optional, Sequence, Union
from .utils import *
from .weight_model import *
from .length_model import *
import numpy as np
from scipy.optimize import minimize


def _nan_to_num(tensor: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(tensor, value)
    return torch.where(
        torch.isfinite(tensor),
        tensor,
        torch.full_like(tensor, value),
    )


def _pinv(matrix: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "pinv"):
        return torch.linalg.pinv(matrix)
    return torch.pinverse(matrix)


def _solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "solve"):
        return torch.linalg.solve(matrix, rhs)
    solution, _ = torch.solve(rhs.unsqueeze(-1), matrix)
    return solution.squeeze(-1)


def _lstsq(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    if hasattr(torch, "linalg") and hasattr(torch.linalg, "lstsq"):
        return torch.linalg.lstsq(matrix, rhs).solution
    if matrix.shape[0] < matrix.shape[1]:
        gram = matrix @ matrix.t()
        scale = gram.diag().abs().mean()
        eps = scale * 1e-8 + 1e-12
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        small_system = gram + eps * eye
        if small_system.is_cuda:
            coeffs = _solve(small_system.cpu(), rhs.cpu()).to(matrix.device)
        else:
            coeffs = _solve(small_system, rhs)
        return matrix.t() @ coeffs
    solution, _ = torch.lstsq(rhs.unsqueeze(-1), matrix)
    return solution[: matrix.shape[1]].squeeze(-1)


def ConFIG_update_double(
    grad_1: torch.Tensor,
    grad_2: torch.Tensor,
    weight_model: WeightModel = EqualWeight(),
    length_model: LengthModel = ProjectionLength(),
    losses: Optional[Sequence] = None,
) -> torch.Tensor:
    """
    ConFIG update for two gradients where no inverse calculation is needed.

    Args:
        grad_1 (torch.Tensor): The first gradient.
        grad_2 (torch.Tensor): The second gradient.
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient.
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        losses (Optional[Sequence], optional): The losses associated with the gradients.
            The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
            you can set this value as None. Defaults to None.

    Returns:
        torch.Tensor: The final update gradient.

    Examples:
        ```python
        from conflictfree.grad_operator import ConFIG_update_double
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector_para_based
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        for input_i in dataset:
            grads=[] # we record gradients rather than losses
            for loss_fn in [loss_fn1, loss_fn2]:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network)) #get loss-specfic gradient
            g_config=ConFIG_update_double(grads) # calculate the conflict-free direction
            apply_gradient_vector_para_based(network,g_config)
            optimizer.step()
        ```

    """
    with torch.no_grad():
        norm_1 = grad_1.norm()
        norm_2 = grad_2.norm()
        unit_1 = grad_1 / norm_1
        unit_2 = grad_2 / norm_2
        cos_angle = get_cos_similarity(grad_1, grad_2)
        or_2 = grad_1 - norm_1 * cos_angle * unit_2
        or_1 = grad_2 - norm_2 * cos_angle * unit_1
        unit_or1 = unit_vector(or_1)
        unit_or2 = unit_vector(or_2)
        coef_1, coef_2 = transfer_coef_double(
            weight_model.get_weights(
                gradients=torch.stack([grad_1, grad_2]),
                losses=losses,
                device=grad_1.device,
            ),
            unit_1,
            unit_2,
            unit_or1,
            unit_or2,
        )
        best_direction = coef_1 * unit_or1 + coef_2 * unit_or2
        return length_model.rescale_length(
            target_vector=best_direction,
            gradients=torch.stack([grad_1, grad_2]),
            losses=losses,
        )


def ConFIG_update(
    grads: Union[torch.Tensor, Sequence[torch.Tensor]],
    weight_model: WeightModel = EqualWeight(),
    length_model: LengthModel = ProjectionLength(),
    use_least_square: bool = True,
    losses: Optional[Sequence] = None,
) -> torch.Tensor:
    """
    Performs the standard ConFIG update step.

    Args:
        grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
            It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient.
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        use_least_square (bool, optional): Whether to use the least square method for calculating the best direction.
            If set to False, we will directly calculate the pseudo-inverse of the gradient matrix. See `torch.linalg.pinv` and `torch.linalg.lstsq` for more details.
            Recommended to set to True. Defaults to True.
        losses (Optional[Sequence], optional): The losses associated with the gradients.
            The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
            you can set this value as None. Defaults to None.

    Returns:
        torch.Tensor: The final update gradient.

    Examples:
        ```python
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector_para_based
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        for input_i in dataset:
            grads=[] # we record gradients rather than losses
            for loss_fn in loss_fns:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network)) #get loss-specfic gradient
            g_config=ConFIG_update(grads) # calculate the conflict-free direction
            apply_gradient_vector_para_based(network,g_config)
            optimizer.step()
        ```
    """
    if not isinstance(grads, torch.Tensor):
        grads = torch.stack(grads)
    with torch.no_grad():
        weights = weight_model.get_weights(
            gradients=grads, losses=losses, device=grads.device
        )
        units = _nan_to_num((grads / (grads.norm(dim=1)).unsqueeze(1)), 0)
        if use_least_square:
            best_direction = _lstsq(units, weights)
        else:
            best_direction = _pinv(units) @ weights
        return length_model.rescale_length(
            target_vector=best_direction,
            gradients=grads,
            losses=losses,
        )


class GradientOperator:
    """
    A base class that represents a gradient operator.

    """

    def __init__(self):
        pass
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient based on the given gradients and losses.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients.
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient.

        Raises:
            NotImplementedError: If the method is not implemented.

        """
        raise NotImplementedError("calculate_gradient method must be implemented")

class ConFIGOperator(GradientOperator):
    """
    Operator for the ConFIG algorithm.

    Args:
        weight_model (WeightModel, optional): The weight model for calculating the direction weights.
            Defaults to EqualWeight(), which will make the final update gradient not biased towards any gradient.
        length_model (LengthModel, optional): The length model for rescaling the length of the final gradient.
            Defaults to ProjectionLength(), which will project each gradient vector onto the final gradient vector to get the final length.
        allow_simplified_model (bool, optional): Whether to allow simplified model for calculating the gradient.
            If set to True, will use simplified form of ConFIG method when there are only two losses (ConFIG_update_double). Defaults to True.
        use_least_square (bool, optional): Whether to use the least square method for calculating the best direction.
            If set to False, we will directly calculate the pseudo-inverse of the gradient matrix. See `torch.linalg.pinv` and `torch.linalg.lstsq` for more details.
            Recommended to set to True. Defaults to True.

    Examples:
        ```python
        from conflictfree.grad_operator import ConFIGOperator
        from conflictfree.utils import get_gradient_vector,apply_gradient_vector_para_based
        optimizer=torch.Adam(network.parameters(),lr=1e-3)
        operator=ConFIGOperator() # initialize operator
        for input_i in dataset:
            grads=[]
            for loss_fn in loss_fns:
                optimizer.zero_grad()
                loss_i=loss_fn(input_i)
                loss_i.backward()
                grads.append(get_gradient_vector(network))
            g_config=operator.calculate_gradient(grads) # calculate the conflict-free direction
            apply_gradient_vector_para_based(network,g_config)
            optimizer.step()
        ```

    """

    def __init__(
        self,
        weight_model: WeightModel = EqualWeight(),
        length_model: LengthModel = ProjectionLength(),
        allow_simplified_model: bool = True,
        use_least_square: bool = True,
    ):
        super().__init__()
        self.weight_model = weight_model
        self.length_model = length_model
        self.allow_simplified_model = allow_simplified_model
        self.use_least_square = use_least_square

    # Reference:
    # @inproceedings{liu2025config,
    #   title={ConFIG: Towards Conflict-free Training of Physics Informed Neural Networks},
    #   author={Liu, Qiang and Chu, Mengyu and Thuerey, Nils},
    #   booktitle={The Thirteenth International Conference on Learning Representations},
    #   year={2025},
    #   url={https://arxiv.org/abs/2408.11104}
    # }
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the ConFIG algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): The losses associated with the gradients.
                The losses will be passed to the weight and length model. If your weight/length model doesn't require loss information,
                you can set this value as None. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)
        if grads.shape[0] == 2 and self.allow_simplified_model:
            return ConFIG_update_double(
                grads[0],
                grads[1],
                weight_model=self.weight_model,
                length_model=self.length_model,
                losses=losses,
            )
        else:
            return ConFIG_update(
                grads,
                weight_model=self.weight_model,
                length_model=self.length_model,
                use_least_square=self.use_least_square,
                losses=losses,
            )


class PCGradOperator(GradientOperator):
    """
    PCGradOperator class represents a gradient operator for PCGrad algorithm.
    """

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    # Reference:
    # @inproceedings{yu2020gradient,
    #   title={Gradient Surgery for Multi-Task Learning},
    #   author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey
    #     and Hausman, Karol and Finn, Chelsea},
    #   booktitle={Advances in Neural Information Processing Systems},
    #   volume={33},
    #   year={2020},
    #   url={https://arxiv.org/abs/2001.06782}
    # }
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the PCGrad algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using PCGrad method.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)
        with torch.no_grad():
            grads_pc = torch.clone(grads)
            length = grads.shape[0]
            for i in range(length):
                shuffled_indices = torch.randperm(length, device=grads.device)
                for j in shuffled_indices:
                    j = j.item()
                    if j == i:
                        continue

                    dot = grads_pc[i].dot(grads[j])
                    if dot < 0:
                        grads_pc[i] -= dot * grads[j] / (
                            grads[j].norm() ** 2 + self.eps
                        )
            return torch.sum(grads_pc, dim=0)


class IMTLGOperator(GradientOperator):
    """
    Gradient operator for the IMTL-G algorithm.
    """

    # Reference:
    # @inproceedings{liu2021imtl,
    #   title={Towards Impartial Multi-Task Learning},
    #   author={Liu, Liyang and Li, Yi and Kuang, Zhanghui and Xue, Jing-Hao
    #     and Chen, Yimin and Yang, Wenming and Liao, Qingmin and Zhang, Wayne},
    #   booktitle={International Conference on Learning Representations},
    #   year={2021},
    #   url={https://openreview.net/forum?id=IMPnRXEWpvr}
    # }
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the IMTL-G algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors (at dim 0) or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using IMTL-G method.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)
        with torch.no_grad():
            ut_norm = grads / grads.norm(dim=1).unsqueeze(1)
            ut_norm = _nan_to_num(ut_norm, 0)
            ut = torch.stack(
                [ut_norm[0] - ut_norm[i + 1] for i in range(grads.shape[0] - 1)], dim=0
            ).T
            d = torch.stack(
                [grads[0] - grads[i + 1] for i in range(grads.shape[0] - 1)], dim=0
            )
            at = grads[0] @ ut @ _pinv(d @ ut)
            return (1 - torch.sum(at)) * grads[0] + torch.sum(
                at.unsqueeze(1) * grads[1:], dim=0
            )




class CAGradOperator(GradientOperator):
    """
    CAGradOperator class represents a gradient operator for CAGrad algorithm.

    CAGrad: Conflict-Averse Gradient Descent for Multi-task Learning.

    Args:
        alpha (float, optional): Conflict-aversion coefficient. Defaults to 0.5.
        rescale (int, optional): Rescale mode in the official CAGrad implementation.
            0 means no rescale, 1 means dividing by (1 + alpha ** 2), and 2 means
            dividing by (1 + alpha). Defaults to 1.
        scale_to_sum (bool, optional): Whether to multiply the final gradient by
            the number of tasks. If set to True, alpha=0 reduces to the summed
            gradient; otherwise alpha=0 reduces to the averaged gradient. Defaults to True.
        eps (float, optional): Numerical stability constant. Defaults to 1e-8.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        rescale: int = 1,
        scale_to_sum: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.rescale = rescale
        self.scale_to_sum = scale_to_sum
        self.eps = eps

    # Reference:
    # @inproceedings{liu2021cagrad,
    #   title={Conflict-Averse Gradient Descent for Multi-task Learning},
    #   author={Liu, Bo and Liu, Xingchao and Jin, Xiaojie and Stone, Peter
    #     and Liu, Qiang},
    #   booktitle={Advances in Neural Information Processing Systems},
    #   volume={34},
    #   year={2021},
    #   url={https://arxiv.org/abs/2110.14048}
    # }
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using the CAGrad algorithm.

        Args:
            grads (Union[torch.Tensor,Sequence[torch.Tensor]]): The gradients to update.
                It can be a stack of gradient vectors at dim 0 or a sequence of gradient vectors.
            losses (Optional[Sequence], optional): This parameter should not be set for current operator. Defaults to None.

        Returns:
            torch.Tensor: The calculated gradient using CAGrad method.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)

        if grads.dim() != 2:
            raise ValueError(
                f"CAGradOperator expects grads with shape "
                f"[num_tasks, grad_dim], but got {tuple(grads.shape)}"
            )

        if self.alpha < 0:
            raise ValueError("alpha should be non-negative.")

        if self.rescale not in [0, 1, 2]:
            raise ValueError("rescale should be 0, 1, or 2.")

        with torch.no_grad():
            num_tasks = grads.shape[0]

            if num_tasks == 0:
                raise ValueError("CAGradOperator received zero gradients.")

            if num_tasks == 1:
                return grads[0]

            GG = grads.mm(grads.t()).detach().cpu().double()
            g0 = grads.mean(dim=0)

            if self.alpha == 0:
                return g0 * num_tasks if self.scale_to_sum else g0

            g0_norm = torch.sqrt(torch.clamp(GG.mean(), min=0.0) + self.eps)

            x_start = np.ones(num_tasks, dtype=np.float64) / num_tasks

            bounds = tuple((0.0, 1.0) for _ in range(num_tasks))

            constraints = ({
                "type": "eq",
                "fun": lambda x: 1.0 - np.sum(x),
            },)

            A = GG.numpy()
            b = x_start.copy()
            c = float(self.alpha * g0_norm.item() + self.eps)

            def objfn(x: np.ndarray) -> float:
                x = x.reshape(1, num_tasks)
                first_term = x.dot(A).dot(b.reshape(num_tasks, 1))
                second_term = c * np.sqrt(
                    x.dot(A).dot(x.reshape(num_tasks, 1)) + self.eps
                )

                return float((first_term + second_term).sum())

            res = minimize(
                objfn,
                x_start,
                bounds=bounds,
                constraints=constraints,
                method="SLSQP",
            )

            if not res.success or not np.isfinite(res.x).all():
                raise RuntimeError(f"CAGrad SLSQP failed: {res.message}")
            w_cpu = res.x

            w = torch.as_tensor(
                w_cpu,
                dtype=grads.dtype,
                device=grads.device,
            )

            gw = torch.sum(grads * w.view(-1, 1), dim=0)

            gw_norm = gw.norm()

            if gw_norm.item() <= self.eps:
                g = g0
            else:
                lmbda = c / (gw_norm + self.eps)
                g = g0 + lmbda * gw

            if self.rescale == 0:
                pass
            elif self.rescale == 1:
                g = g / (1.0 + self.alpha ** 2)
            elif self.rescale == 2:
                g = g / (1.0 + self.alpha)

            if self.scale_to_sum:
                g = g * num_tasks

            return g

class UPGradOperator(GradientOperator):
    """
    Operator for UPGrad algorithm using TorchJD official implementation.
    """

    def __init__(
        self,
        norm_eps: float = 1e-4,
        reg_eps: float = 1e-4,
        scale_to_sum: bool = False,
    ):
        super().__init__()
        from torchjd.aggregation import UPGrad

        kwargs = {}
        signature = inspect.signature(UPGrad)
        if "norm_eps" in signature.parameters:
            kwargs["norm_eps"] = norm_eps
        if "reg_eps" in signature.parameters:
            kwargs["reg_eps"] = reg_eps
        self.aggregator = UPGrad(**kwargs)
        self.scale_to_sum = scale_to_sum

    # Reference:
    # @article{quinton2024jacobian,
    #   title={Jacobian Descent for Multi-Objective Optimization},
    #   author={Quinton, Pierre and Rey, Val{\'e}rian},
    #   journal={arXiv preprint arXiv:2406.16232},
    #   year={2024},
    #   url={https://arxiv.org/abs/2406.16232}
    # }
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)

        if grads.dim() != 2:
            raise ValueError(
                f"UPGradOperator expects grads with shape "
                f"[num_tasks, grad_dim], but got {tuple(grads.shape)}"
            )

        with torch.no_grad():
            g = self.aggregator(grads)

            if self.scale_to_sum:
                g = g * grads.shape[0]

            return g

class AlignedMTLOperator(GradientOperator):
    """
    AlignedMTLOperator using TorchJD official implementation.

    Aligned-MTL: Independent Component Alignment for Multi-Task Learning.

    Expected gradient shape:
        grads.shape == [num_tasks, grad_dim]

    Args:
        pref_vector:
            Task preference vector. If None, TorchJD uses equal preference:
            [1 / num_tasks, ..., 1 / num_tasks].

        scale_mode:
            Scaling mode used by Aligned-MTL.
            Options:
                "min"
                "median"
                "rmse"

        scale_to_sum:
            If False, keep TorchJD official aggregation scale.
            If True, multiply the final gradient by num_tasks, making the scale
            closer to sum-gradient methods.

    """

    def __init__(
        self,
        pref_vector: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        scale_mode: str = "rmse",
        scale_to_sum: bool = False,
    ):
        super().__init__()
        try:
            from torchjd.aggregation import AlignedMTL
        except ImportError as e:
            raise ImportError(
                "AlignedMTLOperator requires TorchJD. "
                "Please install it with: pip install torchjd"
            ) from e

        if scale_mode not in ["min", "median", "rmse"]:
            raise ValueError(
                "scale_mode should be one of: 'min', 'median', 'rmse'."
            )

        self._aggregator_cls = AlignedMTL
        self.pref_vector = pref_vector
        self.scale_mode = scale_mode
        self.scale_to_sum = scale_to_sum

    # Reference:
    # @inproceedings{senushkin2023alignedmtl,
    #   title={Independent Component Alignment for Multi-Task Learning},
    #   author={Senushkin, Dmitry and Patakin, Nikolay and Kuznetsov, Arseny
    #     and Konushin, Anton},
    #   booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
    #     Pattern Recognition},
    #   pages={20083--20093},
    #   year={2023},
    #   url={https://arxiv.org/abs/2305.19000}
    # }
    def calculate_gradient(
        self,
        grads: Union[torch.Tensor, Sequence[torch.Tensor]],
        losses: Optional[Sequence] = None,
    ) -> torch.Tensor:
        """
        Calculates the gradient using Aligned-MTL.

        Args:
            grads:
                Task-specific gradients.
                Expected shape after stacking:
                    [num_tasks, grad_dim]

            losses:
                Not used by Aligned-MTL. Kept for compatibility with the
                GradientOperator interface.

        Returns:
            torch.Tensor:
                The final Aligned-MTL update gradient.
        """
        if not isinstance(grads, torch.Tensor):
            grads = torch.stack(grads)

        if grads.dim() != 2:
            raise ValueError(
                f"AlignedMTLOperator expects grads with shape "
                f"[num_tasks, grad_dim], but got {tuple(grads.shape)}"
            )

        with torch.no_grad():
            num_tasks = grads.shape[0]

            if num_tasks == 0:
                raise ValueError("AlignedMTLOperator received zero gradients.")

            if num_tasks == 1:
                return grads[0]

            if self.pref_vector is None:
                pref_vector = None
            else:
                pref_vector = torch.as_tensor(
                    self.pref_vector,
                    dtype=grads.dtype,
                    device=grads.device,
                )

                if pref_vector.dim() != 1:
                    raise ValueError(
                        f"pref_vector should be 1D, "
                        f"but got shape {tuple(pref_vector.shape)}"
                    )

                if pref_vector.numel() != num_tasks:
                    raise ValueError(
                        f"pref_vector length should match num_tasks. "
                        f"Got {pref_vector.numel()} and num_tasks={num_tasks}."
                    )

            aggregator = self._aggregator_cls(
                pref_vector=pref_vector,
                scale_mode=self.scale_mode,
            )

            g = aggregator(grads)

            if self.scale_to_sum:
                g = g * num_tasks

            return g
