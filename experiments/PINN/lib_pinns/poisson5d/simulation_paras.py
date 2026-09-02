import numpy as np
import torch


DIM = 5
X_START = 0.0
X_END = 1.0
N_VALIDATION = 20000


def exact_solution_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(0.5 * np.pi * x).sum(dim=-1)


def source_term_torch(x: torch.Tensor) -> torch.Tensor:
    return 0.25 * np.pi**2 * exact_solution_torch(x)


def exact_solution_np(x: np.ndarray) -> np.ndarray:
    return np.sin(0.5 * np.pi * x).sum(axis=-1)
