from .simulation_paras import DIM, source_term_torch
from ..helpers import derivative


def physical_residual(u, x):
    x.grad = None
    laplacian = 0.0
    for dim in range(DIM):
        first = derivative(u, x, order=1)[:, dim]
        second = derivative(first, x, order=1)[:, dim]
        laplacian = laplacian + second
    return laplacian + source_term_torch(x)
