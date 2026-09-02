from typing import Literal, Union

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

from .simulation_paras import *


EVALUATION_PROTOCOL = "pinnacle_uniform_grid_v1"


def pinnacle_uniform_grid(
    n_point: int = N_VALIDATION,
    x_start: float = X_START,
    x_end: float = X_END,
) -> np.ndarray:
    """Match PINNacle's deterministic Hypercube.uniform_points evaluation set."""
    if n_point <= 0:
        raise ValueError("n_point must be positive.")
    side_length = x_end - x_start
    if side_length <= 0:
        raise ValueError("x_end must be greater than x_start.")
    spacing = ((side_length**DIM) / n_point) ** (1.0 / DIM)
    points_per_axis = int(np.ceil(side_length / spacing))
    axis = np.linspace(
        x_start,
        x_end,
        num=points_per_axis + 1,
        endpoint=False,
        dtype=np.float32,
    )[1:]
    mesh = np.meshgrid(*([axis] * DIM), indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, DIM)


class Poisson5DSamplerBase:
    def __init__(
        self,
        n_internal: int,
        n_boundary: int,
        device: Union[str, torch.device],
        update_data: bool = False,
        seed: int = 21339,
        x_start: float = X_START,
        x_end: float = X_END,
    ) -> None:
        self.fake_data = [0]
        self.n_internal = n_internal
        self.n_boundary = n_boundary
        self.device = device
        self.update_data = update_data
        self.seed = seed
        self.x_start = x_start
        self.x_end = x_end
        self.rng = np.random.default_rng(seed)
        if self.update_data:
            self.sample_boundary = self._sample_boundary
            self.sample_internal = self._sample_internal
        else:
            x_b, u_b = self._sample_boundary()
            x_i = self._sample_internal()
            self.sample_boundary = lambda: (x_b, u_b)
            self.sample_internal = lambda: x_i

    def _sample_boundary(self):
        with torch.no_grad():
            x = torch.rand(self.n_boundary, DIM, device=self.device)
            x = x * (self.x_end - self.x_start) + self.x_start
            dims = torch.from_numpy(self.rng.integers(0, DIM, size=self.n_boundary)).long().to(self.device)
            sides = torch.from_numpy(self.rng.integers(0, 2, size=self.n_boundary)).float().to(self.device)
            x[torch.arange(self.n_boundary, device=self.device), dims] = sides * (self.x_end - self.x_start) + self.x_start
            values = exact_solution_torch(x)
            return x, values

    def _sample_internal(self):
        raise NotImplementedError

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.fake_data[idx]


class Poisson5DSamplerLH(Poisson5DSamplerBase):
    def __init__(self, *args, seed: int = 21339, **kwargs) -> None:
        self.random_engine_internal = LatinHypercube(d=DIM, seed=seed)
        super().__init__(*args, seed=seed, **kwargs)

    def _sample_internal(self):
        sample = self.random_engine_internal.random(n=self.n_internal)
        x = torch.tensor(sample, device=self.device, dtype=torch.float32)
        x = x * (self.x_end - self.x_start) + self.x_start
        x.requires_grad = True
        return x


class Poisson5DSamplerMC(Poisson5DSamplerBase):
    def _sample_internal(self):
        x = torch.rand(self.n_internal, DIM, device=self.device)
        x = x * (self.x_end - self.x_start) + self.x_start
        x.requires_grad = True
        return x


def Poisson5DSampler(
    n_internal: int,
    n_boundary: int,
    device: Union[str, torch.device],
    update_data: bool = False,
    seed: int = 21339,
    data_sampler: Literal["latin_hypercube", "monte_carlo"] = "latin_hypercube",
    x_start: float = X_START,
    x_end: float = X_END,
):
    sampler_cls = Poisson5DSamplerLH if data_sampler == "latin_hypercube" else Poisson5DSamplerMC
    return sampler_cls(
        n_internal=n_internal,
        n_boundary=n_boundary,
        device=device,
        update_data=update_data,
        seed=seed,
        x_start=x_start,
        x_end=x_end,
    )


class Poisson5DValidationDataSet:
    def __init__(
        self,
        n_point: int = N_VALIDATION,
        x_start: float = X_START,
        x_end: float = X_END,
    ) -> None:
        self.x = pinnacle_uniform_grid(
            n_point=n_point,
            x_start=x_start,
            x_end=x_end,
        )
        self.u = exact_solution_np(self.x).astype(np.float32)


class Poisson5DValidationDataLoader:
    def __init__(self, validation_dataset: Poisson5DValidationDataSet, device: Union[str, torch.device] = "cuda:0") -> None:
        self.fake_data = [0]
        self.xs = torch.from_numpy(validation_dataset.x).float().to(device)
        self.us = torch.from_numpy(validation_dataset.u).float().to(device)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.fake_data[idx]
