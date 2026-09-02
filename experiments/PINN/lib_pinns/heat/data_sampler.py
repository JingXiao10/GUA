from typing import Literal, Union

import numpy as np
import torch
from scipy.stats.qmc import LatinHypercube

from .simulation_paras import *


class HeatSamplerBase:
    def __init__(
        self,
        n_internal: int,
        n_initial: int,
        n_boundary: int,
        device: Union[str, torch.device],
        update_data: bool = False,
        seed: int = 21339,
        x_start: float = X_START,
        x_end: float = X_END,
        y_start: float = Y_START,
        y_end: float = Y_END,
        simulation_time: float = SIMULATION_TIME,
    ) -> None:
        self.fake_data = [0]
        self.n_internal = n_internal
        self.n_initial = n_initial
        self.n_boundary = n_boundary
        self.device = device
        self.update_data = update_data
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.simulation_time = simulation_time
        self.rng = np.random.default_rng(seed)

        if self.update_data:
            self.sample_initial_boundary = self._sample_initial_boundary
            self.sample_initial = self._sample_initial
            self.sample_boundary = self._sample_boundary
            self.sample_internal = self._sample_internal
        else:
            x_b, y_b, t_b, v_b = self._sample_boundary()
            self.sample_boundary = lambda: (x_b, y_b, t_b, v_b)
            x_i, y_i, t_i, v_i = self._sample_initial()
            self.sample_initial = lambda: (x_i, y_i, t_i, v_i)
            self.sample_initial_boundary = lambda: (
                torch.cat([x_i, x_b]),
                torch.cat([y_i, y_b]),
                torch.cat([t_i, t_b]),
                torch.cat([v_i, v_b]),
            )
            x_int, y_int, t_int = self._sample_internal()
            self.sample_internal = lambda: (x_int, y_int, t_int)

    def _choice(self, high, size):
        indices = self.rng.integers(0, high, size=size)
        return torch.from_numpy(indices).long().to(self.device)

    def _sample_initial_boundary(self):
        with torch.no_grad():
            x_b, y_b, t_b, value_b = self._sample_boundary()
            x_i, y_i, t_i, value_i = self._sample_initial()
            return (
                torch.cat([x_i, x_b]),
                torch.cat([y_i, y_b]),
                torch.cat([t_i, t_b]),
                torch.cat([value_i, value_b]),
            )

    def _sample_initial(self):
        with torch.no_grad():
            x = torch.tensor(
                self.rng.random(self.n_initial),
                device=self.device,
                dtype=torch.float32,
            ) * (self.x_end - self.x_start) + self.x_start
            y = torch.tensor(
                self.rng.random(self.n_initial),
                device=self.device,
                dtype=torch.float32,
            ) * (self.y_end - self.y_start) + self.y_start
            t = torch.zeros_like(x)
            value = torch.sin(20.0 * torch.pi * x) * torch.sin(torch.pi * y)
            return x, y, t, value

    def _sample_boundary(self):
        with torch.no_grad():
            side = torch.randint(0, 4, (self.n_boundary,), device=self.device)
            s = torch.rand(self.n_boundary, device=self.device)
            x = torch.empty(self.n_boundary, device=self.device)
            y = torch.empty(self.n_boundary, device=self.device)
            left = side == 0
            right = side == 1
            bottom = side == 2
            top = side == 3
            x[left] = self.x_start
            y[left] = s[left] * (self.y_end - self.y_start) + self.y_start
            x[right] = self.x_end
            y[right] = s[right] * (self.y_end - self.y_start) + self.y_start
            x[bottom] = s[bottom] * (self.x_end - self.x_start) + self.x_start
            y[bottom] = self.y_start
            x[top] = s[top] * (self.x_end - self.x_start) + self.x_start
            y[top] = self.y_end
            t = torch.rand(self.n_boundary, device=self.device) * self.simulation_time
            value = torch.zeros_like(t)
            return x, y, t, value

    def _sample_internal(self):
        raise NotImplementedError

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.fake_data[idx]


class HeatSamplerLH(HeatSamplerBase):
    def __init__(self, *args, seed: int = 21339, **kwargs) -> None:
        self.random_engine_internal = LatinHypercube(d=3, seed=seed)
        super().__init__(*args, seed=seed, **kwargs)

    def _sample_internal(self):
        sample = self.random_engine_internal.random(n=self.n_internal)
        x = torch.tensor(sample[:, 0], device=self.device, dtype=torch.float32)
        y = torch.tensor(sample[:, 1], device=self.device, dtype=torch.float32)
        t = torch.tensor(sample[:, 2], device=self.device, dtype=torch.float32)
        x = x * (self.x_end - self.x_start) + self.x_start
        y = y * (self.y_end - self.y_start) + self.y_start
        t = t * self.simulation_time
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True
        return x, y, t


class HeatSamplerMC(HeatSamplerBase):
    def _sample_internal(self):
        x = torch.rand(self.n_internal, device=self.device) * (self.x_end - self.x_start) + self.x_start
        y = torch.rand(self.n_internal, device=self.device) * (self.y_end - self.y_start) + self.y_start
        t = torch.rand(self.n_internal, device=self.device) * self.simulation_time
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True
        return x, y, t


def HeatSampler(
    n_internal: int,
    n_initial: int,
    n_boundary: int,
    device: Union[str, torch.device],
    update_data: bool = False,
    seed: int = 21339,
    data_sampler: Literal["latin_hypercube", "monte_carlo"] = "latin_hypercube",
    x_start: float = X_START,
    x_end: float = X_END,
    y_start: float = Y_START,
    y_end: float = Y_END,
    simulation_time: float = SIMULATION_TIME,
):
    sampler_cls = HeatSamplerLH if data_sampler == "latin_hypercube" else HeatSamplerMC
    return sampler_cls(
        n_internal=n_internal,
        n_initial=n_initial,
        n_boundary=n_boundary,
        device=device,
        update_data=update_data,
        seed=seed,
        x_start=x_start,
        x_end=x_end,
        y_start=y_start,
        y_end=y_end,
        simulation_time=simulation_time,
    )


class HeatValidationDataSet:
    def __init__(self) -> None:
        self.xy, self.times, self.values = heat_validation_grid()


class HeatValidationDataLoader:
    def __init__(
        self,
        validation_dataset: HeatValidationDataSet,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        self.fake_data = [0]
        self.xy = torch.from_numpy(validation_dataset.xy).float().to(device)
        self.times = torch.from_numpy(validation_dataset.times).float().to(device)
        self.values = torch.from_numpy(validation_dataset.values).float().to(device)
        self.xs = self.xy[:, 0].unsqueeze(1).repeat(1, self.times.numel())
        self.ys = self.xy[:, 1].unsqueeze(1).repeat(1, self.times.numel())
        self.ts = self.times.unsqueeze(0).repeat(self.xy.shape[0], 1)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.fake_data[idx]
