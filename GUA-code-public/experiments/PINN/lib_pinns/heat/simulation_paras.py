import numpy as np


X_START = 0.0
X_END = 1.0
Y_START = 0.0
Y_END = 1.0
SIMULATION_TIME = 5.0
HEAT_DIFFUSIVITY_X = 1.0 / (500.0 * np.pi) ** 2
HEAT_DIFFUSIVITY_Y = 1.0 / np.pi**2
HEAT_VALIDATION_N_X = 121
HEAT_VALIDATION_N_Y = 121
HEAT_VALIDATION_N_T = 51


def heat_exact_solution(x, y, t):
    decay = (HEAT_DIFFUSIVITY_X * (20.0 * np.pi) ** 2) + (HEAT_DIFFUSIVITY_Y * np.pi**2)
    return np.sin(20.0 * np.pi * x) * np.sin(np.pi * y) * np.exp(-decay * t)


def heat_validation_grid(
    n_x=HEAT_VALIDATION_N_X,
    n_y=HEAT_VALIDATION_N_Y,
    n_t=HEAT_VALIDATION_N_T,
):
    xs = np.linspace(X_START, X_END, n_x, dtype=np.float32)
    ys = np.linspace(Y_START, Y_END, n_y, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    times = np.linspace(0.0, SIMULATION_TIME, n_t, dtype=np.float32)
    values = heat_exact_solution(xy[:, 0:1], xy[:, 1:2], times[None, :]).astype(np.float32)
    return xy, times, values
