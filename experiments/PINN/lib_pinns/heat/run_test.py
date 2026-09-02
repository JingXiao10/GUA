import numpy as np
import torch

from .simulation_paras import *


def run_test(network, device="cuda"):
    xy_np, times_np, target = heat_validation_grid()
    xy = torch.from_numpy(xy_np).float().to(device)
    times = torch.from_numpy(times_np).float().to(device)

    network.to(device)
    network.eval()
    with torch.no_grad():
        xs = xy[:, 0].unsqueeze(1).repeat(1, times.numel())
        ys = xy[:, 1].unsqueeze(1).repeat(1, times.numel())
        ts = times.unsqueeze(0).repeat(xy.shape[0], 1)
        prediction = network(xs, ys, ts).detach().cpu().numpy()

    mse = (target - prediction) ** 2
    mse_value = np.mean(mse)
    relative_l2 = np.linalg.norm(prediction.reshape(-1) - target.reshape(-1)) / np.linalg.norm(
        target.reshape(-1)
    )
    return float(mse_value), mse, prediction, target, float(relative_l2)
