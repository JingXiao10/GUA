import numpy as np
import torch

from .data_sampler import Poisson5DValidationDataSet


def run_test(network, n_point=20000, device="cuda"):
    dataset = Poisson5DValidationDataSet(n_point=n_point)
    x = torch.from_numpy(dataset.x).float().to(device)
    target = dataset.u
    network.to(device)
    network.eval()
    with torch.no_grad():
        prediction = network(x).detach().cpu().numpy()
    mse = (target - prediction) ** 2
    mse_value = np.mean(mse)
    relative_l2 = np.linalg.norm(prediction.reshape(-1) - target.reshape(-1)) / np.linalg.norm(target.reshape(-1))
    return float(mse_value), mse, prediction, target, float(relative_l2)
