import torch
import torch.nn as nn

from .simulation_paras import DIM


class Poisson5DNet(nn.Module):
    def __init__(self, channel_basics=50, n_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net = nn.Sequential(nn.Linear(DIM, channel_basics), nn.Tanh())
        self.net = nn.Sequential(
            *[nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh()) for _ in range(n_layers)]
        )
        self.out_net = nn.Linear(channel_basics, 1)

    def forward(self, x):
        ini_shape = x.shape[:-1]
        outputs = self.ini_net(x.reshape(-1, DIM))
        outputs = self.net(outputs)
        outputs = self.out_net(outputs).squeeze(-1)
        return outputs.reshape(ini_shape)
