import torch
import torch.nn as nn


class HeatNet(nn.Module):
    def __init__(self, channel_basics=50, n_layers=4, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ini_net = nn.Sequential(nn.Linear(3, channel_basics), nn.Tanh())
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Sequential(nn.Linear(channel_basics, channel_basics), nn.Tanh()))
        self.net = nn.Sequential(*layers)
        self.out_net = nn.Linear(channel_basics, 1)

    def forward(self, x, y, t):
        ini_shape = x.shape
        inputs = torch.stack([x.reshape(-1), y.reshape(-1), t.reshape(-1)], dim=-1)
        outputs = self.ini_net(inputs)
        outputs = self.net(outputs)
        outputs = self.out_net(outputs)
        return outputs.reshape(ini_shape)
