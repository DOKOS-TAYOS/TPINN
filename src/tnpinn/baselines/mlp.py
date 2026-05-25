from __future__ import annotations

import torch


class MLP(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 64, n_layers: int = 3
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for _ in range(max(n_layers, 1)):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(torch.nn.Tanh())
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
