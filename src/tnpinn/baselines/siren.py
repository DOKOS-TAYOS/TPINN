from __future__ import annotations

import math

import torch


class SineLayer(torch.nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, omega0: float = 30.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.omega0 = omega0
        bound = 1.0 / in_dim if is_first else math.sqrt(6.0 / in_dim) / omega0
        torch.nn.init.uniform_(self.linear.weight, -bound, bound)
        torch.nn.init.uniform_(self.linear.bias, -bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * self.linear(coords))


class SIREN(torch.nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dim: int = 64, n_layers: int = 3
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [
            SineLayer(input_dim, hidden_dim, is_first=True),
        ]
        for _ in range(max(n_layers - 1, 0)):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
