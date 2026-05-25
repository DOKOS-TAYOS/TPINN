from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ModelLike(Protocol):
    def __call__(self, coords: torch.Tensor) -> torch.Tensor: ...


@dataclass
class PhysicsProblem:
    name: str
    input_dim: int
    output_dim: int
    coordinates: tuple[str, ...]

    def sample_collocation(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def sample_boundary(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return {"coords": torch.empty(0, self.input_dim, device=device, dtype=dtype)}

    def sample_initial(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return {"coords": torch.empty(0, self.input_dim, device=device, dtype=dtype)}

    def residual(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def boundary_loss(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = batch["coords"]
        if coords.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        target = batch.get("target")
        if target is None:
            target = torch.zeros(
                coords.shape[0], self.output_dim, device=coords.device, dtype=coords.dtype
            )
        return torch.mean((model(coords) - target).square())

    def initial_loss(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = batch["coords"]
        if coords.numel() == 0:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        target = batch.get("target")
        if target is None:
            target = torch.zeros(
                coords.shape[0], self.output_dim, device=coords.device, dtype=coords.dtype
            )
        return torch.mean((model(coords) - target).square())

    def reference_solution(self, coords: torch.Tensor) -> torch.Tensor | None:
        return None

    def make_eval_grid(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return self.sample_collocation(n, device, dtype)


def uniform_box(
    n: int,
    bounds: tuple[tuple[float, float], ...],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    unit = torch.rand(n, len(bounds), device=device, dtype=dtype)
    lows = torch.tensor([b[0] for b in bounds], device=device, dtype=dtype)
    highs = torch.tensor([b[1] for b in bounds], device=device, dtype=dtype)
    return lows + (highs - lows) * unit
