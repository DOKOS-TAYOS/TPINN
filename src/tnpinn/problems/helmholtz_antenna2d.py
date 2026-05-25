from __future__ import annotations

from typing import Any

import torch

from tnpinn.pinn.derivatives import ensure_requires_grad, laplacian
from tnpinn.problems.base import ModelLike, PhysicsProblem, uniform_box


class HelmholtzAntenna2D(PhysicsProblem):
    def __init__(self, config: dict[str, Any]) -> None:
        problem = config.get("problem", {})
        domain = problem.get("domain", {})
        self.x_bounds = tuple(domain.get("x", [-1.0, 1.0]))
        self.y_bounds = tuple(domain.get("y", [-1.0, 1.0]))
        self.k = float(problem.get("k", 3.141592653589793))
        self.source_sigma = float(problem.get("source_sigma", 0.2))
        super().__init__("helmholtz_antenna2d", 2, 2, ("x", "y"))

    def source(self, coords: torch.Tensor) -> torch.Tensor:
        radius2 = coords[:, :1].square() + coords[:, 1:2].square()
        real = torch.exp(-radius2 / (2.0 * self.source_sigma**2))
        imag = torch.zeros_like(real)
        return torch.cat([real, imag], dim=1)

    def sample_collocation(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return {"coords": uniform_box(n, (self.x_bounds, self.y_bounds), device, dtype)}

    def sample_boundary(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        per_side = max(n // 4, 1)
        y = -1.0 + 2.0 * torch.rand(per_side, 1, device=device, dtype=dtype)
        x = -1.0 + 2.0 * torch.rand(per_side, 1, device=device, dtype=dtype)
        neg = -torch.ones_like(x)
        pos = torch.ones_like(x)
        coords = torch.cat(
            [
                torch.cat([neg, y], dim=1),
                torch.cat([pos, y], dim=1),
                torch.cat([x, neg], dim=1),
                torch.cat([x, pos], dim=1),
            ],
            dim=0,
        )[:n]
        return {
            "coords": coords,
            "target": torch.zeros(coords.shape[0], 2, device=device, dtype=dtype),
        }

    def residual(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = ensure_requires_grad(batch["coords"])
        u = model(coords)
        source = self.source(coords)
        residuals = []
        for channel in range(2):
            values = u[:, channel : channel + 1]
            residuals.append(
                laplacian(values, coords, dims=(0, 1))
                + self.k**2 * values
                - source[:, channel : channel + 1]
            )
        return torch.cat(residuals, dim=1)

    def make_eval_grid(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        axis = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        return {"coords": coords, "shape": torch.tensor([n, n], device=device)}
