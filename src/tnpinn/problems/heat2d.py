from __future__ import annotations

import math
from typing import Any

import torch

from tnpinn.pinn.derivatives import ensure_requires_grad, laplacian, partial_derivative
from tnpinn.problems.base import ModelLike, PhysicsProblem, uniform_box


class Heat2DLaplace(PhysicsProblem):
    def __init__(self, config: dict[str, Any]) -> None:
        problem = config.get("problem", {})
        domain = problem.get("domain", {})
        self.x_bounds = tuple(domain.get("x", [0.0, 1.0]))
        self.y_bounds = tuple(domain.get("y", [0.0, 1.0]))
        default_boundary_mode = (
            "hard" if bool(problem.get("use_hard_constraints", False)) else "soft"
        )
        self.boundary_mode = str(problem.get("boundary_mode", default_boundary_mode))
        self.soft_boundary_with_hard = bool(problem.get("soft_boundary_with_hard", False))
        super().__init__("heat2d_laplace", 2, 1, ("x", "y"))

    def sample_collocation(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        coords = uniform_box(n, (self.x_bounds, self.y_bounds), device, dtype)
        return {"coords": coords}

    def sample_boundary(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        per_side = max(n // 4, 1)
        y = torch.rand(per_side, 1, device=device, dtype=dtype)
        x = torch.rand(per_side, 1, device=device, dtype=dtype)
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        coords = torch.cat(
            [
                torch.cat([zeros, y], dim=1),
                torch.cat([ones, y], dim=1),
                torch.cat([x, zeros], dim=1),
                torch.cat([x, ones], dim=1),
            ],
            dim=0,
        )[:n]
        return {"coords": coords, "target": self.reference_solution(coords)}

    def residual(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = ensure_requires_grad(batch["coords"])
        u = model(coords)[:, :1]
        return -laplacian(u, coords, dims=(0, 1))

    def boundary_loss(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = batch["coords"]
        if self.boundary_mode == "hard" and not self.soft_boundary_with_hard:
            return torch.zeros((), device=coords.device, dtype=coords.dtype)
        return super().boundary_loss(model, batch)

    def reference_solution(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords[:, :1]
        y = coords[:, 1:2]
        numerator = torch.sinh(math.pi * (1.0 - x))
        denominator = torch.sinh(torch.tensor(math.pi, dtype=coords.dtype, device=coords.device))
        return numerator / denominator * torch.sin(math.pi * y)

    def make_eval_grid(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        axis = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        return {"coords": coords, "shape": torch.tensor([n, n], device=device)}


class Heat2DTransient(PhysicsProblem):
    def __init__(self, config: dict[str, Any]) -> None:
        problem = config.get("problem", {})
        domain = problem.get("domain", {})
        self.x_bounds = tuple(domain.get("x", [0.0, 1.0]))
        self.y_bounds = tuple(domain.get("y", [0.0, 1.0]))
        self.t_bounds = tuple(domain.get("t", [0.0, float(problem.get("t_final", 1.0))]))
        self.alpha = float(problem.get("alpha", 0.1))
        super().__init__("heat2d_transient", 3, 1, ("x", "y", "t"))

    def sample_collocation(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        return {
            "coords": uniform_box(n, (self.x_bounds, self.y_bounds, self.t_bounds), device, dtype)
        }

    def sample_boundary(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        per_side = max(n // 4, 1)
        y = torch.rand(per_side, 1, device=device, dtype=dtype)
        x = torch.rand(per_side, 1, device=device, dtype=dtype)
        t = torch.rand(per_side, 1, device=device, dtype=dtype) * self.t_bounds[1]
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        coords = torch.cat(
            [
                torch.cat([zeros, y, t], dim=1),
                torch.cat([ones, y, t], dim=1),
                torch.cat([x, zeros, t], dim=1),
                torch.cat([x, ones, t], dim=1),
            ],
            dim=0,
        )[:n]
        target = torch.zeros(coords.shape[0], 1, device=device, dtype=dtype)
        target[:, :1] = torch.where(
            torch.isclose(coords[:, :1], torch.zeros((), device=device, dtype=dtype)),
            torch.sin(math.pi * coords[:, 1:2]),
            target,
        )
        return {"coords": coords, "target": target}

    def sample_initial(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        xy = uniform_box(n, (self.x_bounds, self.y_bounds), device, dtype)
        t = torch.zeros(n, 1, device=device, dtype=dtype)
        return {
            "coords": torch.cat([xy, t], dim=1),
            "target": torch.zeros(n, 1, device=device, dtype=dtype),
        }

    def residual(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = ensure_requires_grad(batch["coords"])
        u = model(coords)[:, :1]
        u_t = partial_derivative(u, coords, 2)
        return u_t - self.alpha * laplacian(u, coords, dims=(0, 1))

    def make_eval_grid(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        axis = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
        xx, yy = torch.meshgrid(axis, axis, indexing="ij")
        t = torch.full_like(xx.reshape(-1, 1), self.t_bounds[1])
        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), t], dim=1)
        return {"coords": coords, "shape": torch.tensor([n, n], device=device)}
