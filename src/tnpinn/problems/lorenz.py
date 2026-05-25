from __future__ import annotations

from typing import Any

import torch

from tnpinn.pinn.derivatives import ensure_requires_grad, partial_derivative
from tnpinn.problems.base import ModelLike, PhysicsProblem


class LorenzProblem(PhysicsProblem):
    def __init__(self, config: dict[str, Any]) -> None:
        problem = config.get("problem", {})
        self.t_final = float(problem.get("t_final", problem.get("T", 2.0)))
        self.sigma = float(problem.get("sigma", 10.0))
        self.rho = float(problem.get("rho", 28.0))
        self.beta = float(problem.get("beta", 8.0 / 3.0))
        self.u0 = tuple(problem.get("u0", [1.0, 1.0, 1.0]))
        super().__init__("lorenz", 1, 3, ("t",))

    def sample_collocation(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        coords = torch.rand(n, 1, device=device, dtype=dtype) * self.t_final
        return {"coords": coords}

    def sample_initial(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        coords = torch.zeros(max(n, 1), 1, device=device, dtype=dtype)
        target = (
            torch.tensor(self.u0, device=device, dtype=dtype)
            .reshape(1, 3)
            .repeat(coords.shape[0], 1)
        )
        return {"coords": coords, "target": target}

    def residual(self, model: ModelLike, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        coords = ensure_requires_grad(batch["coords"])
        u = model(coords)
        x = u[:, 0:1]
        y = u[:, 1:2]
        z = u[:, 2:3]
        dx = partial_derivative(x, coords, 0)
        dy = partial_derivative(y, coords, 0)
        dz = partial_derivative(z, coords, 0)
        return torch.cat(
            [
                dx - self.sigma * (y - x),
                dy - x * (self.rho - z) + y,
                dz - x * y + self.beta * z,
            ],
            dim=1,
        )

    def make_eval_grid(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        coords = torch.linspace(0.0, self.t_final, n, device=device, dtype=dtype).reshape(-1, 1)
        return {"coords": coords}
