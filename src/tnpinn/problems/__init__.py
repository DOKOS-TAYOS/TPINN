from __future__ import annotations

from typing import Any

from tnpinn.problems.base import PhysicsProblem
from tnpinn.problems.heat2d import Heat2DLaplace, Heat2DTransient
from tnpinn.problems.helmholtz_antenna2d import HelmholtzAntenna2D
from tnpinn.problems.lorenz import LorenzProblem


def build_problem(config: dict[str, Any]) -> PhysicsProblem:
    name = str(config.get("problem", {}).get("name", ""))
    if name == "heat2d_laplace":
        return Heat2DLaplace(config)
    if name == "heat2d_transient":
        return Heat2DTransient(config)
    if name == "helmholtz_antenna2d":
        return HelmholtzAntenna2D(config)
    if name == "lorenz":
        return LorenzProblem(config)
    raise ValueError(f"Unknown problem: {name}")


__all__ = [
    "Heat2DLaplace",
    "Heat2DTransient",
    "HelmholtzAntenna2D",
    "LorenzProblem",
    "PhysicsProblem",
    "build_problem",
]
