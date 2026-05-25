import torch

from tnpinn.problems import build_problem
from tnpinn.problems.base import PhysicsProblem


class ReferenceModel(torch.nn.Module):
    def __init__(self, problem: PhysicsProblem) -> None:
        super().__init__()
        self.problem = problem

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        reference = self.problem.reference_solution(coords)
        if reference is None:
            raise RuntimeError("reference solution is required")
        return reference


class ZeroLorenzModel(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros(coords.shape[0], 3, dtype=coords.dtype, device=coords.device)


def test_heat2d_reference_solution_has_small_laplace_residual() -> None:
    cfg = {"problem": {"name": "heat2d_laplace", "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}}
    problem = build_problem(cfg)
    coords = torch.rand(8, 2, dtype=torch.float64)
    residual = problem.residual(ReferenceModel(problem), {"coords": coords})

    assert residual.shape == (8, 1)
    assert residual.abs().max().item() < 1.0e-8


def test_heat2d_sampling_respects_domain_and_boundary() -> None:
    cfg = {"problem": {"name": "heat2d_laplace", "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}}
    problem = build_problem(cfg)
    collocation = problem.sample_collocation(16, torch.device("cpu"), torch.float64)["coords"]
    boundary = problem.sample_boundary(16, torch.device("cpu"), torch.float64)["coords"]

    assert torch.all((collocation >= 0.0) & (collocation <= 1.0))
    on_edge = (
        torch.isclose(boundary[:, 0], torch.tensor(0.0, dtype=torch.float64))
        | torch.isclose(boundary[:, 0], torch.tensor(1.0, dtype=torch.float64))
        | torch.isclose(boundary[:, 1], torch.tensor(0.0, dtype=torch.float64))
        | torch.isclose(boundary[:, 1], torch.tensor(1.0, dtype=torch.float64))
    )
    assert on_edge.all()


def test_lorenz_residual_shape() -> None:
    cfg = {"problem": {"name": "lorenz", "t_final": 1.0}}
    problem = build_problem(cfg)
    batch = {"coords": torch.linspace(0, 1, 5, dtype=torch.float64).reshape(-1, 1)}
    residual = problem.residual(ZeroLorenzModel(), batch)

    assert residual.shape == (5, 3)
