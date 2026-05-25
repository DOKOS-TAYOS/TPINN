from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch

from tnpinn.models import build_model
from tnpinn.pinn.derivatives import laplacian
from tnpinn.pinn.losses import compute_losses
from tnpinn.pinn.trainer import train
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


class RecordingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
        self.saw_requires_grad = False

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        self.saw_requires_grad = bool(coords.requires_grad)
        return self.weight * (coords[:, :1].pow(3) + coords[:, 1:2].pow(3))


def _laplace_config(**problem_overrides: object) -> dict:
    problem = {
        "name": "heat2d_laplace",
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    }
    problem.update(problem_overrides)
    return {
        "seed": 1234,
        "device": "cpu",
        "dtype": "float64",
        "problem": problem,
        "model": {
            "family": "baseline",
            "baseline": "mlp",
            "hidden_dim": 32,
            "n_layers": 2,
            "output_dim": 1,
        },
        "training": {
            "steps": 250,
            "n_collocation": 64,
            "n_boundary": 64,
            "n_initial": 0,
            "lr": 3.0e-3,
            "grad_clip": 10.0,
        },
        "loss": {
            "residual": 1.0,
            "boundary": 10.0,
            "initial": 0.0,
            "data": 0.0,
            "regularization": 0.0,
        },
        "eval": {"grid_size": 16},
        "output": {
            "root_dir": "runs",
            "save_checkpoints": False,
            "save_predictions": False,
            "save_figures": False,
        },
    }


def test_laplace_analytic_solution_residual_rmse_is_small_float64() -> None:
    problem = build_problem(_laplace_config())
    axis = torch.linspace(0.05, 0.95, 17, dtype=torch.float64)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    residual = problem.residual(ReferenceModel(problem), {"coords": coords})
    residual_rmse = residual.square().mean().sqrt().item()

    assert residual_rmse < 1.0e-6


def test_hard_constraint_laplace_satisfies_all_boundaries_and_zero_boundary_loss() -> None:
    cfg = _laplace_config(boundary_mode="hard")
    problem = build_problem(cfg)
    model = build_model(cfg, problem).to(dtype=torch.float64)
    values = torch.linspace(0.0, 1.0, 21, dtype=torch.float64)
    zeros = torch.zeros_like(values)
    ones = torch.ones_like(values)

    left = torch.stack([zeros, values], dim=1)
    right = torch.stack([ones, values], dim=1)
    bottom = torch.stack([values, zeros], dim=1)
    top = torch.stack([values, ones], dim=1)

    assert torch.allclose(model(left), torch.sin(math.pi * values).reshape(-1, 1), atol=1.0e-10)
    assert torch.allclose(model(right), torch.zeros(21, 1, dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(model(bottom), torch.zeros(21, 1, dtype=torch.float64), atol=1.0e-10)
    assert torch.allclose(model(top), torch.zeros(21, 1, dtype=torch.float64), atol=1.0e-10)

    boundary_batch = problem.sample_boundary(64, torch.device("cpu"), torch.float64)
    loss_boundary = problem.boundary_loss(model, boundary_batch)

    assert loss_boundary.item() == pytest.approx(0.0, abs=1.0e-14)


def test_laplacian_identity_requires_grad_and_no_detach_in_residual_loss() -> None:
    coords = torch.tensor([[0.2, 0.3], [0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    values = torch.sin(math.pi * coords[:, :1]) * torch.sin(math.pi * coords[:, 1:2])
    got = laplacian(values, coords, dims=(0, 1))

    assert torch.allclose(got, -2.0 * math.pi**2 * values, atol=1.0e-8, rtol=1.0e-8)

    problem = build_problem(_laplace_config())
    model = RecordingModel()
    raw_coords = torch.rand(16, 2, dtype=torch.float64, requires_grad=True)
    batches = {
        "collocation": {"coords": raw_coords},
        "boundary": problem.sample_boundary(16, torch.device("cpu"), torch.float64),
        "initial": problem.sample_initial(0, torch.device("cpu"), torch.float64),
    }

    losses = compute_losses(model, problem, batches, {"residual": 1.0, "boundary": 0.0})
    losses["loss_total"].backward()

    assert model.saw_requires_grad is True
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad)
    assert raw_coords.grad is not None
    assert torch.isfinite(raw_coords.grad).all()


def test_helmholtz_manufactured_solution_has_small_residual() -> None:
    cfg = {
        "problem": {
            "name": "helmholtz_antenna2d",
            "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
            "source_kind": "manufactured",
            "k": math.pi,
        }
    }
    problem = build_problem(cfg)
    axis = torch.linspace(-0.9, 0.9, 13, dtype=torch.float64)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    residual = problem.residual(ReferenceModel(problem), {"coords": coords})

    assert residual.square().mean().sqrt().item() < 1.0e-6


def test_minimal_soft_laplace_training_improves_loss_boundary_and_relative_l2(
    tmp_path: Path,
) -> None:
    cfg = _laplace_config(boundary_mode="soft")
    cfg["output"]["root_dir"] = str(tmp_path / "runs")
    run_dir = train(cfg, run_id="physics_soft_laplace")
    metrics = pd.read_csv(run_dir / "metrics.csv")

    first = metrics.iloc[0]
    final = metrics.iloc[-1]

    assert len(metrics) == 250
    assert final["loss_total"] < first["loss_total"]
    assert final["boundary_rmse"] < first["boundary_rmse"]
    assert final["relative_l2"] < first["relative_l2"]
