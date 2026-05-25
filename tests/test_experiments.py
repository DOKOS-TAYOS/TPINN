from pathlib import Path

import pytest

from tnpinn.experiments import load_benchmark_jobs, load_sweep_jobs


def test_load_sweep_jobs_applies_fixed_to_every_grid_job(tmp_path: Path) -> None:
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        """
base_config: configs/heat2d_laplace.yaml
sweep:
  name: tiny
  fixed:
    training.steps: 10
    training.n_collocation: 64
  grid:
    model.tensor_network.bond_dim: [2, 4]
    model.tensor_network.architecture: [global_mps]
""",
        encoding="utf-8",
    )

    jobs = load_sweep_jobs(sweep_path)

    assert len(jobs) == 2
    assert jobs[0]["overrides"] == [
        "training.steps=10",
        "training.n_collocation=64",
        "model.tensor_network.bond_dim=2",
        "model.tensor_network.architecture=global_mps",
    ]
    assert jobs[1]["overrides"] == [
        "training.steps=10",
        "training.n_collocation=64",
        "model.tensor_network.bond_dim=4",
        "model.tensor_network.architecture=global_mps",
    ]


def test_load_benchmark_jobs_applies_fixed_and_rejects_conflicts(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """
suite:
  name: tiny_suite
  sweeps:
    - name: heat
      base_config: configs/heat2d_laplace.yaml
      fixed:
        training.steps: 10
      grid:
        model.tensor_network.bond_dim: [2, 4]
    - name: bad
      base_config: configs/heat2d_laplace.yaml
      fixed:
        training.steps: 10
      grid:
        training.steps: [20]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="fixed.*grid.*training\\.steps"):
        load_benchmark_jobs(suite_path)


def test_benchmark_smoke_jobs_include_fixed_overrides() -> None:
    jobs = load_benchmark_jobs("configs/benchmark_smoke.yaml")

    assert len(jobs) == 5
    assert "training.steps=10" in jobs[0]["overrides"]
    assert "training.n_collocation=64" in jobs[0]["overrides"]
    assert "eval.grid_size=16" in jobs[0]["overrides"]
    assert "training.steps=10" in jobs[-1]["overrides"]
    assert "eval.grid_size=64" in jobs[-1]["overrides"]
