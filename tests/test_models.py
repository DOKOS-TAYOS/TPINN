import pytest
import torch

from tnpinn.models import build_model
from tnpinn.problems import build_problem


def _small_config(architecture: str) -> dict:
    return {
        "problem": {"name": "heat2d_laplace", "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]}},
        "model": {
            "family": "tensor_network",
            "output_dim": 1,
            "feature_map": {
                "kind": "fourier_sites",
                "site_dim": 2,
                "n_sites_per_coord": {"x": 2, "y": 2},
                "frequency_scale": 3.141592653589793,
            },
            "tensor_network": {
                "backend": "tensorkrowch_hybrid",
                "architecture": architecture,
                "bond_dim": 3,
                "branch_factor": 2,
                "init_std": 0.05,
            },
        },
    }


@pytest.mark.parametrize(
    "architecture",
    ["global_mps", "coordinate_branch_mps", "binary_ttn", "branched_mps"],
)
def test_tensor_network_architecture_forward_and_derivatives(architecture: str) -> None:
    problem = build_problem(_small_config(architecture))
    model = build_model(_small_config(architecture), problem).to(dtype=torch.float64)
    coords = torch.rand(6, 2, dtype=torch.float64, requires_grad=True)

    out = model(coords)
    assert out.shape == (6, 1)

    loss = out.square().mean()
    grad_coords = torch.autograd.grad(loss, coords, create_graph=True, retain_graph=True)[0]
    second = torch.autograd.grad(grad_coords[:, 0].sum(), coords, create_graph=True)[0][:, 0]
    loss.backward()

    assert grad_coords.shape == coords.shape
    assert second.shape == (6,)
    assert any(param.grad is not None for param in model.parameters())


@pytest.mark.parametrize("baseline", ["mlp", "siren"])
def test_baseline_models_forward(baseline: str) -> None:
    cfg = _small_config("global_mps")
    cfg["model"]["family"] = "baseline"
    cfg["model"]["baseline"] = baseline
    cfg["model"]["hidden_dim"] = 8
    cfg["model"]["n_layers"] = 2
    problem = build_problem(cfg)
    model = build_model(cfg, problem)

    out = model(torch.rand(3, 2))

    assert out.shape == (3, 1)
