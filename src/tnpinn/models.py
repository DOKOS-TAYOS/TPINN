from __future__ import annotations

from typing import Any

import torch

from tnpinn.baselines import MLP, SIREN
from tnpinn.problems.base import PhysicsProblem
from tnpinn.tn.feature_maps import SiteFeatureMap, build_feature_map
from tnpinn.tn.tensorkrowch_backend import (
    BinaryTTNHybridBackend,
    BranchedMPSHybridBackend,
    CoordinateBranchMPSHybridBackend,
    GlobalMPSHybridBackend,
)
from tnpinn.tn.tensorkrowch_full_backend import TensorKrowchFullBackend
from tnpinn.tn.torch_reference_backend import (
    TorchReferenceBinaryTTN,
    TorchReferenceBranchedMPS,
    TorchReferenceCoordinateBranchMPS,
    TorchReferenceGlobalMPS,
)


class TensorNetworkPINN(torch.nn.Module):
    def __init__(
        self,
        feature_map: SiteFeatureMap,
        backend: torch.nn.Module,
        output_transform: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.feature_map = feature_map
        self.backend = backend
        self.output_transform = output_transform

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        features = self.feature_map(coords)
        raw = self.backend(features)
        if self.output_transform is None:
            return raw
        return self.output_transform(coords, raw)

    def diagnostics(self) -> dict[str, float | int | str]:
        diagnostics = getattr(self.backend, "diagnostics", lambda: {})()
        diagnostics["feature_map"] = self.feature_map.kind
        diagnostics["site_dim"] = self.feature_map.site_dim
        diagnostics["n_sites_total"] = self.feature_map.n_sites_total
        return diagnostics


class OutputTransformedModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, output_transform: torch.nn.Module) -> None:
        super().__init__()
        self.model = model
        self.output_transform = output_transform

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.output_transform(coords, self.model(coords))


class HeatLaplaceHardConstraint(torch.nn.Module):
    def forward(self, coords: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        x = coords[:, :1]
        y = coords[:, 1:2]
        return (1.0 - x) * torch.sin(torch.pi * y) + x * (1.0 - x) * y * (1.0 - y) * raw


def _make_tn_backend(
    backend_name: str,
    architecture: str,
    feature_map: SiteFeatureMap,
    output_dim: int,
    tn_config: dict[str, Any],
) -> torch.nn.Module:
    bond_dim = int(tn_config.get("bond_dim", 8))
    init_std = float(tn_config.get("init_std", 0.05))
    branch_factor = int(tn_config.get("branch_factor", 2))

    if backend_name == "tensorkrowch_nodes":
        raise ValueError(
            "Tensor-network backend 'tensorkrowch_nodes' was renamed to "
            "'tensorkrowch_hybrid' because it is a hybrid implementation, not a full "
            "TensorKrowch contraction backend."
        )
    if backend_name == "tensorkrowch_full":
        return TensorKrowchFullBackend(architecture)
    if backend_name == "tensorkrowch_hybrid":
        classes = {
            "global_mps": GlobalMPSHybridBackend,
            "coordinate_branch_mps": CoordinateBranchMPSHybridBackend,
            "binary_ttn": BinaryTTNHybridBackend,
            "branched_mps": BranchedMPSHybridBackend,
        }
    elif backend_name in {"torch_reference", "torch_einsum_reference"}:
        classes = {
            "global_mps": TorchReferenceGlobalMPS,
            "coordinate_branch_mps": TorchReferenceCoordinateBranchMPS,
            "binary_ttn": TorchReferenceBinaryTTN,
            "branched_mps": TorchReferenceBranchedMPS,
        }
    else:
        raise ValueError(f"Unknown tensor-network backend: {backend_name}")

    if architecture not in classes:
        raise ValueError(f"Unknown tensor-network architecture: {architecture}")

    cls = classes[architecture]
    if architecture == "coordinate_branch_mps":
        return cls(feature_map.site_slices, feature_map.site_dim, output_dim, bond_dim, init_std)
    if architecture == "branched_mps":
        return cls(
            feature_map.n_sites_total,
            feature_map.site_dim,
            output_dim,
            bond_dim,
            branch_factor,
            init_std,
        )
    return cls(feature_map.n_sites_total, feature_map.site_dim, output_dim, bond_dim, init_std)


def _uses_laplace_hard_constraint(config: dict[str, Any], problem: PhysicsProblem) -> bool:
    problem_config = config.get("problem", {})
    return problem.name == "heat2d_laplace" and (
        problem_config.get("boundary_mode") == "hard"
        or bool(problem_config.get("use_hard_constraints", False))
    )


def build_model(config: dict[str, Any], problem: PhysicsProblem) -> torch.nn.Module:
    model_config = config.get("model", {})
    family = str(model_config.get("family", "tensor_network"))
    baseline = model_config.get("baseline")
    output_dim = int(model_config.get("output_dim") or problem.output_dim)

    if family == "baseline" or baseline in {"mlp", "siren"}:
        hidden_dim = int(model_config.get("hidden_dim", 64))
        n_layers = int(model_config.get("n_layers", 3))
        if baseline == "siren":
            base_model: torch.nn.Module = SIREN(problem.input_dim, output_dim, hidden_dim, n_layers)
        else:
            base_model = MLP(problem.input_dim, output_dim, hidden_dim, n_layers)
        if _uses_laplace_hard_constraint(config, problem):
            return OutputTransformedModel(base_model, HeatLaplaceHardConstraint())
        return base_model

    feature_map = build_feature_map(model_config.get("feature_map", {}), problem.coordinates)
    tn_config = model_config.get("tensor_network", {})
    backend = _make_tn_backend(
        str(tn_config.get("backend", "tensorkrowch_hybrid")),
        str(tn_config.get("architecture", "coordinate_branch_mps")),
        feature_map,
        output_dim,
        tn_config,
    )
    output_transform = None
    if _uses_laplace_hard_constraint(config, problem):
        output_transform = HeatLaplaceHardConstraint()
    return TensorNetworkPINN(feature_map, backend, output_transform)
