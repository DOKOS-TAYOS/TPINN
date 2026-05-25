from __future__ import annotations

import inspect
from typing import Protocol, cast

import pytest

from tnpinn.config.schema import DEFAULT_CONFIG
from tnpinn.models import _make_tn_backend
from tnpinn.tn import tensorkrowch_backend, torch_reference_backend
from tnpinn.tn.feature_maps import SiteFeatureMap, build_feature_map


class _BackendWithDiagnostics(Protocol):
    def diagnostics(self) -> dict[str, float | int | str]: ...


def _feature_map() -> SiteFeatureMap:
    config = {
        "kind": "fourier_sites",
        "site_dim": 2,
        "n_sites_per_coord": {"x": 2, "y": 2},
        "frequency_scale": 3.141592653589793,
    }
    return build_feature_map(config, ("x", "y"))


def test_torch_reference_backend_is_pure_torch() -> None:
    source = inspect.getsource(torch_reference_backend)

    assert "import tensorkrowch" not in source
    assert "tk." not in source
    assert "ParamNode" not in source
    assert "torch.einsum" in source


def test_tensorkrowch_hybrid_backend_declares_tensor_projection_and_torch_contraction() -> None:
    source = inspect.getsource(tensorkrowch_backend.TensorKrowchHybridBackendBase)
    feature_map = _feature_map()
    backend = cast(
        _BackendWithDiagnostics,
        _make_tn_backend(
            "tensorkrowch_hybrid",
            "global_mps",
            feature_map,
            1,
            {"bond_dim": 3, "init_std": 0.05},
        ),
    )

    diagnostics = backend.diagnostics()

    assert "tk.ParamNode" in source
    assert "tk.Node" in source
    assert "tk.contract_between" in source
    assert "torch.einsum" in source
    assert diagnostics["tn_backend"] == "tensorkrowch_hybrid"
    assert diagnostics["contraction_kind"] == "hybrid_site_nodes_torch_bonds"


def test_legacy_tensorkrowch_nodes_name_is_rejected_with_clear_message() -> None:
    feature_map = _feature_map()

    with pytest.raises(ValueError, match="renamed to 'tensorkrowch_hybrid'"):
        _make_tn_backend(
            "tensorkrowch_nodes",
            "global_mps",
            feature_map,
            1,
            {"bond_dim": 3, "init_std": 0.05},
        )


def test_tensorkrowch_full_is_reserved_and_does_not_claim_hybrid_as_full() -> None:
    from tnpinn.tn.tensorkrowch_full_backend import TensorKrowchFullBackend

    source = inspect.getsource(TensorKrowchFullBackend)
    feature_map = _feature_map()

    assert "torch.einsum" not in source
    with pytest.raises(NotImplementedError, match="tensorkrowch_full"):
        _make_tn_backend(
            "tensorkrowch_full",
            "global_mps",
            feature_map,
            1,
            {"bond_dim": 3, "init_std": 0.05},
        )


def test_default_config_uses_hybrid_backend_and_not_misleading_nodes_name() -> None:
    backend = DEFAULT_CONFIG["model"]["tensor_network"]["backend"]

    assert backend == "tensorkrowch_hybrid"
    assert backend != "tensorkrowch_nodes"
