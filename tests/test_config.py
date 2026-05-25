from pathlib import Path

import pytest

from tnpinn.config.loading import load_config
from tnpinn.config.overrides import apply_overrides, parse_override_value


def test_parse_override_value_converts_common_types() -> None:
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("null") is None
    assert parse_override_value("12") == 12
    assert parse_override_value("1.0e-3") == pytest.approx(1.0e-3)
    assert parse_override_value("[1, 2, 3]") == [1, 2, 3]


def test_load_config_applies_dotlist_overrides() -> None:
    cfg = load_config(
        Path("configs/heat2d_laplace.yaml"),
        overrides=[
            "training.steps=7",
            "model.tensor_network.bond_dim=3",
            "output.save_figures=false",
        ],
    )

    assert cfg["training"]["steps"] == 7
    assert cfg["model"]["tensor_network"]["bond_dim"] == 3
    assert cfg["output"]["save_figures"] is False


def test_apply_overrides_rejects_bad_override() -> None:
    with pytest.raises(ValueError, match="KEY=VALUE"):
        apply_overrides({}, ["training.steps"])
