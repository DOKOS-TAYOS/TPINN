from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 1234,
    "device": "auto",
    "dtype": "float64",
    "model": {
        "family": "tensor_network",
        "baseline": None,
        "hidden_dim": 64,
        "n_layers": 3,
        "feature_map": {
            "kind": "fourier_sites",
            "site_dim": 2,
            "n_sites_per_coord": {},
            "frequency_scale": 3.141592653589793,
            "frequency_mode": "powers_of_two",
            "trainable_frequencies": False,
            "include_identity_site": False,
        },
        "tensor_network": {
            "backend": "tensorkrowch_nodes",
            "architecture": "coordinate_branch_mps",
            "bond_dim": 8,
            "branch_factor": 2,
            "init_std": 0.05,
        },
    },
    "training": {
        "optimizer": "adam",
        "lr": 1.0e-3,
        "steps": 1000,
        "n_collocation": 512,
        "n_boundary": 128,
        "n_initial": 0,
        "resample_every": 1,
        "grad_clip": 10.0,
    },
    "loss": {
        "residual": 1.0,
        "boundary": 1.0,
        "initial": 1.0,
        "data": 0.0,
        "regularization": 0.0,
    },
    "eval": {
        "grid_size": 64,
        "eval_every": 100,
        "save_figures_every": 1000,
    },
    "output": {
        "root_dir": "runs",
        "save_checkpoints": True,
        "save_predictions": True,
        "save_figures": True,
    },
}


def deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def with_defaults(config: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(DEFAULT_CONFIG, config)


def validate_config(config: dict[str, Any]) -> None:
    if "problem" not in config or "name" not in config["problem"]:
        raise ValueError("Config must define problem.name")
    if "model" not in config:
        raise ValueError("Config must define model")
