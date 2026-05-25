from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tnpinn.config.overrides import apply_overrides
from tnpinn.config.schema import validate_config, with_defaults


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    config = with_defaults(raw)
    apply_overrides(config, overrides)
    validate_config(config)
    return config


def save_config(config: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
