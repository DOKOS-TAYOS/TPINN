from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any

import yaml


def parse_override_value(raw: str) -> Any:
    """Parse a CLI override value using YAML's scalar/list syntax."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _set_nested(mapping: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    if any(part == "" for part in parts):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    cursor: MutableMapping[str, Any] = mapping
    for part in parts[:-1]:
        child = cursor.get(part)
        if child is None:
            child = {}
            cursor[part] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot set {dotted_key!r}; {part!r} is not a mapping")
        cursor = child
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply dotlist KEY=VALUE overrides to a config dictionary in place."""
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must have KEY=VALUE form, got {override!r}")
        key, raw_value = override.split("=", 1)
        _set_nested(config, key.strip(), parse_override_value(raw_value.strip()))
    return config
