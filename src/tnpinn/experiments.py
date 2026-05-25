from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from tnpinn.config.loading import load_config
from tnpinn.pinn.trainer import train


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def _set_nested_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _format_override(key: str, value: Any) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    elif isinstance(value, (list, dict)):
        text = yaml.safe_dump(value, default_flow_style=True).strip()
    else:
        text = str(value)
    return f"{key}={text}"


def expand_grid(grid: dict[str, list[Any]]) -> list[list[str]]:
    keys = list(grid.keys())
    values = [grid[key] for key in keys]
    jobs = []
    for combo in product(*values):
        jobs.append([_format_override(key, value) for key, value in zip(keys, combo, strict=True)])
    return jobs


def load_sweep_jobs(path: str | Path) -> list[dict[str, Any]]:
    raw = _read_yaml(path)
    base_config = raw["base_config"]
    sweep = raw.get("sweep", raw)
    grid = sweep.get("grid", {})
    jobs = []
    for index, overrides in enumerate(expand_grid(grid), start=1):
        jobs.append(
            {
                "name": f"{sweep.get('name', Path(path).stem)}_{index:04d}",
                "base_config": base_config,
                "overrides": overrides,
            }
        )
    return jobs


def load_benchmark_jobs(path: str | Path) -> list[dict[str, Any]]:
    raw = _read_yaml(path)
    suite = raw.get("suite", raw)
    jobs: list[dict[str, Any]] = []
    for sweep in suite.get("sweeps", []):
        grid = sweep.get("grid", {})
        for index, overrides in enumerate(expand_grid(grid), start=1):
            jobs.append(
                {
                    "name": f"{sweep.get('name', 'benchmark')}_{index:04d}",
                    "base_config": sweep["base_config"],
                    "overrides": overrides,
                }
            )
    return jobs


def run_jobs(jobs: list[dict[str, Any]], dry_run: bool = False, limit: int | None = None) -> None:
    selected = jobs[:limit] if limit is not None else jobs
    for index, job in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] {job['name']}")
        print(f"  base_config: {job['base_config']}")
        print(f"  overrides: {job['overrides']}")
        if not dry_run:
            config = load_config(job["base_config"], overrides=job["overrides"])
            train(deepcopy(config), run_id=job["name"], dry_run=False)
