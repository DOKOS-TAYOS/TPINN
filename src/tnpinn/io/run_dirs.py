from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]


def make_run_id(config: dict[str, Any]) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{config_hash(config)}"


def make_run_dir(config: dict[str, Any], run_id: str | None = None) -> Path:
    problem = str(config.get("problem", {}).get("name", "unknown_problem"))
    root = Path(str(config.get("output", {}).get("root_dir", "runs")))
    return root / problem / (run_id or make_run_id(config))
