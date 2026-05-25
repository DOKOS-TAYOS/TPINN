from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

import pytest


def _make_process_basetemp() -> Path:
    root = Path.cwd() / ".pytest_tmp_runs"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"pytest-{os.getpid()}-{uuid.uuid4().hex}"


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: Any) -> None:
    if getattr(config.option, "basetemp", None) is not None:
        return
    config.option.basetemp = str(_make_process_basetemp())
