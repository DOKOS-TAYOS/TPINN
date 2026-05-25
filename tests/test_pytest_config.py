from __future__ import annotations

from pathlib import Path

import tomllib


def test_pytest_config_does_not_force_repo_local_basetemp() -> None:
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    pytest_options = config["tool"]["pytest"]["ini_options"]
    addopts = str(pytest_options.get("addopts", ""))

    assert "--basetemp" not in addopts
    assert ".pytest_tmp" not in addopts
    assert "-p tnpinn.pytest_plugin" in addopts
