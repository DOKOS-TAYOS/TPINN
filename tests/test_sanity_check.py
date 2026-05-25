from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tnpinn.cli.main import main as cli_main
from tnpinn.config.loading import save_config


def test_sanity_check_cli_writes_report_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tnpinn.cli import sanity_check

    def fake_run_sanity_checks(out: Path, options: Any) -> dict[str, Any]:
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "passed",
            "passed": True,
            "checks": [{"name": "fake", "status": "passed"}],
        }
        (out / "sanity_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "sanity_report.md").write_text(
            "# sanity check\n\n- fake: passed\n",
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(sanity_check, "run_sanity_checks", fake_run_sanity_checks)

    cli_main(["sanity-check", "--out", str(tmp_path / "sanity")])

    assert (tmp_path / "sanity" / "sanity_summary.json").exists()
    assert (tmp_path / "sanity" / "sanity_report.md").exists()


def test_sanity_check_cli_exits_nonzero_when_a_check_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tnpinn.cli import sanity_check

    def fake_run_sanity_checks(out: Path, options: Any) -> dict[str, Any]:
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "status": "failed",
            "passed": False,
            "checks": [{"name": "fake", "status": "failed"}],
        }
        (out / "sanity_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        (out / "sanity_report.md").write_text(
            "# sanity check\n\n- fake: failed\n",
            encoding="utf-8",
        )
        return summary

    monkeypatch.setattr(sanity_check, "run_sanity_checks", fake_run_sanity_checks)

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["sanity-check", "--out", str(tmp_path / "sanity")])

    assert exc_info.value.code == 1


def test_sanity_benchmark_smoke_writes_run_index_and_checks_fixed_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tnpinn.sanity import SanityOptions, run_benchmark_smoke_check

    def fake_train(
        config: dict[str, Any],
        run_id: str | None = None,
        dry_run: bool = False,
    ) -> Path:
        run_dir = tmp_path / "sanity" / "benchmark_runs" / "heat2d_laplace" / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, run_dir / "resolved_config.yaml")
        pd.DataFrame(
            [{"step": 1, "loss_total": 1.0, "boundary_rmse": 1.0, "relative_l2": 1.0}]
        ).to_csv(run_dir / "metrics.csv", index=False)
        (run_dir / "summary.json").write_text(
            json.dumps({"status": "completed", "run_id": run_id, "run_dir": str(run_dir)}),
            encoding="utf-8",
        )
        return run_dir

    monkeypatch.setattr("tnpinn.sanity.train", fake_train)

    result = run_benchmark_smoke_check(tmp_path / "sanity", SanityOptions(benchmark_steps=1))
    run_index = pd.read_csv(tmp_path / "sanity" / "benchmark" / "run_index.csv")

    assert result.status == "passed"
    assert result.details["dry_run_jobs"] == 5
    assert len(run_index) == 5
    assert set(run_index["status"]) == {"completed"}


def test_sanity_analysis_smoke_accepts_nonempty_figures_or_insufficient_data(
    tmp_path: Path,
) -> None:
    from tnpinn.sanity import run_analysis_smoke_check

    run_dir = tmp_path / "runs" / "heat2d_laplace" / "invalid"
    run_dir.mkdir(parents=True)
    pd.DataFrame(columns=pd.Index(["step", "loss_total"])).to_csv(
        run_dir / "metrics.csv",
        index=False,
    )

    result = run_analysis_smoke_check(tmp_path / "runs", tmp_path / "analysis")

    assert result.status == "passed"
    assert (tmp_path / "analysis" / "summary.csv").exists()
    assert (tmp_path / "analysis" / "report.md").exists()
    assert "insufficient data" in (tmp_path / "analysis" / "report.md").read_text(encoding="utf-8")
