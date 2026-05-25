from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tnpinn.results.aggregate import collect_runs
from tnpinn.results.analysis import analyze_runs

METRIC_COLUMNS = [
    "step",
    "loss_total",
    "loss_residual",
    "loss_boundary",
    "loss_initial",
    "loss_data",
    "loss_regularization",
    "residual_rmse",
    "boundary_rmse",
    "initial_rmse",
    "relative_l2",
    "max_abs_error",
    "grad_norm",
    "num_parameters",
    "forward_time_s",
    "backward_time_s",
    "step_time_s",
    "bond_dim",
    "n_sites_total",
    "site_dim",
    "architecture",
    "feature_map",
    "tn_backend",
    "core_norm_mean",
    "core_norm_max",
    "effective_rank_mean",
]


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=pd.Index(METRIC_COLUMNS)).to_csv(path, index=False)


def _metric_row(step: int, loss_total: float, relative_l2: float = 1.0) -> dict[str, object]:
    return {
        "step": step,
        "loss_total": loss_total,
        "loss_residual": 0.01 * step,
        "loss_boundary": 0.04 * step,
        "loss_initial": 0.0,
        "loss_data": 0.0,
        "loss_regularization": 0.0,
        "residual_rmse": 0.1 * step,
        "boundary_rmse": 0.2 * step,
        "initial_rmse": 0.0,
        "relative_l2": relative_l2,
        "max_abs_error": 1.0,
        "grad_norm": 0.0,
        "num_parameters": 12,
        "forward_time_s": 0.01,
        "backward_time_s": 0.02,
        "step_time_s": 0.03,
        "bond_dim": 2,
        "n_sites_total": 4,
        "site_dim": 2,
        "architecture": "global_mps",
        "feature_map": "fourier_sites",
        "tn_backend": "tensorkrowch_hybrid",
        "core_norm_mean": 0.0,
        "core_norm_max": 0.0,
        "effective_rank_mean": 0.0,
    }


def _write_summary(run_dir: Path, **updates: object) -> None:
    summary = {
        "status": "completed",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "problem": "heat2d_laplace",
        "model_family": "tensor_network",
        "architecture": "global_mps",
        "feature_map": "fourier_sites",
        "n_sites_total": 4,
        "site_dim": 2,
        "bond_dim": 2,
        "num_parameters": 12,
        "seed": 123,
        "best_step": 3,
        "final_step": 3,
        "best_loss_total": 0.2,
        "final_loss_total": 0.2,
        "best_residual_rmse": 0.3,
        "final_residual_rmse": 0.3,
        "best_relative_l2": 1.0,
        "final_relative_l2": 1.0,
        "time_total_s": 1.0,
        "mean_step_time_s": 0.03,
    }
    summary.update(updates)
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_collect_runs_uses_summary_as_primary_source_when_relative_l2_is_constant(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "heat" / "run_a"
    run_dir.mkdir(parents=True)
    _write_metrics(
        run_dir / "metrics.csv",
        [
            _metric_row(1, 0.9, relative_l2=1.0),
            _metric_row(2, 0.7, relative_l2=1.0),
            _metric_row(3, 0.2, relative_l2=1.0),
        ],
    )
    _write_summary(run_dir, best_step=3, best_loss_total=0.2)

    summary = collect_runs(tmp_path / "runs")
    row = summary.iloc[0]

    assert int(row["best_step"]) == 3
    assert float(row["best_loss_total"]) == 0.2
    assert row["record_source"] == "summary_json"
    assert row["status"] == "completed"


def test_collect_runs_falls_back_to_metrics_without_summary_and_marks_incomplete(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "heat" / "run_without_summary"
    run_dir.mkdir(parents=True)
    _write_metrics(
        run_dir / "metrics.csv",
        [
            _metric_row(1, 0.9, relative_l2=1.0),
            _metric_row(2, 0.2, relative_l2=1.0),
        ],
    )

    summary = collect_runs(tmp_path / "runs")
    row = summary.iloc[0]

    assert int(row["best_step"]) == 2
    assert float(row["best_loss_total"]) == 0.2
    assert row["status"] == "incomplete"
    assert row["record_source"] == "metrics_fallback"
    assert "missing summary.json" in row["status_reason"]


def test_collect_runs_classifies_invalid_metrics_and_missing_metrics(tmp_path: Path) -> None:
    empty_metrics = tmp_path / "runs" / "heat" / "empty_metrics"
    empty_metrics.mkdir(parents=True)
    _write_summary(empty_metrics)
    pd.DataFrame(columns=pd.Index(METRIC_COLUMNS)).to_csv(
        empty_metrics / "metrics.csv", index=False
    )

    missing_metrics = tmp_path / "runs" / "heat" / "missing_metrics"
    missing_metrics.mkdir(parents=True)
    _write_summary(missing_metrics)

    bad_columns = tmp_path / "runs" / "heat" / "bad_columns"
    bad_columns.mkdir(parents=True)
    _write_summary(bad_columns)
    pd.DataFrame([{"step": 1, "loss_total": 1.0}]).to_csv(bad_columns / "metrics.csv", index=False)

    summary = collect_runs(tmp_path / "runs").set_index("run_id")

    assert summary.loc["empty_metrics", "status"] == "invalid_metrics"
    assert "no metric rows" in summary.loc["empty_metrics", "status_reason"]
    assert summary.loc["missing_metrics", "status"] == "incomplete"
    assert "missing metrics.csv" in summary.loc["missing_metrics", "status_reason"]
    assert summary.loc["bad_columns", "status"] == "invalid_metrics"
    assert "missing metric columns" in summary.loc["bad_columns", "status_reason"]


def test_analyze_runs_writes_invalid_runs_and_real_loss_curve_for_one_and_many_runs(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    for name, loss in [("run_a", 0.2), ("run_b", 0.1)]:
        run_dir = runs_root / "heat" / name
        run_dir.mkdir(parents=True)
        _write_metrics(
            run_dir / "metrics.csv",
            [_metric_row(1, 0.9), _metric_row(2, loss)],
        )
        _write_summary(run_dir, run_id=name, best_step=2, best_loss_total=loss)

    invalid_dir = runs_root / "heat" / "invalid"
    invalid_dir.mkdir(parents=True)
    _write_summary(invalid_dir, run_id="invalid")
    pd.DataFrame(columns=pd.Index(METRIC_COLUMNS)).to_csv(invalid_dir / "metrics.csv", index=False)

    out_dir = tmp_path / "reports"
    analyze_runs(runs_root, out_dir)

    invalid = pd.read_csv(out_dir / "invalid_runs.csv")
    report = (out_dir / "report.md").read_text(encoding="utf-8")

    assert "invalid" in invalid["run_id"].tolist()
    assert (out_dir / "figures" / "loss_curves_best_runs.png").stat().st_size > 5_000
    assert "loss_curves_best_runs.png" in report


def test_analyze_runs_marks_loss_curve_insufficient_when_metrics_empty(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "heat" / "empty"
    run_dir.mkdir(parents=True)
    _write_summary(run_dir, run_id="empty")
    pd.DataFrame(columns=pd.Index(METRIC_COLUMNS)).to_csv(run_dir / "metrics.csv", index=False)

    out_dir = tmp_path / "reports"
    analyze_runs(tmp_path / "runs", out_dir)

    report = (out_dir / "report.md").read_text(encoding="utf-8")
    invalid = pd.read_csv(out_dir / "invalid_runs.csv")

    assert "insufficient data" in report
    assert "no completed runs with at least two metric rows" in report
    assert invalid.loc[0, "status"] == "invalid_metrics"
