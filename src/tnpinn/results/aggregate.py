from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

SUMMARY_COLUMNS = [
    "run_id",
    "run_dir",
    "problem",
    "model_family",
    "architecture",
    "feature_map",
    "n_sites_total",
    "site_dim",
    "bond_dim",
    "num_parameters",
    "seed",
    "best_step",
    "final_step",
    "best_loss_total",
    "final_loss_total",
    "best_residual_rmse",
    "final_residual_rmse",
    "best_relative_l2",
    "final_relative_l2",
    "time_total_s",
    "mean_step_time_s",
    "status",
    "status_reason",
    "record_source",
]

REQUIRED_METRIC_COLUMNS = {
    "step",
    "loss_total",
    "loss_residual",
    "loss_boundary",
    "residual_rmse",
    "boundary_rmse",
    "relative_l2",
    "step_time_s",
}


def _empty_metric_result(status_reason: str, status: str) -> dict[str, Any]:
    return {
        "metrics_status": status,
        "metrics_status_reason": status_reason,
    }


def _read_metrics(metrics_path: Path) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    if not metrics_path.exists():
        return None, _empty_metric_result("missing metrics.csv", "incomplete")
    try:
        df = pd.read_csv(metrics_path)
    except Exception as exc:
        return None, _empty_metric_result(f"cannot read metrics.csv: {exc}", "invalid_metrics")
    missing = sorted(REQUIRED_METRIC_COLUMNS.difference(df.columns))
    if missing:
        return df, _empty_metric_result(
            f"missing metric columns: {', '.join(missing)}", "invalid_metrics"
        )
    if df.empty:
        return df, _empty_metric_result("no metric rows", "invalid_metrics")
    return df, {"metrics_status": "valid", "metrics_status_reason": ""}


def _best_from_metrics(metrics_path: Path) -> dict[str, Any]:
    df, status = _read_metrics(metrics_path)
    if df is None or status["metrics_status"] != "valid":
        return status
    if df.empty:
        return {}
    best = df.sort_values("loss_total", na_position="last").iloc[0]
    final = df.iloc[-1]
    values: dict[str, Any] = {
        "best_step": int(best["step"]),
        "final_step": int(final["step"]),
        "best_loss_total": float(best["loss_total"]),
        "final_loss_total": float(final["loss_total"]),
        "best_residual_rmse": float(best["residual_rmse"]),
        "final_residual_rmse": float(final["residual_rmse"]),
        "best_relative_l2": (
            float(best["relative_l2"]) if not math.isnan(float(best["relative_l2"])) else math.nan
        ),
        "final_relative_l2": (
            float(final["relative_l2"]) if not math.isnan(float(final["relative_l2"])) else math.nan
        ),
        "mean_step_time_s": float(df["step_time_s"].mean()),
        "metrics_status": "valid",
        "metrics_status_reason": "",
    }
    for column in ["num_parameters", "bond_dim", "n_sites_total", "site_dim"]:
        if column in df.columns:
            values[column] = final[column]
    for column in ["architecture", "feature_map", "tn_backend"]:
        if column in df.columns:
            values[column] = final[column]
    return values


def _run_dirs(runs_root: Path) -> list[Path]:
    dirs = {path.parent for path in runs_root.glob("**/summary.json")}
    dirs.update(path.parent for path in runs_root.glob("**/metrics.csv"))
    dirs.update(path for path in runs_root.glob("*/*") if path.is_dir())
    return sorted(dirs)


def _read_summary(summary_path: Path) -> tuple[dict[str, Any], str, str]:
    if not summary_path.exists():
        return {}, "metrics_fallback", "missing summary.json"
    try:
        return json.loads(summary_path.read_text(encoding="utf-8")), "summary_json", ""
    except json.JSONDecodeError as exc:
        return {"status": "failed"}, "summary_json", f"cannot read summary.json: {exc}"


def _row_from_summary(run_dir: Path, summary: dict[str, Any], record_source: str) -> dict[str, Any]:
    row = {column: summary.get(column, math.nan) for column in SUMMARY_COLUMNS}
    row["run_id"] = summary.get("run_id", run_dir.name)
    row["run_dir"] = str(run_dir)
    row["record_source"] = record_source
    row["status_reason"] = ""
    return row


def _apply_status_validation(
    row: dict[str, Any],
    summary_reason: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    reasons = [
        reason for reason in [summary_reason, metrics.get("metrics_status_reason", "")] if reason
    ]
    metrics_status = metrics.get("metrics_status")
    current_status = row.get("status")
    if row.get("record_source") == "metrics_fallback":
        row["status"] = "incomplete"
    elif row.get("status") == "completed" and metrics_status in {"incomplete", "invalid_metrics"}:
        row["status"] = metrics_status
    elif current_status in {None, ""} or (
        isinstance(current_status, float) and math.isnan(current_status)
    ):
        row["status"] = "incomplete"
    row["status_reason"] = "; ".join(reasons)
    return row


def collect_runs(runs_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    root = Path(runs_root)
    for run_dir in _run_dirs(root):
        summary, record_source, summary_reason = _read_summary(run_dir / "summary.json")
        metrics = _best_from_metrics(run_dir / "metrics.csv")
        if record_source == "metrics_fallback":
            row: dict[str, Any] = {column: math.nan for column in SUMMARY_COLUMNS}
            row.update({key: value for key, value in metrics.items() if key in SUMMARY_COLUMNS})
            row["run_id"] = run_dir.name
            row["run_dir"] = str(run_dir)
            row["record_source"] = record_source
        else:
            row = _row_from_summary(run_dir, summary, record_source)
        row = _apply_status_validation(row, summary_reason, metrics)
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index(SUMMARY_COLUMNS))
