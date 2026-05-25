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
]


def _best_from_metrics(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {}
    df = pd.read_csv(metrics_path)
    if df.empty:
        return {}
    relative_l2 = pd.Series(pd.to_numeric(df["relative_l2"], errors="coerce"))
    sort_column = "relative_l2" if int(relative_l2.count()) > 0 else "loss_total"
    best = df.sort_values(sort_column, na_position="last").iloc[0]
    final = df.iloc[-1]
    return {
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
    }


def collect_runs(runs_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary_path in Path(runs_root).glob("**/summary.json"):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {"status": "failed", "run_dir": str(summary_path.parent)}
        row = {column: summary.get(column, math.nan) for column in SUMMARY_COLUMNS}
        row["run_dir"] = str(summary_path.parent)
        metrics = _best_from_metrics(summary_path.parent / "metrics.csv")
        row.update({key: value for key, value in metrics.items() if key in SUMMARY_COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows, columns=pd.Index(SUMMARY_COLUMNS))
