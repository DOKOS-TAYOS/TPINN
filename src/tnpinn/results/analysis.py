from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from tnpinn.results.aggregate import SUMMARY_COLUMNS, collect_runs


def _completed(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "status" not in summary:
        return summary
    return cast(pd.DataFrame, summary.loc[summary["status"] == "completed"])


def _write_best_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        columns = pd.Index(SUMMARY_COLUMNS)
        pd.DataFrame(columns=columns).to_csv(out_dir / "best_by_problem.csv", index=False)
        pd.DataFrame(columns=columns).to_csv(out_dir / "best_by_architecture.csv", index=False)
        pd.DataFrame(columns=columns).to_csv(out_dir / "failed_runs.csv", index=False)
        pd.DataFrame(columns=columns).to_csv(out_dir / "invalid_runs.csv", index=False)
        return

    valid_summary = _completed(summary)
    if valid_summary.empty:
        pd.DataFrame(columns=summary.columns).to_csv(out_dir / "best_by_problem.csv", index=False)
        pd.DataFrame(columns=summary.columns).to_csv(
            out_dir / "best_by_architecture.csv", index=False
        )
    else:
        sort_cols = ["best_relative_l2", "best_residual_rmse", "best_loss_total"]
        best_problem = (
            valid_summary.sort_values(sort_cols, na_position="last")
            .groupby("problem", dropna=False)
            .head(1)
        )
        best_arch = (
            valid_summary.groupby(["problem", "architecture"], dropna=False)
            .agg(
                count=("run_id", "count"),
                mean_best_relative_l2=("best_relative_l2", "mean"),
                median_best_relative_l2=("best_relative_l2", "median"),
                std_best_relative_l2=("best_relative_l2", "std"),
                mean_best_residual_rmse=("best_residual_rmse", "mean"),
                mean_time_total_s=("time_total_s", "mean"),
                mean_num_parameters=("num_parameters", "mean"),
            )
            .reset_index()
            .sort_values(["problem", "mean_best_relative_l2"], na_position="last")
        )
        best_problem.to_csv(out_dir / "best_by_problem.csv", index=False)
        best_arch.to_csv(out_dir / "best_by_architecture.csv", index=False)

    invalid_statuses = ["invalid_metrics", "incomplete"]
    summary[summary["status"] != "completed"].to_csv(out_dir / "failed_runs.csv", index=False)
    summary[summary["status"].isin(invalid_statuses)].to_csv(
        out_dir / "invalid_runs.csv", index=False
    )


def _line_or_scatter(df: pd.DataFrame, x: str, y: str, hue: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(6, 4))
    if df.empty or x not in df or y not in df or hue not in df:
        plt.savefig(path, dpi=150)
        plt.close()
        return
    for label, group in df.groupby(hue, dropna=False):
        group = group.sort_values(x)
        plt.plot(group[x], group[y], marker="o", linestyle="-", label=str(label))
    plt.xlabel(x)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _draw_insufficient_data(path: Path, reason: str) -> str:
    plt.figure(figsize=(6, 4))
    plt.text(0.5, 0.55, "insufficient data", ha="center", va="center", fontsize=14)
    plt.text(0.5, 0.42, reason, ha="center", va="center", fontsize=9, wrap=True)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return reason


def _plot_loss_curves(summary: pd.DataFrame, figures_dir: Path) -> str | None:
    path = figures_dir / "loss_curves_best_runs.png"
    if summary.empty:
        return _draw_insufficient_data(path, "no runs found")

    plotted = 0
    plt.figure(figsize=(6, 4))
    for _, row in _completed(summary).iterrows():
        metrics_path = Path(str(row["run_dir"])) / "metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        if len(metrics) < 2 or not {"step", "loss_total"}.issubset(metrics.columns):
            continue
        label = f"{row['problem']}:{row['run_id']}"
        plt.plot(metrics["step"], metrics["loss_total"], marker="o", label=label)
        plotted += 1

    if plotted == 0:
        plt.close()
        return _draw_insufficient_data(path, "no completed runs with at least two metric rows")

    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("loss_total")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return None


def _write_figures(summary: pd.DataFrame, figures_dir: Path) -> list[str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    valid_summary = _completed(summary)
    _line_or_scatter(
        valid_summary,
        "bond_dim",
        "best_relative_l2",
        "architecture",
        figures_dir / "rel_l2_vs_bond_dim.png",
        "best relative L2",
    )
    _line_or_scatter(
        valid_summary,
        "n_sites_total",
        "best_relative_l2",
        "architecture",
        figures_dir / "rel_l2_vs_n_sites.png",
        "best relative L2",
    )
    _line_or_scatter(
        valid_summary,
        "num_parameters",
        "time_total_s",
        "architecture",
        figures_dir / "runtime_vs_n_params.png",
        "time total (s)",
    )
    _line_or_scatter(
        valid_summary,
        "architecture",
        "best_relative_l2",
        "problem",
        figures_dir / "rel_l2_by_architecture.png",
        "best relative L2",
    )
    _line_or_scatter(
        valid_summary,
        "architecture",
        "best_residual_rmse",
        "problem",
        figures_dir / "residual_by_architecture.png",
        "best residual RMSE",
    )
    _line_or_scatter(
        valid_summary,
        "architecture",
        "time_total_s",
        "problem",
        figures_dir / "runtime_by_architecture.png",
        "time total (s)",
    )
    _line_or_scatter(
        valid_summary,
        "n_sites_total",
        "best_relative_l2",
        "problem",
        figures_dir / "sites_vs_error.png",
        "best relative L2",
    )
    plt.figure(figsize=(6, 4))
    if not valid_summary.empty:
        pivot = valid_summary.pivot_table(
            index="problem",
            columns="architecture",
            values="best_relative_l2",
            aggfunc="mean",
        )
        plt.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto")
        plt.xticks(
            range(len(pivot.columns)),
            [str(value) for value in pivot.columns],
            rotation=45,
            ha="right",
        )
        plt.yticks(range(len(pivot.index)), [str(value) for value in pivot.index])
        plt.colorbar(label="mean best relative L2")
    plt.tight_layout()
    plt.savefig(figures_dir / "heatmap_architecture_problem.png", dpi=150)
    plt.close()

    loss_reason = _plot_loss_curves(summary, figures_dir)
    if loss_reason is None:
        return []
    return [f"loss_curves_best_runs.png: insufficient data - {loss_reason}"]


def _write_report(summary: pd.DataFrame, out_dir: Path, figure_notes: list[str]) -> None:
    completed = int((summary["status"] == "completed").sum()) if not summary.empty else 0
    failed = int((summary["status"] != "completed").sum()) if not summary.empty else 0
    lines = [
        "# tnpinn report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Total runs: {len(summary)}",
        f"Completed runs: {completed}",
        f"Failed or incomplete runs: {failed}",
        "",
        "Figures are stored in `figures/`.",
        "- `figures/loss_curves_best_runs.png`",
    ]
    if figure_notes:
        lines.extend(["", "## Figure notes", ""])
        lines.extend(f"- {note}" for note in figure_notes)
    valid_summary = _completed(summary)
    if not valid_summary.empty:
        best = valid_summary.sort_values(
            ["best_relative_l2", "best_residual_rmse", "best_loss_total"],
            na_position="last",
        )
        lines.extend(["", "## Best runs by problem", ""])
        for problem, group in best.groupby("problem", dropna=False):
            row = group.iloc[0]
            lines.append(
                f"- {problem}: `{row['run_id']}` with best_relative_l2={row['best_relative_l2']}"
            )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_runs(runs: str | Path, out: str | Path) -> pd.DataFrame:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    summary = collect_runs(runs)
    summary.to_csv(out_dir / "summary.csv", index=False)
    try:
        summary.to_parquet(out_dir / "summary.parquet", index=False)
    except Exception:
        pass
    _write_best_tables(summary, out_dir)
    figure_notes = _write_figures(summary, figures_dir)
    _write_report(summary, out_dir, figure_notes)
    return summary
