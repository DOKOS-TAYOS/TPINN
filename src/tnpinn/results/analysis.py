from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from tnpinn.results.aggregate import SUMMARY_COLUMNS, collect_runs


def _write_best_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    if summary.empty:
        columns = pd.Index(SUMMARY_COLUMNS)
        pd.DataFrame(columns=columns).to_csv(out_dir / "best_by_problem.csv", index=False)
        pd.DataFrame(columns=columns).to_csv(out_dir / "best_by_architecture.csv", index=False)
        pd.DataFrame(columns=columns).to_csv(out_dir / "failed_runs.csv", index=False)
        return
    sort_cols = ["best_relative_l2", "best_residual_rmse", "best_loss_total"]
    best_problem = (
        summary.sort_values(sort_cols, na_position="last").groupby("problem", dropna=False).head(1)
    )
    best_arch = (
        summary.groupby(["problem", "architecture"], dropna=False)
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
    summary[summary["status"] != "completed"].to_csv(out_dir / "failed_runs.csv", index=False)


def _line_or_scatter(df: pd.DataFrame, x: str, y: str, hue: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(6, 4))
    if df.empty or x not in df or y not in df:
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


def _write_figures(summary: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    _line_or_scatter(
        summary,
        "bond_dim",
        "best_relative_l2",
        "architecture",
        figures_dir / "rel_l2_vs_bond_dim.png",
        "best relative L2",
    )
    _line_or_scatter(
        summary,
        "n_sites_total",
        "best_relative_l2",
        "architecture",
        figures_dir / "rel_l2_vs_n_sites.png",
        "best relative L2",
    )
    _line_or_scatter(
        summary,
        "num_parameters",
        "time_total_s",
        "architecture",
        figures_dir / "runtime_vs_n_params.png",
        "time total (s)",
    )
    _line_or_scatter(
        summary,
        "architecture",
        "best_relative_l2",
        "problem",
        figures_dir / "rel_l2_by_architecture.png",
        "best relative L2",
    )
    _line_or_scatter(
        summary,
        "architecture",
        "best_residual_rmse",
        "problem",
        figures_dir / "residual_by_architecture.png",
        "best residual RMSE",
    )
    _line_or_scatter(
        summary,
        "architecture",
        "time_total_s",
        "problem",
        figures_dir / "runtime_by_architecture.png",
        "time total (s)",
    )
    _line_or_scatter(
        summary,
        "n_sites_total",
        "best_relative_l2",
        "problem",
        figures_dir / "sites_vs_error.png",
        "best relative L2",
    )
    plt.figure(figsize=(6, 4))
    if not summary.empty:
        pivot = summary.pivot_table(
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
    plt.figure(figsize=(6, 4))
    plt.tight_layout()
    plt.savefig(figures_dir / "loss_curves_best_runs.png", dpi=150)
    plt.close()


def _write_report(summary: pd.DataFrame, out_dir: Path) -> None:
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
    ]
    if not summary.empty:
        best = summary.sort_values(
            ["best_relative_l2", "best_residual_rmse", "best_loss_total"], na_position="last"
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
    _write_figures(summary, figures_dir)
    _write_report(summary, out_dir)
    return summary
