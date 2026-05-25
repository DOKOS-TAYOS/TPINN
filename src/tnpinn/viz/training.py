from __future__ import annotations

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


def plot_training_curves(metrics_csv: str | Path, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return
    plt.figure(figsize=(6, 4))
    for column in ["loss_total", "loss_residual", "loss_boundary", "relative_l2"]:
        if column in df and bool(df[column].notna().any()):
            plt.plot(df["step"], df[column], label=column)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "training_curves.png", dpi=150)
    plt.close()
