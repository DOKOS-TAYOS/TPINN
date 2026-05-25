from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


def plot_heat_prediction(predictions_npz: str | Path, out_dir: str | Path) -> None:
    path = Path(predictions_npz)
    if not path.exists():
        return
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(path)
    if "u_pred" not in data or "grid_shape" not in data:
        return
    shape = tuple(int(x) for x in data["grid_shape"])
    pred = data["u_pred"][:, 0].reshape(shape)
    plt.figure(figsize=(5, 4))
    plt.imshow(pred.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto")
    plt.colorbar(label="u_pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(out / "heat_prediction.png", dpi=150)
    plt.close()
