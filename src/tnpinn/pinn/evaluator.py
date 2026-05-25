from __future__ import annotations

from pathlib import Path

import yaml

from tnpinn.io.checkpoints import load_checkpoint
from tnpinn.models import build_model
from tnpinn.pinn.trainer import _save_predictions, resolve_device, resolve_dtype
from tnpinn.problems import build_problem
from tnpinn.viz.heat import plot_heat_prediction
from tnpinn.viz.training import plot_training_curves


def evaluate_run(
    run: str | Path,
    checkpoint: str = "best",
    grid_size: int | None = None,
    save_predictions: bool = True,
    make_figures: bool = True,
) -> None:
    run_dir = Path(run)
    config = yaml.safe_load((run_dir / "resolved_config.yaml").read_text(encoding="utf-8"))
    if grid_size is not None:
        config.setdefault("eval", {})["grid_size"] = grid_size
    device = resolve_device(str(config.get("device", "auto")))
    dtype = resolve_dtype(str(config.get("dtype", "float64")))
    problem = build_problem(config)
    model = build_model(config, problem).to(device=device, dtype=dtype)
    if checkpoint in {"best", "last"}:
        checkpoint_path = run_dir / f"checkpoint_{checkpoint}.pt"
    else:
        checkpoint_path = Path(checkpoint)
    load_checkpoint(checkpoint_path, model)
    if save_predictions:
        _save_predictions(run_dir, model, problem, config, device, dtype)
    if make_figures:
        figures_dir = run_dir / "figures"
        plot_training_curves(run_dir / "metrics.csv", figures_dir)
        plot_heat_prediction(run_dir / "predictions.npz", figures_dir)
