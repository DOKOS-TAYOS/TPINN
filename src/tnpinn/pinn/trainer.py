from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tnpinn.config.loading import save_config
from tnpinn.io.checkpoints import save_checkpoint
from tnpinn.io.logging import MetricsWriter
from tnpinn.io.run_dirs import make_run_dir
from tnpinn.models import build_model
from tnpinn.pinn.losses import compute_losses
from tnpinn.problems import build_problem
from tnpinn.problems.base import PhysicsProblem
from tnpinn.tn.diagnostics import parameter_count, tensor_diagnostics
from tnpinn.viz.heat import plot_heat_prediction
from tnpinn.viz.training import plot_training_curves


def resolve_device(name: str | None) -> torch.device:
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str | None) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64" or name is None:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def _sample_batches(
    problem: PhysicsProblem,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, torch.Tensor]]:
    training = config.get("training", {})
    return {
        "collocation": problem.sample_collocation(
            int(training.get("n_collocation", 512)), device, dtype
        ),
        "boundary": problem.sample_boundary(int(training.get("n_boundary", 128)), device, dtype),
        "initial": problem.sample_initial(int(training.get("n_initial", 0)), device, dtype),
    }


def _grad_norm(model: torch.nn.Module) -> float:
    values = [
        param.grad.detach().norm()
        for param in model.parameters()
        if param.grad is not None and param.requires_grad
    ]
    if not values:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack(values)).cpu())


def _relative_metrics(
    model: torch.nn.Module,
    problem: PhysicsProblem,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float]:
    grid_size = min(int(config.get("eval", {}).get("grid_size", 64)), 64)
    grid = problem.make_eval_grid(grid_size, device, dtype)
    coords = grid["coords"]
    reference = problem.reference_solution(coords)
    if reference is None:
        return math.nan, math.nan
    with torch.no_grad():
        prediction = model(coords)
        error = prediction - reference
        rel = error.norm() / (reference.norm() + 1.0e-12)
        max_abs = error.abs().max()
    return float(rel.cpu()), float(max_abs.cpu())


def _to_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _save_predictions(
    run_dir: Path,
    model: torch.nn.Module,
    problem: PhysicsProblem,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    grid_size = min(int(config.get("eval", {}).get("grid_size", 64)), 128)
    grid = problem.make_eval_grid(grid_size, device, dtype)
    coords = grid["coords"]
    with torch.no_grad():
        prediction = model(coords).detach().cpu().numpy()
        reference = problem.reference_solution(coords)
        reference_np = np.array([]) if reference is None else reference.detach().cpu().numpy()
    grid_shape = grid.get("shape")
    np.savez(
        run_dir / "predictions.npz",
        coords=coords.detach().cpu().numpy(),
        u_pred=prediction,
        u_ref=reference_np,
        grid_shape=np.array([] if grid_shape is None else grid_shape.detach().cpu().numpy()),
    )


def _summary_from_metrics(
    config: dict[str, Any],
    run_dir: Path,
    model: torch.nn.Module,
    rows: list[dict[str, Any]],
    status: str,
    time_total_s: float,
) -> dict[str, Any]:
    final = rows[-1] if rows else {}
    best = min(rows, key=lambda row: float(row.get("loss_total", math.inf))) if rows else {}
    diagnostics = getattr(model, "diagnostics", lambda: {})()
    return {
        "status": status,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "problem": config.get("problem", {}).get("name"),
        "model_family": config.get("model", {}).get("family"),
        "architecture": diagnostics.get("architecture", config.get("model", {}).get("baseline")),
        "feature_map": diagnostics.get("feature_map", ""),
        "n_sites_total": diagnostics.get("n_sites_total", 0),
        "site_dim": diagnostics.get("site_dim", 0),
        "bond_dim": diagnostics.get("bond_dim", 0),
        "tn_backend": diagnostics.get("tn_backend", ""),
        "num_parameters": parameter_count(model),
        "seed": config.get("seed"),
        "best_step": best.get("step", 0),
        "final_step": final.get("step", 0),
        "best_loss_total": best.get("loss_total", math.nan),
        "final_loss_total": final.get("loss_total", math.nan),
        "best_residual_rmse": best.get("residual_rmse", math.nan),
        "final_residual_rmse": final.get("residual_rmse", math.nan),
        "best_relative_l2": best.get("relative_l2", math.nan),
        "final_relative_l2": final.get("relative_l2", math.nan),
        "time_total_s": time_total_s,
        "mean_step_time_s": float(np.mean([row.get("step_time_s", 0.0) for row in rows]))
        if rows
        else 0.0,
    }


def train(config: dict[str, Any], run_id: str | None = None, dry_run: bool = False) -> Path:
    torch.manual_seed(int(config.get("seed", 1234)))
    device = resolve_device(str(config.get("device", "auto")))
    dtype = resolve_dtype(str(config.get("dtype", "float64")))
    problem = build_problem(config)
    model = build_model(config, problem).to(device=device, dtype=dtype)
    run_dir = make_run_dir(config, run_id)

    if dry_run:
        print(f"problem: {problem.name}")
        print(f"run_dir: {run_dir}")
        print(f"parameters: {parameter_count(model)}")
        print(f"device: {device}")
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, run_dir / "resolved_config.yaml")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config.get("training", {}).get("lr", 1.0e-3))
    )
    steps = int(config.get("training", {}).get("steps", 1000))
    grad_clip = config.get("training", {}).get("grad_clip")
    rows: list[dict[str, Any]] = []
    best_loss = math.inf
    start_total = time.perf_counter()

    with MetricsWriter(run_dir / "metrics.csv") as writer:
        for step in range(1, steps + 1):
            step_start = time.perf_counter()
            batches = _sample_batches(problem, config, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            forward_start = time.perf_counter()
            losses = compute_losses(model, problem, batches, config.get("loss", {}))
            forward_time = time.perf_counter() - forward_start
            backward_start = time.perf_counter()
            losses["loss_total"].backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()
            backward_time = time.perf_counter() - backward_start

            relative_l2, max_abs_error = _relative_metrics(model, problem, config, device, dtype)
            diagnostics = getattr(model, "diagnostics", lambda: {})()
            diagnostics.update(tensor_diagnostics(model))
            row: dict[str, Any] = {
                "step": step,
                "loss_total": _to_float(losses["loss_total"]),
                "loss_residual": _to_float(losses["loss_residual"]),
                "loss_boundary": _to_float(losses["loss_boundary"]),
                "loss_initial": _to_float(losses["loss_initial"]),
                "loss_data": _to_float(losses["loss_data"]),
                "loss_regularization": _to_float(losses["loss_regularization"]),
                "residual_rmse": math.sqrt(max(_to_float(losses["loss_residual"]), 0.0)),
                "boundary_rmse": math.sqrt(max(_to_float(losses["loss_boundary"]), 0.0)),
                "initial_rmse": math.sqrt(max(_to_float(losses["loss_initial"]), 0.0)),
                "relative_l2": relative_l2,
                "max_abs_error": max_abs_error,
                "grad_norm": _grad_norm(model),
                "num_parameters": parameter_count(model),
                "forward_time_s": forward_time,
                "backward_time_s": backward_time,
                "step_time_s": time.perf_counter() - step_start,
                **diagnostics,
            }
            writer.write(row)
            rows.append(row)
            if row["loss_total"] < best_loss:
                best_loss = float(row["loss_total"])
                if config.get("output", {}).get("save_checkpoints", True):
                    save_checkpoint(run_dir / "checkpoint_best.pt", model, optimizer, step, row)

    if config.get("output", {}).get("save_checkpoints", True):
        save_checkpoint(
            run_dir / "checkpoint_last.pt", model, optimizer, steps, rows[-1] if rows else {}
        )
    if config.get("output", {}).get("save_predictions", True):
        _save_predictions(run_dir, model, problem, config, device, dtype)
    if config.get("output", {}).get("save_figures", True):
        plot_training_curves(run_dir / "metrics.csv", figures_dir)
        plot_heat_prediction(run_dir / "predictions.npz", figures_dir)

    summary = _summary_from_metrics(
        config, run_dir, model, rows, "completed", time.perf_counter() - start_total
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return run_dir
