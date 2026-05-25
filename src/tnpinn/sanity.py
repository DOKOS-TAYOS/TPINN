from __future__ import annotations

import contextlib
import io
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from tnpinn.config.loading import load_config
from tnpinn.config.overrides import parse_override_value
from tnpinn.experiments import load_benchmark_jobs, run_jobs
from tnpinn.models import HeatLaplaceHardConstraint, OutputTransformedModel, build_model
from tnpinn.pinn.losses import compute_losses
from tnpinn.pinn.trainer import resolve_dtype, train
from tnpinn.problems import build_problem
from tnpinn.results.analysis import analyze_runs


@dataclass
class SanityOptions:
    function_steps: int = 500
    laplace_steps: int = 250
    hard_steps: int = 50
    benchmark_steps: int | None = None
    seed: int = 1234


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    details: dict[str, Any]


def _bounded_steps(value: int, maximum: int = 500) -> int:
    if value < 1:
        raise ValueError("Sanity-check step counts must be positive")
    return min(value, maximum)


def _target_function(coords: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * coords[:, :1]) * torch.sin(math.pi * coords[:, 1:2])


def _assert_finite_metrics(metrics: pd.DataFrame, columns: set[str]) -> None:
    if metrics.empty:
        raise AssertionError("metrics.csv has no rows")
    missing = columns.difference(metrics.columns)
    if missing:
        raise AssertionError(f"metrics.csv is missing columns: {sorted(missing)}")
    for column in columns:
        values = pd.Series(pd.to_numeric(metrics[column], errors="coerce"))
        if bool(values.isna().any()):
            raise AssertionError(f"metrics column contains NaN or non-numeric values: {column}")


def _base_laplace_config(out: Path, steps: int, boundary_mode: str) -> dict[str, Any]:
    return {
        "seed": 1234,
        "device": "cpu",
        "dtype": "float64",
        "problem": {
            "name": "heat2d_laplace",
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "boundary_mode": boundary_mode,
        },
        "model": {
            "family": "baseline",
            "baseline": "mlp",
            "hidden_dim": 32,
            "n_layers": 2,
            "output_dim": 1,
        },
        "training": {
            "steps": steps,
            "n_collocation": 64,
            "n_boundary": 64,
            "n_initial": 0,
            "lr": 3.0e-3,
            "grad_clip": 10.0,
        },
        "loss": {
            "residual": 1.0,
            "boundary": 10.0 if boundary_mode == "soft" else 0.0,
            "initial": 0.0,
            "data": 0.0,
            "regularization": 0.0,
        },
        "eval": {"grid_size": 16},
        "output": {
            "root_dir": str(out),
            "save_checkpoints": False,
            "save_predictions": False,
            "save_figures": False,
        },
    }


def run_function_fitting_check(out: Path, options: SanityOptions) -> CheckResult:
    steps = _bounded_steps(options.function_steps)
    torch.manual_seed(options.seed)
    dtype = torch.float64
    device = torch.device("cpu")
    config = {
        "seed": options.seed,
        "device": "cpu",
        "dtype": "float64",
        "problem": {"name": "heat2d_laplace"},
        "model": {
            "family": "tensor_network",
            "output_dim": 1,
            "feature_map": {
                "kind": "fourier_sites",
                "site_dim": 2,
                "n_sites_per_coord": {"x": 1, "y": 1},
                "frequency_scale": math.pi,
                "frequency_mode": "linear",
            },
            "tensor_network": {
                "backend": "torch_reference",
                "architecture": "global_mps",
                "bond_dim": 4,
                "init_std": 0.1,
            },
        },
    }
    problem = build_problem(config)
    model = build_model(config, problem).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=5.0e-2)
    coords = torch.rand(128, 2, device=device, dtype=dtype)
    target = _target_function(coords)

    with torch.no_grad():
        initial_loss = float(torch.mean((model(coords) - target).square()).cpu())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(coords) - target).square())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(torch.mean((model(coords) - target).square()).cpu())

    ratio = initial_loss / max(final_loss, 1.0e-16)
    details = {
        "backend": "torch_reference",
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "improvement_factor": ratio,
    }
    if not math.isfinite(final_loss) or ratio < 5.0:
        raise AssertionError(f"function fitting loss improved only x{ratio:.3g}")
    return CheckResult("function_fitting", "passed", "loss improved by at least x5", details)


def run_laplace_soft_check(out: Path, options: SanityOptions) -> CheckResult:
    steps = _bounded_steps(options.laplace_steps)
    config = _base_laplace_config(out / "laplace_soft_runs", steps, "soft")
    run_dir = train(config, run_id="laplace_soft")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    _assert_finite_metrics(metrics, {"loss_total", "boundary_rmse"})
    first = metrics.iloc[0]
    final = metrics.iloc[-1]
    details = {
        "run_dir": str(run_dir),
        "steps": int(len(metrics)),
        "initial_loss_total": float(first["loss_total"]),
        "final_loss_total": float(final["loss_total"]),
        "initial_boundary_rmse": float(first["boundary_rmse"]),
        "final_boundary_rmse": float(final["boundary_rmse"]),
    }
    if float(final["loss_total"]) >= float(first["loss_total"]):
        raise AssertionError("Laplace soft loss_total did not decrease")
    if float(final["boundary_rmse"]) >= float(first["boundary_rmse"]):
        raise AssertionError("Laplace soft boundary_rmse did not decrease")
    return CheckResult("laplace_soft", "passed", "loss and boundary_rmse decreased", details)


def _boundary_error(model: torch.nn.Module, dtype: torch.dtype) -> float:
    values = torch.linspace(0.0, 1.0, 41, dtype=dtype)
    zeros = torch.zeros_like(values)
    ones = torch.ones_like(values)
    batches = [
        (
            torch.stack([zeros, values], dim=1),
            torch.sin(math.pi * values).reshape(-1, 1),
        ),
        (torch.stack([ones, values], dim=1), torch.zeros(41, 1, dtype=dtype)),
        (torch.stack([values, zeros], dim=1), torch.zeros(41, 1, dtype=dtype)),
        (torch.stack([values, ones], dim=1), torch.zeros(41, 1, dtype=dtype)),
    ]
    with torch.no_grad():
        return max(float((model(coords) - target).abs().max().cpu()) for coords, target in batches)


class _ZeroRawModel(torch.nn.Module):
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.zeros(coords.shape[0], 1, device=coords.device, dtype=coords.dtype)


def run_laplace_hard_check(out: Path, options: SanityOptions) -> CheckResult:
    steps = _bounded_steps(options.hard_steps)
    config = _base_laplace_config(out / "laplace_hard_runs", steps, "hard")
    torch.manual_seed(options.seed)
    dtype = resolve_dtype(config["dtype"])
    problem = build_problem(config)
    model = build_model(config, problem).to(dtype=dtype)
    before_error = _boundary_error(model, dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))
    for _ in range(steps):
        batches = {
            "collocation": problem.sample_collocation(64, torch.device("cpu"), dtype),
            "boundary": problem.sample_boundary(64, torch.device("cpu"), dtype),
            "initial": problem.sample_initial(0, torch.device("cpu"), dtype),
        }
        optimizer.zero_grad(set_to_none=True)
        losses = compute_losses(model, problem, batches, config["loss"])
        losses["loss_total"].backward()
        optimizer.step()

    after_error = _boundary_error(model, dtype)
    coords = torch.rand(64, 2, dtype=dtype)
    raw_residual = problem.residual(_ZeroRawModel(), {"coords": coords}).square().mean().sqrt()
    physical_model = OutputTransformedModel(_ZeroRawModel(), HeatLaplaceHardConstraint())
    physical_residual = problem.residual(physical_model, {"coords": coords}).square().mean().sqrt()
    details = {
        "steps": steps,
        "boundary_error_before": before_error,
        "boundary_error_after": after_error,
        "raw_zero_residual_rmse": float(raw_residual.detach().cpu()),
        "physical_ansatz_residual_rmse": float(physical_residual.detach().cpu()),
    }
    if before_error > 1.0e-10 or after_error > 1.0e-10:
        raise AssertionError("Laplace hard boundary is not exact before and after training")
    if float(physical_residual.detach().cpu()) <= float(raw_residual.detach().cpu()) + 1.0e-6:
        raise AssertionError("Laplace residual did not use the hard-constraint physical ansatz")
    return CheckResult(
        "laplace_hard",
        "passed",
        "hard boundary and physical residual verified",
        details,
    )


def _get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict):
            raise KeyError(dotted_key)
        cursor = cursor[part]
    return cursor


def _override_matches(config: dict[str, Any], override: str) -> bool:
    key, raw_value = override.split("=", 1)
    return _get_nested(config, key) == parse_override_value(raw_value)


def run_benchmark_smoke_check(out: Path, options: SanityOptions) -> CheckResult:
    del options
    benchmark_dir = out / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    benchmark_runs = out / "benchmark_runs"
    jobs = load_benchmark_jobs("configs/benchmark_smoke.yaml")
    dry_run_output = io.StringIO()
    with contextlib.redirect_stdout(dry_run_output):
        run_jobs(jobs, dry_run=True, limit=5)
    selected = jobs[:5]
    if len(selected) != 5 or "[5/5]" not in dry_run_output.getvalue():
        raise AssertionError("benchmark dry-run did not show exactly 5 jobs")

    rows: list[dict[str, Any]] = []
    for job in selected:
        overrides = [
            *job["overrides"],
            f"output.root_dir={benchmark_runs}",
            "output.save_checkpoints=false",
            "output.save_predictions=false",
            "output.save_figures=false",
        ]
        config = load_config(job["base_config"], overrides=overrides)
        run_dir = train(config, run_id=str(job["name"]))
        resolved = load_config(run_dir / "resolved_config.yaml")
        fixed_ok = all(_override_matches(resolved, override) for override in job["overrides"])
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "job_name": job["name"],
                "run_dir": str(run_dir),
                "status": str(summary.get("status", "unknown")),
                "fixed_overrides_present": fixed_ok,
            }
        )

    run_index = pd.DataFrame(rows)
    run_index.to_csv(benchmark_dir / "run_index.csv", index=False)
    if len(run_index) != 5:
        raise AssertionError("benchmark real run did not create 5 run_index rows")
    fixed_ok = bool(run_index["fixed_overrides_present"].astype(bool).all())
    if not fixed_ok:
        raise AssertionError("not all benchmark overrides appeared in resolved_config.yaml")
    if set(run_index["status"]) != {"completed"}:
        raise AssertionError("not all benchmark smoke runs completed")
    return CheckResult(
        "benchmark_smoke",
        "passed",
        "dry-run and 5 real benchmark jobs completed",
        {"dry_run_jobs": 5, "run_index": str(benchmark_dir / "run_index.csv")},
    )


def run_analysis_smoke_check(runs: Path, out: Path) -> CheckResult:
    summary = analyze_runs(runs, out)
    required = [out / "summary.csv", out / "report.md"]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise AssertionError(f"analysis smoke missing required artifacts: {missing}")
    figures = sorted((out / "figures").glob("*.png"))
    report = (out / "report.md").read_text(encoding="utf-8")
    if not figures:
        raise AssertionError("analysis smoke created no figures")
    empty_figures = [str(path) for path in figures if path.stat().st_size == 0]
    if empty_figures:
        raise AssertionError(f"analysis smoke created empty figures: {empty_figures}")
    if summary.empty and "insufficient data" not in report:
        raise AssertionError("analysis smoke had no data but did not report insufficient data")
    return CheckResult(
        "analysis_smoke",
        "passed",
        "summary, report, and figures generated",
        {"rows": int(len(summary)), "figures": len(figures), "out": str(out)},
    )


def _run_check(name: str, fn: Callable[[], CheckResult]) -> CheckResult:
    try:
        return fn()
    except Exception as exc:
        return CheckResult(name, "failed", str(exc), {})


def _write_sanity_outputs(out: Path, results: list[CheckResult]) -> dict[str, Any]:
    passed = all(result.status == "passed" for result in results)
    summary = {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "checks": [asdict(result) for result in results],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sanity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# tnpinn sanity check",
        "",
        f"Status: {summary['status']}",
        "",
        "| Check | Status | Message |",
        "| --- | --- | --- |",
    ]
    for result in results:
        lines.append(f"| {result.name} | {result.status} | {result.message} |")
    lines.append("")
    lines.append("## Details")
    for result in results:
        lines.extend(
            [
                "",
                f"### {result.name}",
                "",
                "```json",
                json.dumps(result.details, indent=2),
                "```",
            ]
        )
    (out / "sanity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run_sanity_checks(out: Path, options: SanityOptions) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    results = [
        _run_check("function_fitting", lambda: run_function_fitting_check(out, options)),
        _run_check("laplace_soft", lambda: run_laplace_soft_check(out, options)),
        _run_check("laplace_hard", lambda: run_laplace_hard_check(out, options)),
        _run_check("benchmark_smoke", lambda: run_benchmark_smoke_check(out, options)),
        _run_check(
            "analysis_smoke",
            lambda: run_analysis_smoke_check(out / "benchmark_runs", out / "analysis"),
        ),
    ]
    return _write_sanity_outputs(out, results)
