import json
import os
import subprocess
import sys
from pathlib import Path


def _run_cli(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    src = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", *args],
        check=True,
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )


def test_train_sweep_benchmark_and_analyze_cli(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"

    _run_cli(
        [
            "tnpinn.cli.train",
            "--config",
            "configs/heat2d_laplace.yaml",
            "--set",
            "training.steps=2",
            "--set",
            "training.n_collocation=16",
            "--set",
            "training.n_boundary=16",
            "--set",
            "eval.grid_size=8",
            "--set",
            f"output.root_dir={runs_dir}",
            "--set",
            "output.save_predictions=false",
        ],
        tmp_path,
    )

    summaries = list(runs_dir.glob("heat2d_laplace/*/summary.json"))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert (summaries[0].parent / "config.yaml").exists()
    assert (summaries[0].parent / "resolved_config.yaml").exists()
    assert (summaries[0].parent / "metrics.csv").exists()
    assert (summaries[0].parent / "checkpoint_last.pt").exists()
    assert (summaries[0].parent / "figures").exists()

    _run_cli(
        [
            "tnpinn.cli.sweep",
            "--config",
            "configs/sweeps/heat2d_bond_sites_arch.yaml",
            "--dry-run",
            "--limit",
            "3",
        ],
        tmp_path,
    )
    _run_cli(
        [
            "tnpinn.cli.benchmark",
            "--suite",
            "configs/benchmark_all.yaml",
            "--dry-run",
            "--limit",
            "3",
        ],
        tmp_path,
    )
    _run_cli(
        ["tnpinn.cli.analyze", "--runs", str(runs_dir), "--out", str(reports_dir)],
        tmp_path,
    )

    assert (reports_dir / "summary.csv").exists()
    assert (reports_dir / "report.md").exists()
