# Audit Fixes

Date: 2026-05-25

## Scope

This pass fixes the bugs called out in `AUDIT.md` before adding new functionality.
The changes are intentionally limited to sweep/benchmark expansion, result aggregation,
run validation, loss-curve reporting, and run config output.

## Bugs Fixed

### 1. `fixed:` Overrides In Sweeps And Benchmarks

Root cause: `load_sweep_jobs` and `load_benchmark_jobs` expanded only `grid:` values.
The `fixed:` section was parsed from YAML but never merged into each job.

Fix:

- Added `fixed:` override formatting for every expanded job.
- Kept `grid:` as the Cartesian product source.
- Added a conflict check so a key cannot appear in both `fixed:` and `grid:`.
- Conflict errors include the duplicated key name.

Tests added:

- `test_load_sweep_jobs_applies_fixed_to_every_grid_job`
- `test_load_benchmark_jobs_applies_fixed_and_rejects_conflicts`
- `test_benchmark_smoke_jobs_include_fixed_overrides`

### 2. Result Aggregation Source Of Truth

Root cause: `collect_runs` loaded `summary.json`, then recalculated best metrics from
`metrics.csv` and overwrote `best_step`, `best_loss_total`, and related values. When
`relative_l2` was constant, the first row could be selected even when `summary.json`
said another step was best.

Fix:

- `summary.json` is now the primary source when it exists and parses correctly.
- Metrics are still validated, but no longer overwrite summary best/final fields.
- If `summary.json` is missing, aggregation falls back to `metrics.csv`, reconstructs
  best/final fields from `loss_total`, and marks the run as `incomplete`.
- Added `record_source` and `status_reason` columns to make fallback/validation visible.

Tests added:

- `test_collect_runs_uses_summary_as_primary_source_when_relative_l2_is_constant`
- `test_collect_runs_falls_back_to_metrics_without_summary_and_marks_incomplete`

### 3. Invalid Run Classification

Root cause: aggregation only scanned `summary.json` files and did not validate metrics
before keeping a run as completed.

Fix:

- Aggregation now scans run directories from both `summary.json` and `metrics.csv`.
- A completed summary with missing metrics becomes `incomplete`.
- Empty metrics or missing required metric columns become `invalid_metrics`.
- `failed_runs.csv` remains available.
- New `invalid_runs.csv` is written for `invalid_metrics` and `incomplete` runs.

Tests added:

- `test_collect_runs_classifies_invalid_metrics_and_missing_metrics`
- `test_analyze_runs_writes_invalid_runs_and_real_loss_curve_for_one_and_many_runs`
- `test_analyze_runs_marks_loss_curve_insufficient_when_metrics_empty`

### 4. `loss_curves_best_runs.png`

Root cause: the report created and saved `loss_curves_best_runs.png` without plotting
anything.

Fix:

- The figure now plots `loss_total` versus `step` from real `metrics.csv` files for
  completed runs.
- If no completed run has at least two metric rows, the image contains the text
  `insufficient data`.
- The cause is also written to `report.md`.
- Matplotlib now uses the non-interactive `Agg` backend in report/figure modules, which
  avoids Tk failures on this Windows environment.

Tests added:

- `test_analyze_runs_writes_invalid_runs_and_real_loss_curve_for_one_and_many_runs`
- `test_analyze_runs_marks_loss_curve_insufficient_when_metrics_empty`

### 5. Run Config Output

Root cause: runs wrote `resolved_config.yaml` only. Some checks and user workflows expect
both `config.yaml` and `resolved_config.yaml`.

Fix:

- Training now writes both `config.yaml` and `resolved_config.yaml`.
- At this stage they contain the same resolved configuration.

Test updated:

- `test_train_sweep_benchmark_and_analyze_cli`

## Commands Executed

During development:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_experiments.py tests\test_results_analysis.py tests\test_cli_smoke.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_experiments.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_results_analysis.py tests\test_cli_smoke.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\tnpinn.exe benchmark --suite configs\benchmark_smoke.yaml --dry-run
```

Final verification commands are listed in the assistant response for this turn.

