# Changelog

## 0.1.0 - 2026-05-25

- Bootstrapped the `tnpinn` package, CLI, configs, tests, and smoke workflows.
- Fixed audit P1/P2 issues: sweep/benchmark `fixed:` overrides, result aggregation
  source-of-truth handling, invalid run classification, loss-curve reporting, and
  dual `config.yaml` / `resolved_config.yaml` run output.
- Added physical validation tests for Laplace residuals, Relative L2 semantics,
  hard-boundary constraints, autograd continuity, Helmholtz manufactured residuals,
  and minimum Laplace training improvement.
- Renamed the active TensorKrowch backend to `tensorkrowch_hybrid`, reserved
  `tensorkrowch_full` with an explicit `NotImplementedError`, and documented the
  backend status with tests that prevent accidental misrepresentation.
- Added `tnpinn sanity-check` to run short function-fitting, Laplace, benchmark,
  and analysis smoke checks with `sanity_report.md` and `sanity_summary.json`.
- Fixed plain `pytest -q` collection by adding `src` to pytest's configured
  Python path.
- Removed the repo-local `.pytest_tmp` base temp setting so Windows test runs do
  not fail when that directory is locked by another process.
- Added a local pytest plugin that gives each test run a unique process-specific
  temp base, avoiding locked Windows temp roots.
