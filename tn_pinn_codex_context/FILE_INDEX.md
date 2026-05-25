# File index

## Core guidance

- `AGENTS.md`: instrucciones persistentes para Codex dentro del repo.
- `docs/00_PROJECT_BRIEF.md`: objetivo general.
- `docs/01_MATH_MODEL.md`: formulación matemática.
- `docs/02_TENSOR_NETWORK_ARCHITECTURES.md`: arquitecturas tensoriales.
- `docs/03_PINN_PROBLEMS.md`: problemas físicos.
- `docs/04_CLI_AND_CONFIG_SPEC.md`: CLI y YAMLs.
- `docs/05_EXPERIMENTS_BENCHMARKS.md`: sweeps y benchmark completo.
- `docs/06_RESULTS_ANALYSIS.md`: análisis agregado.
- `docs/07_TENSORKROWCH_IMPLEMENTATION_NOTES.md`: backend TensorKrowch.
- `docs/08_ACCEPTANCE_TESTS.md`: criterios de aceptación.

## Prompts

- `prompts/00_master_prompt.md`: prompt único de implementación completa.
- `prompts/01_bootstrap_repo.md`: estructura, configs y CLI dry-run.
- `prompts/02_feature_maps_and_tn_models.md`: feature maps y TNs.
- `prompts/03_pinn_problems_and_derivatives.md`: problemas físicos y derivadas.
- `prompts/04_training_cli_configs.md`: trainer y salidas.
- `prompts/05_benchmarks_and_sweeps.md`: sweeps y benchmark.
- `prompts/06_visualization_and_analysis.md`: análisis y visualización.
- `prompts/07_tests_refactor_hardening.md`: cierre, tests y hardening.

## Configs

- `configs/heat2d_laplace.yaml`
- `configs/heat2d_transient.yaml`
- `configs/helmholtz_antenna2d.yaml`
- `configs/lorenz.yaml`
- `configs/benchmark_all.yaml`
- `configs/benchmark_smoke.yaml`
- `configs/sweeps/*.yaml`

## Templates

- `templates/pyproject.toml.template`
- `templates/gitignore.template`

## Codex support

- `codex/CHECKLIST.md`
- `codex/RUNBOOK.md`
