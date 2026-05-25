# Prompt 00: Master prompt for Codex

Quiero que implementes este repositorio desde cero siguiendo los documentos de contexto incluidos en esta carpeta.

Lee primero, en este orden:

1. `AGENTS.md`
2. `docs/00_PROJECT_BRIEF.md`
3. `docs/01_MATH_MODEL.md`
4. `docs/02_TENSOR_NETWORK_ARCHITECTURES.md`
5. `docs/03_PINN_PROBLEMS.md`
6. `docs/04_CLI_AND_CONFIG_SPEC.md`
7. `docs/05_EXPERIMENTS_BENCHMARKS.md`
8. `docs/06_RESULTS_ANALYSIS.md`
9. `docs/07_TENSORKROWCH_IMPLEMENTATION_NOTES.md`
10. `docs/08_ACCEPTANCE_TESTS.md`

Implementa un paquete Python llamado `tnpinn`.

Necesito que al final pueda ejecutar:

```bash
python -m pip install -e ".[dev]"
pytest -q
tnpinn train --config configs/heat2d_laplace.yaml --set training.steps=10 --set training.n_collocation=64 --set training.n_boundary=64
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml --dry-run --limit 5
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 10
tnpinn analyze --runs runs --out reports/latest
```

Requisitos principales:

- Usa PyTorch y TensorKrowch.
- La implementación principal de tensor networks debe usar nodos/aristas/nodos parametrizables de TensorKrowch, no capas prehechas como `tk.models.MPSLayer`.
- Implementa al menos estas arquitecturas:
  - `global_mps`
  - `coordinate_branch_mps`
  - `binary_ttn`
  - `branched_mps`
- Implementa al menos estas feature maps:
  - `fourier_sites`
  - `polynomial_sites`
  - `rbf_sites`
- Implementa problemas:
  - `heat2d_laplace`
  - `heat2d_transient`
  - `helmholtz_antenna2d`
  - `lorenz`
- Implementa baselines:
  - `mlp`
  - `siren`
- Implementa CLI:
  - `tnpinn train`
  - `tnpinn sweep`
  - `tnpinn benchmark`
  - `tnpinn evaluate`
  - `tnpinn analyze`
  - `tnpinn plot`
- Implementa métricas, logging, checkpoints, figuras y análisis agregado.

Plan de trabajo recomendado:

1. Crea estructura de paquete, `pyproject.toml`, configs y tests mínimos.
2. Implementa configs y overrides.
3. Implementa feature maps.
4. Implementa derivadas PINN.
5. Implementa problemas físicos.
6. Implementa modelos TN y baselines.
7. Implementa training loop.
8. Implementa CLI.
9. Implementa sweeps y benchmark suite.
10. Implementa análisis de resultados.
11. Añade tests y smoke runs.

No des la tarea por completada hasta ejecutar tests y al menos un smoke run corto. Si TensorKrowch tiene una API ligeramente distinta, inspecciónala localmente y adapta el backend manteniendo la restricción de no usar capas prehechas como implementación principal.
