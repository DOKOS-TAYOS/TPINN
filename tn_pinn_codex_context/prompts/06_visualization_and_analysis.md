# Prompt 06: Visualización y análisis de resultados

Implementa visualización y análisis agregado siguiendo:

- `docs/06_RESULTS_ANALYSIS.md`

Objetivo:

1. Implementar `results/aggregate.py`.
2. Implementar `results/analysis.py`.
3. Implementar `results/reports.py`.
4. Implementar `tnpinn analyze`.
5. Mejorar figuras por problema:
   - calor: superficie 3D, referencia, error, residual;
   - Helmholtz: real, imaginario, amplitud, fase, residual;
   - Lorenz: trayectoria 3D, componentes temporales, error temporal.
6. Generar `reports/latest/report.md`.

Debe funcionar:

```bash
tnpinn analyze --runs runs --out reports/latest
```

Debe producir:

```text
reports/latest/
├── summary.csv
├── best_by_problem.csv
├── best_by_architecture.csv
├── failed_runs.csv
├── figures/
└── report.md
```

Criterios:

- No fallar si un run está incompleto.
- Marcar runs incompletos en `failed_runs.csv`.
- Usar matplotlib.
- Una figura por archivo.
