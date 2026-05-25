# Prompt 07: Tests, hardening y refactor final

Revisa todo el repositorio e implementa los tests y mejoras faltantes.

Objetivo:

1. Ejecutar `pytest -q`.
2. Arreglar todos los fallos.
3. Ejecutar smoke train corto.
4. Ejecutar dry-run de benchmark.
5. Ejecutar análisis de runs.
6. Revisar que la CLI documentada en README funciona.
7. Revisar que no se están usando capas prehechas de TensorKrowch como implementación principal.
8. Añadir documentación faltante.

Comandos finales esperados:

```bash
python -m pip install -e ".[dev]"
pytest -q

tnpinn train \
  --config configs/heat2d_laplace.yaml \
  --set training.steps=10 \
  --set training.n_collocation=64 \
  --set training.n_boundary=64 \
  --set eval.grid_size=16 \
  --set eval.eval_every=5

tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 10
tnpinn analyze --runs runs --out reports/latest
```

Entrega final esperada de Codex:

- resumen de archivos creados;
- comandos ejecutados;
- estado de tests;
- advertencias técnicas si alguna parte queda experimental.
