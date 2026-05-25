# Prompt 04: Trainer, evaluación, configs y CLI real

Implementa entrenamiento completo siguiendo:

- `docs/04_CLI_AND_CONFIG_SPEC.md`
- `docs/08_ACCEPTANCE_TESTS.md`

Objetivo:

1. Implementar `pinn/trainer.py`.
2. Implementar `pinn/evaluator.py`.
3. Implementar logging en `metrics.csv`.
4. Implementar checkpoints `checkpoint_last.pt` y `checkpoint_best.pt`.
5. Implementar `summary.json`.
6. Implementar `predictions.npz`.
7. Implementar figuras básicas por problema.
8. Conectar `tnpinn train` a entrenamiento real.
9. Conectar `tnpinn evaluate`.
10. Conectar `tnpinn plot`.

Debe funcionar:

```bash
tnpinn train \
  --config configs/heat2d_laplace.yaml \
  --set training.steps=20 \
  --set training.n_collocation=128 \
  --set training.n_boundary=64 \
  --set eval.eval_every=10 \
  --set eval.grid_size=32
```

Debe crear:

```text
runs/heat2d_laplace/<run_id>/
├── resolved_config.yaml
├── metrics.csv
├── summary.json
├── checkpoint_last.pt
├── checkpoint_best.pt
├── predictions.npz
└── figures/
```

Tests requeridos:

```bash
pytest -q tests/test_training_smoke.py tests/test_run_outputs.py
```

Criterios:

- Si ocurre una excepción durante el entrenamiento, `summary.json` debe registrar `status: failed`.
- El entrenamiento debe ser reproducible con seed.
- Debe poder correr en CPU.
