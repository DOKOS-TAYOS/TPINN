# Prompt 01: Bootstrap del repositorio

Implementa la primera capa del repositorio siguiendo `AGENTS.md` y `docs/04_CLI_AND_CONFIG_SPEC.md`.

Objetivo de esta fase:

1. Crear `pyproject.toml`.
2. Crear estructura `src/tnpinn`.
3. Crear entrypoints CLI con Typer.
4. Crear loader de configs YAML con overrides dotlist.
5. Crear configs base en `configs/`.
6. Crear tests iniciales de config y CLI dry-run.
7. Crear README inicial del proyecto implementado.

No implementes todavía el entrenamiento completo. En esta fase pueden existir stubs, pero deben estar bien definidos.

Debe funcionar:

```bash
python -m pip install -e ".[dev]"
tnpinn train --config configs/heat2d_laplace.yaml --dry-run
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml --dry-run --limit 3
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 3
pytest -q
```

Criterios:

- `--set key=value` debe modificar configs anidadas.
- `--dry-run` debe imprimir config resuelta y no entrenar.
- Los comandos deben estar conectados aunque algunas funciones profundas sean stubs.
- Añade tests en `tests/test_config.py` y `tests/test_cli_dryrun.py`.
