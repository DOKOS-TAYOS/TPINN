# Prompt 05: Sweeps y benchmark completo

Implementa sweeps y benchmark suite siguiendo:

- `docs/05_EXPERIMENTS_BENCHMARKS.md`
- `docs/04_CLI_AND_CONFIG_SPEC.md`

Objetivo:

1. Implementar expansión de grid de sweeps.
2. Implementar `tnpinn sweep`.
3. Implementar `tnpinn benchmark`.
4. Implementar `--dry-run`, `--limit`, `--resume`, `--max-parallel`.
5. Implementar `configs/benchmark_all.yaml`.
6. Implementar `configs/benchmark_smoke.yaml`.
7. Asegurar que se pueda variar:
   - `model.feature_map.n_sites_per_coord.*`
   - `model.tensor_network.architecture`
   - `model.tensor_network.bond_dim`
   - `model.feature_map.kind`
   - `model.family` o baseline.

Debe funcionar:

```bash
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml --dry-run --limit 10
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 20
tnpinn benchmark --suite configs/benchmark_smoke.yaml
```

Criterios:

- `dry-run` debe listar los jobs con overrides resueltos.
- `resume` debe saltar runs completados.
- Cada run debe incluir metadata suficiente para análisis posterior.
