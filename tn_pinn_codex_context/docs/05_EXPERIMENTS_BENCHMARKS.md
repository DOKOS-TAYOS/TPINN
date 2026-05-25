# 05. Experiments and benchmarks

## Objetivo

El benchmark debe comparar:

- arquitectura tensorial;
- número de sites;
- bond dimension;
- feature map;
- baseline.

## Suite principal

`configs/benchmark_all.yaml` debe definir una lista de sweeps.

Ejemplo conceptual:

```yaml
suite:
  name: benchmark_all
  output_root: runs
  max_parallel: 1

  sweeps:
    - name: heat2d_laplace_core
      base_config: configs/heat2d_laplace.yaml
      grid:
        model.feature_map.n_sites_per_coord.x: [4, 8, 12]
        model.feature_map.n_sites_per_coord.y: [4, 8, 12]
        model.tensor_network.architecture: [global_mps, coordinate_branch_mps, binary_ttn]
        model.tensor_network.bond_dim: [2, 4, 8, 16]

    - name: lorenz_core
      base_config: configs/lorenz.yaml
      grid:
        model.feature_map.n_sites_per_coord.t: [8, 16, 24]
        model.tensor_network.architecture: [global_mps, binary_ttn, branched_mps]
        model.tensor_network.bond_dim: [2, 4, 8, 16]

    - name: helmholtz_core
      base_config: configs/helmholtz_antenna2d.yaml
      grid:
        problem.k: [3.141592653589793, 6.283185307179586]
        model.feature_map.n_sites_per_coord.x: [8, 12]
        model.feature_map.n_sites_per_coord.y: [8, 12]
        model.tensor_network.architecture: [coordinate_branch_mps, binary_ttn]
        model.tensor_network.bond_dim: [4, 8, 16]
```

## Smoke benchmark

También debe existir un modo rápido:

```bash
tnpinn benchmark --suite configs/benchmark_all.yaml --limit 3
```

o un suite pequeño:

```bash
tnpinn benchmark --suite configs/benchmark_smoke.yaml
```

## Naming de runs

Cada job debe etiquetarse con metadata:

```json
{
  "problem": "heat2d_laplace",
  "architecture": "coordinate_branch_mps",
  "feature_map": "fourier_sites",
  "n_sites_total": 16,
  "bond_dim": 8,
  "seed": 1234
}
```

## Comparaciones mínimas

El análisis debe responder:

1. Mejor arquitectura por problema.
2. Error vs bond dimension.
3. Error vs número total de sites.
4. Tiempo por step vs arquitectura.
5. Número de parámetros vs error.
6. Comparación TN vs MLP/SIREN.

## Métricas agregadas

`summary.csv` debe incluir por run:

```text
run_id
run_dir
problem
model_family
architecture
feature_map
n_sites_total
site_dim
bond_dim
num_parameters
seed
best_step
final_step
best_loss_total
final_loss_total
best_residual_rmse
final_residual_rmse
best_relative_l2
final_relative_l2
time_total_s
mean_step_time_s
status
```

## Reanudación

`--resume` debe saltar jobs con `summary.json` finalizado y status `completed`.

## Estados de run

Usar:

```text
created
running
completed
failed
interrupted
```

Guardar `status` en `summary.json`.
