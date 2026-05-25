# 04. CLI and configuration specification

## Principio

Toda ejecución debe partir de un YAML y permitir overrides tipo dotlist por CLI.

Ejemplo:

```bash
tnpinn train \
  --config configs/heat2d_laplace.yaml \
  --set model.feature_map.n_sites_per_coord.x=12 \
  --set model.feature_map.n_sites_per_coord.y=12 \
  --set model.tensor_network.bond_dim=16 \
  --set training.steps=5000
```

## Comandos

### `tnpinn train`

Entrena un experimento.

```bash
tnpinn train --config configs/heat2d_laplace.yaml
```

Opciones mínimas:

```text
--config PATH
--set KEY=VALUE, repetible
--run-id TEXT, opcional
--dry-run
--device cpu|cuda|auto
--seed INT
```

`--dry-run` debe:

- cargar config;
- resolver overrides;
- construir problema y modelo;
- imprimir número de parámetros;
- imprimir run dir previsto;
- no entrenar.

### `tnpinn sweep`

Ejecuta una rejilla de experimentos.

```bash
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml
```

Opciones:

```text
--config PATH
--max-parallel INT
--dry-run
--limit INT
--resume
```

El sweep debe expandir una rejilla como:

```yaml
base_config: configs/heat2d_laplace.yaml
sweep:
  name: heat2d_bond_sites_arch
  grid:
    model.feature_map.n_sites_per_coord.x: [4, 8, 12]
    model.feature_map.n_sites_per_coord.y: [4, 8, 12]
    model.tensor_network.architecture: [global_mps, coordinate_branch_mps, binary_ttn]
    model.tensor_network.bond_dim: [2, 4, 8, 16]
```

### `tnpinn benchmark`

Ejecuta todos los benchmarks del suite.

```bash
tnpinn benchmark --suite configs/benchmark_all.yaml
```

Opciones:

```text
--suite PATH
--max-parallel INT
--dry-run
--limit INT
--resume
```

### `tnpinn evaluate`

Evalúa un checkpoint o run.

```bash
tnpinn evaluate --run runs/heat2d_laplace/20260525_001
```

Opciones:

```text
--run PATH
--checkpoint best|last|PATH
--grid-size INT
--save-predictions
--make-figures
```

### `tnpinn analyze`

Agrega runs y genera reportes.

```bash
tnpinn analyze --runs runs --out reports/latest
```

Debe producir:

```text
reports/latest/
├── summary.csv
├── best_by_problem.csv
├── best_by_architecture.csv
├── figures/
│   ├── rel_l2_by_architecture.png
│   ├── residual_by_bond_dim.png
│   ├── runtime_by_architecture.png
│   └── sites_vs_error.png
└── report.md
```

### `tnpinn plot`

Regenera figuras de un run.

```bash
tnpinn plot --run runs/lorenz/20260525_001
```

## Estructura de config

```yaml
seed: 1234
device: auto
dtype: float64

problem:
  name: heat2d_laplace
  domain:
    x: [0.0, 1.0]
    y: [0.0, 1.0]

model:
  family: tensor_network
  baseline: null

  feature_map:
    kind: fourier_sites
    site_dim: 2
    n_sites_per_coord:
      x: 8
      y: 8
    frequency_scale: 3.141592653589793
    frequency_mode: powers_of_two
    trainable_frequencies: false
    include_identity_site: false

  tensor_network:
    backend: tensorkrowch_nodes
    architecture: coordinate_branch_mps
    bond_dim: 8
    branch_factor: 2
    share_blocks: false
    init_std: 0.05
    normalize_cores: false

training:
  optimizer: adam
  lr: 1.0e-3
  steps: 20000
  n_collocation: 4096
  n_boundary: 1024
  n_initial: 0
  resample_every: 10
  grad_clip: 10.0
  lbfgs_after_adam: false

loss:
  residual: 1.0
  boundary: 10.0
  initial: 10.0
  data: 0.0
  regularization: 1.0e-8

eval:
  grid_size: 128
  eval_every: 500
  save_figures_every: 1000

output:
  root_dir: runs
  save_checkpoints: true
  save_predictions: true
  save_figures: true
```

## Overrides

Implementar conversión robusta de strings:

```text
"true" -> bool
"false" -> bool
"null" -> None
"12" -> int
"1.0e-3" -> float
"[1,2,3]" -> list if es YAML válido
```

## Run IDs

Si no se pasa `--run-id`, crear:

```text
YYYYMMDD_HHMMSS_<short_hash>
```

El short hash debe depender de la config resuelta.
