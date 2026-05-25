# AGENTS.md

## Misión del repositorio

Implementa un paquete Python llamado `tnpinn` para experimentar con **physics-informed neural networks basadas en tensor networks**. El aproximador funcional principal no debe ser una MLP clásica, sino una tensor network entrenable que reciba sites procedentes de una feature map diferenciable.

El repositorio debe permitir:

- entrenar un experimento individual por CLI;
- lanzar sweeps sobre número de sites, bond dimension, arquitectura y feature map;
- ejecutar un benchmark completo con un solo comando;
- guardar métricas, checkpoints, predicciones y figuras por ejecución;
- analizar todos los runs y generar tablas y gráficos comparativos.

## Restricciones no negociables

1. Todo debe estar en Python.
2. Usar PyTorch para tensores, autograd y entrenamiento.
3. Usar TensorKrowch para el backend principal de tensor networks.
4. No usar `tk.models.MPSLayer`, `tk.models.MPS`, `tk.models.PEPS` ni capas tensoriales prehechas como implementación principal.
5. Construir las tensor networks con componentes de bajo nivel de TensorKrowch: nodos, aristas, nodos parametrizables y contracciones.
6. Permitir un backend `torch_einsum_reference` solo como referencia, test o fallback de depuración, pero el backend por defecto debe ser `tensorkrowch_nodes`.
7. El modelo completo debe ser diferenciable respecto a las coordenadas de entrada.
8. Deben existir tests para primeras y segundas derivadas.
9. No hacer notebooks como entrypoint principal.
10. No ocultar el entrenamiento dentro de scripts ad hoc. La CLI debe ser la interfaz principal.

## Stack recomendado

- Python >= 3.10.
- PyTorch.
- TensorKrowch.
- NumPy.
- SciPy, opcional pero recomendable para referencias numéricas.
- pandas.
- matplotlib.
- PyYAML u OmegaConf para configs.
- Typer para CLI.
- pytest.
- rich, opcional, para logs.
- tqdm, opcional.

## Estructura esperada

```text
src/tnpinn/
├── __init__.py
├── config/
│   ├── loading.py
│   ├── schema.py
│   └── overrides.py
├── tn/
│   ├── feature_maps.py
│   ├── specs.py
│   ├── tensorkrowch_backend.py
│   ├── torch_reference_backend.py
│   ├── mps_branch.py
│   ├── ttn.py
│   ├── branched_mps.py
│   ├── models.py
│   └── diagnostics.py
├── baselines/
│   ├── mlp.py
│   └── siren.py
├── pinn/
│   ├── derivatives.py
│   ├── losses.py
│   ├── sampling.py
│   ├── trainer.py
│   └── evaluator.py
├── problems/
│   ├── base.py
│   ├── heat2d.py
│   ├── helmholtz_antenna2d.py
│   ├── lorenz.py
│   └── reference_solvers.py
├── viz/
│   ├── heat.py
│   ├── helmholtz.py
│   ├── lorenz.py
│   ├── training.py
│   └── comparison.py
├── results/
│   ├── aggregate.py
│   ├── analysis.py
│   └── reports.py
├── io/
│   ├── run_dirs.py
│   ├── checkpoints.py
│   └── logging.py
└── cli/
    ├── train.py
    ├── sweep.py
    ├── benchmark.py
    ├── evaluate.py
    ├── analyze.py
    └── plot.py
```

## CLI requerida

Deben funcionar estos comandos tras instalar el paquete en modo editable:

```bash
tnpinn train --config configs/heat2d_laplace.yaml
tnpinn train --config configs/heat2d_laplace.yaml --set model.feature_map.n_sites_per_coord.x=12 --set model.tensor_network.bond_dim=16
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml
tnpinn benchmark --suite configs/benchmark_all.yaml
tnpinn analyze --runs runs --out reports/latest
tnpinn plot --run runs/<problem>/<run_id>
```

Y también deben funcionar por módulo:

```bash
python -m tnpinn.cli.train --config configs/heat2d_laplace.yaml
python -m tnpinn.cli.benchmark --suite configs/benchmark_all.yaml
```

## Problemas físicos requeridos

Implementar como mínimo:

1. `heat2d_laplace`: Laplace/calor estacionario 2D con solución analítica.
2. `heat2d_transient`: calor transitorio 2D.
3. `helmholtz_antenna2d`: Helmholtz escalar 2D complejo, representado por dos canales reales.
4. `lorenz`: sistema de Lorenz como ODE PINN.

## Arquitecturas tensoriales requeridas

Implementar como mínimo:

1. `coordinate_branch_mps`: una rama MPS por coordenada física y merges tensoriales jerárquicos.
2. `binary_ttn`: árbol binario sobre todos los sites.
3. `branched_mps`: MPS locales por grupo de sites y MPS superior sobre los latentes.
4. `global_mps`: MPS simple sobre todos los sites, útil como baseline tensorial.

## Baselines requeridos

Implementar:

1. `mlp`.
2. `siren`.

## Feature maps requeridas

Implementar:

1. `fourier_sites`.
2. `polynomial_sites`.
3. `rbf_sites`, aunque puede ser experimental.

La configuración debe exponer:

```yaml
model:
  feature_map:
    kind: fourier_sites
    site_dim: 2
    n_sites_per_coord:
      x: 8
      y: 8
      t: 8
```

## Artefactos de salida por run

Cada run debe crear una carpeta:

```text
runs/<problem>/<timestamp_or_id>/
├── resolved_config.yaml
├── metrics.csv
├── summary.json
├── checkpoint_last.pt
├── checkpoint_best.pt
├── predictions.npz
└── figures/
```

## Métricas mínimas

Guardar en `metrics.csv`:

```text
step
loss_total
loss_residual
loss_boundary
loss_initial
loss_data
loss_regularization
residual_rmse
boundary_rmse
initial_rmse
relative_l2
max_abs_error
grad_norm
num_parameters
forward_time_s
backward_time_s
step_time_s
```

Añadir métricas tensoriales si aplica:

```text
bond_dim
n_sites_total
site_dim
architecture
feature_map
tn_backend
core_norm_mean
core_norm_max
effective_rank_mean
```

## Tests mínimos

Deben pasar:

```bash
pytest -q
```

Con tests para:

- parsing de configs y overrides;
- shapes de feature maps;
- shapes de modelos tensoriales;
- gradiente respecto a parámetros;
- gradiente respecto a coordenadas;
- segunda derivada para Laplace;
- residual de heat2d con solución analítica;
- residual de Lorenz con una trayectoria fabricada o caso simple;
- CLI dry-run de train, sweep y benchmark;
- creación de run dir y escritura de métricas.

## Criterio de calidad

No implementes una solución superficial. Antes de dar una tarea por terminada:

1. Ejecuta tests.
2. Ejecuta al menos un entrenamiento smoke de pocas iteraciones.
3. Comprueba que se generan métricas y figuras.
4. Comprueba que `tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run` lista todos los jobs.
5. Comprueba que `tnpinn analyze --runs runs --out reports/latest` produce `summary.csv` aunque los runs sean smoke.

## Notas de implementación

Si la API exacta de TensorKrowch difiere, inspecciona la instalación local con:

```bash
python - <<'PY'
import tensorkrowch as tk
print(tk.__version__ if hasattr(tk, "__version__") else "unknown")
print([x for x in dir(tk) if "Node" in x or "Network" in x])
PY
```

Adapta el backend a la API instalada, pero mantén la restricción: no usar capas prehechas como implementación principal.
