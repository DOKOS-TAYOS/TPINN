# Auditoria tecnica del repositorio TPINN

Fecha: 2026-05-25  
Entorno: Windows / PowerShell / `.venv` local del proyecto.

## 1. Arbol actual del repositorio

Vista de alto nivel observada con `Get-ChildItem -Force`:

```text
.
|-- .git/
|-- .pytest_cache/
|-- .pytest_tmp/
|-- .ruff_cache/
|-- .venv/
|-- configs/
|-- reports/
|-- runs/
|-- src/
|-- tests/
|-- tn_pinn_codex_context/
|-- .gitignore
|-- CHANGELOG.md
|-- pyproject.toml
`-- README.md
```

Arbol relevante de ficheros, excluyendo caches y `.venv`:

```text
.
|-- pyproject.toml
|-- CHANGELOG.md
|-- README.md
|-- configs/
|   |-- benchmark_all.yaml
|   |-- benchmark_smoke.yaml
|   |-- heat2d_laplace.yaml
|   |-- heat2d_transient.yaml
|   |-- helmholtz_antenna2d.yaml
|   |-- lorenz.yaml
|   `-- sweeps/
|       |-- heat2d_bond_sites_arch.yaml
|       |-- helmholtz_bond_sites_arch.yaml
|       `-- lorenz_bond_sites_arch.yaml
|-- src/tnpinn/
|   |-- __init__.py
|   |-- experiments.py
|   |-- models.py
|   |-- baselines/
|   |   |-- mlp.py
|   |   `-- siren.py
|   |-- cli/
|   |   |-- analyze.py
|   |   |-- benchmark.py
|   |   |-- evaluate.py
|   |   |-- main.py
|   |   |-- plot.py
|   |   |-- sweep.py
|   |   `-- train.py
|   |-- config/
|   |   |-- loading.py
|   |   |-- overrides.py
|   |   `-- schema.py
|   |-- io/
|   |   |-- checkpoints.py
|   |   |-- logging.py
|   |   `-- run_dirs.py
|   |-- pinn/
|   |   |-- derivatives.py
|   |   |-- evaluator.py
|   |   |-- losses.py
|   |   `-- trainer.py
|   |-- problems/
|   |   |-- base.py
|   |   |-- heat2d.py
|   |   |-- helmholtz_antenna2d.py
|   |   `-- lorenz.py
|   |-- results/
|   |   |-- aggregate.py
|   |   |-- analysis.py
|   |   `-- reports.py
|   |-- tn/
|   |   |-- diagnostics.py
|   |   |-- feature_maps.py
|   |   |-- models.py
|   |   |-- tensorkrowch_backend.py
|   |   `-- torch_reference_backend.py
|   `-- viz/
|       |-- heat.py
|       `-- training.py
|-- tests/
|   |-- test_cli_smoke.py
|   |-- test_config.py
|   |-- test_derivatives.py
|   |-- test_feature_maps.py
|   |-- test_models.py
|   `-- test_problems.py
|-- runs/
|   `-- heat2d_laplace/
|       |-- 20260525_205249_b5b534d7/
|       `-- 20260525_205743_b5b534d7/
|-- reports/
|   `-- latest/
`-- tn_pinn_codex_context/
```

## 2. Modulos implementados, stubs e incompletos

Implementado con funcionalidad real:

- `tnpinn.config`: carga YAML, defaults basicos y overrides dotlist.
- `tnpinn.tn.feature_maps`: `fourier_sites`, `polynomial_sites`, `rbf_sites`.
- `tnpinn.pinn.derivatives`: gradiente, derivada parcial, segunda derivada y laplaciano.
- `tnpinn.problems`: `heat2d_laplace`, `heat2d_transient`, `helmholtz_antenna2d`, `lorenz`.
- `tnpinn.baselines`: MLP y SIREN.
- `tnpinn.tn.tensorkrowch_backend`: arquitecturas `global_mps`, `coordinate_branch_mps`, `binary_ttn`, `branched_mps`.
- `tnpinn.tn.torch_reference_backend`: backend de referencia con `torch.einsum`.
- `tnpinn.pinn.trainer`: entrenamiento Adam, metricas, checkpoints, predicciones y figuras basicas.
- `tnpinn.cli`: comandos `train`, `sweep`, `benchmark`, `evaluate`, `analyze`, `plot`.
- `tnpinn.results`: agregacion y reportes basicos.

Stubs, placeholders o partes incompletas:

- `src/tnpinn/problems/base.py` tiene metodos abstractos con `raise NotImplementedError` para `sample_collocation` y `residual`. Esto es normal para una clase base, pero aparece en la busqueda.
- `src/tnpinn/tn/tensorkrowch_backend.py` tiene `_contract_projected_mps(...)` con `raise NotImplementedError` y no se usa.
- `src/tnpinn/results/analysis.py` tiene un `pass` al ignorar errores al guardar Parquet.
- Faltan ficheros esperados por el briefing: `pinn/sampling.py`, `problems/reference_solvers.py`, `viz/helmholtz.py`, `viz/lorenz.py`, `viz/comparison.py`, y separacion explicita `tn/mps_branch.py`, `tn/ttn.py`, `tn/branched_mps.py`.
- `viz` solo cubre curvas de entrenamiento y heatmap de calor. No hay figuras especificas para Helmholtz ni Lorenz.
- `loss_data` siempre es cero; no hay flujo de datos supervisados.
- `eval_every`, `save_figures_every` y `resample_every` no se respetan realmente: el entrenamiento remuestrea y evalua metricas en cada step.
- `fixed:` en configs de benchmark/sweep no se aplica. Solo se expanden las claves de `grid`.
- El run guarda `resolved_config.yaml`, no `config.yaml`.

## 3. Resultado de comandos solicitados

### `grep -R "pass\|TODO\|NotImplemented\|placeholder\|dummy\|fake" src tests`

Resultado literal en este PowerShell:

```text
grep : El termino 'grep' no se reconoce como nombre de un cmdlet, funcion,
archivo de script o programa ejecutable.
```

Como equivalente diagnostico use:

```powershell
rg -n "pass|TODO|NotImplemented|placeholder|dummy|fake" src tests
```

Resultado:

```text
src\tnpinn\results\analysis.py:178:        pass
src\tnpinn\problems\base.py:23:        raise NotImplementedError
src\tnpinn\problems\base.py:36:        raise NotImplementedError
src\tnpinn\tn\tensorkrowch_backend.py:61:        raise NotImplementedError
```

### `pytest -q`

Ejecutado como `.venv\Scripts\python.exe -m pytest -q`:

```text
.................                                                        [100%]
17 passed in 13.40s
```

### `tnpinn --help`

Resultado literal en shell no activada:

```text
tnpinn : El termino 'tnpinn' no se reconoce como nombre de un cmdlet, funcion,
archivo de script o programa ejecutable.
```

Equivalente dentro de la `.venv`:

```powershell
.\.venv\Scripts\tnpinn.exe --help
```

Resultado:

```text
usage: tnpinn [-h] {train,sweep,benchmark,evaluate,analyze,plot} ...

positional arguments:
  {train,sweep,benchmark,evaluate,analyze,plot}
    train               train one experiment
    sweep               run or list a sweep
    benchmark           run or list a benchmark suite
    evaluate            evaluate one run
    analyze             aggregate runs
    plot                regenerate run figures

options:
  -h, --help            show this help message and exit
```

### `tnpinn benchmark --suite configs/benchmark_smoke.yaml --dry-run`

Ejecutado como `.venv\Scripts\tnpinn.exe benchmark --suite configs/benchmark_smoke.yaml --dry-run`:

```text
[1/5] heat2d_laplace_smoke_0001
  base_config: configs/heat2d_laplace.yaml
  overrides: ['model.feature_map.n_sites_per_coord.x=4', 'model.feature_map.n_sites_per_coord.y=4', 'model.tensor_network.architecture=global_mps', 'model.tensor_network.bond_dim=2']
[2/5] heat2d_laplace_smoke_0002
  base_config: configs/heat2d_laplace.yaml
  overrides: ['model.feature_map.n_sites_per_coord.x=4', 'model.feature_map.n_sites_per_coord.y=4', 'model.tensor_network.architecture=global_mps', 'model.tensor_network.bond_dim=4']
[3/5] heat2d_laplace_smoke_0003
  base_config: configs/heat2d_laplace.yaml
  overrides: ['model.feature_map.n_sites_per_coord.x=4', 'model.feature_map.n_sites_per_coord.y=4', 'model.tensor_network.architecture=coordinate_branch_mps', 'model.tensor_network.bond_dim=2']
[4/5] heat2d_laplace_smoke_0004
  base_config: configs/heat2d_laplace.yaml
  overrides: ['model.feature_map.n_sites_per_coord.x=4', 'model.feature_map.n_sites_per_coord.y=4', 'model.tensor_network.architecture=coordinate_branch_mps', 'model.tensor_network.bond_dim=4']
[5/5] lorenz_smoke_0001
  base_config: configs/lorenz.yaml
  overrides: ['model.feature_map.n_sites_per_coord.t=8', 'model.tensor_network.architecture=global_mps', 'model.tensor_network.bond_dim=4']
```

Nota importante: las claves `fixed:` de `configs/benchmark_smoke.yaml` no aparecen en los overrides. Por tanto, el dry-run lista 5 jobs, pero si se ejecuta el benchmark real no aplicaria los `training.steps=10`, `n_collocation=64`, etc. definidos en `fixed:`.

## 4. Numero esperado y real de runs del benchmark smoke

Segun `configs/benchmark_smoke.yaml`:

- `heat2d_laplace_smoke`: 1 valor de `x` * 1 valor de `y` * 2 arquitecturas * 2 bond dims = 4 jobs.
- `lorenz_smoke`: 1 valor de sites * 1 arquitectura * 1 bond dim = 1 job.

Numero esperado por la rejilla: `5`.

Numero real expandido por `load_benchmark_jobs('configs/benchmark_smoke.yaml')`: `5`.

Nombres reales:

```text
heat2d_laplace_smoke_0001
heat2d_laplace_smoke_0002
heat2d_laplace_smoke_0003
heat2d_laplace_smoke_0004
lorenz_smoke_0001
```

Advertencia: aunque el conteo es correcto, `fixed:` no se aplica.

## 5. Revision de un run concreto

Run revisado:

```text
runs\heat2d_laplace\20260525_205743_b5b534d7
```

### `config.yaml`

No existe:

```text
Test-Path ...\config.yaml -> False
```

Existe `resolved_config.yaml`:

```text
Test-Path ...\resolved_config.yaml -> True
```

Contenido relevante de `resolved_config.yaml`:

```yaml
problem:
  name: heat2d_laplace
model:
  family: tensor_network
  feature_map:
    kind: fourier_sites
    site_dim: 2
    n_sites_per_coord:
      x: 8
      y: 8
  tensor_network:
    backend: tensorkrowch_nodes
    architecture: coordinate_branch_mps
    bond_dim: 8
training:
  steps: 10
  n_collocation: 64
  n_boundary: 64
eval:
  grid_size: 128
output:
  root_dir: runs
```

### `metrics.csv`

Columnas disponibles:

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

Numero de filas: `10`.

Steps realmente ejecutados:

```text
1, 2, 3, 4, 5, 6, 7, 8, 9, 10
```

Extracto de metricas:

```text
 step  loss_total  loss_residual  loss_boundary  residual_rmse  boundary_rmse  relative_l2  max_abs_error
    1    1.291291   1.098947e-20       0.129129   1.048307e-10       0.359345          1.0       0.999689
    2    1.646770   4.018226e-21       0.164677   6.338948e-11       0.405804          1.0       0.999689
    3    1.144294   8.989376e-21       0.114429   9.481232e-11       0.338274          1.0       0.999689
    4    1.517408   6.929636e-21       0.151741   8.324444e-11       0.389539          1.0       0.999689
    5    1.212857   1.227691e-20       0.121286   1.108012e-10       0.348261          1.0       0.999689
    6    1.438574   5.784331e-21       0.143857   7.605479e-11       0.379285          1.0       0.999689
    7    1.104383   7.750763e-21       0.110438   8.803842e-11       0.332323          1.0       0.999689
    8    1.509287   1.766104e-20       0.150929   1.328949e-10       0.388495          1.0       0.999689
    9    0.979942   8.237014e-21       0.097994   9.075800e-11       0.313040          1.0       0.999689
   10    1.441257   1.242794e-20       0.144126   1.114807e-10       0.379639          1.0       0.999689
```

### `summary.json`

Contenido relevante:

```json
{
  "status": "completed",
  "run_id": "20260525_205743_b5b534d7",
  "problem": "heat2d_laplace",
  "architecture": "coordinate_branch_mps",
  "feature_map": "fourier_sites",
  "n_sites_total": 16,
  "site_dim": 2,
  "bond_dim": 8,
  "tn_backend": "tensorkrowch_nodes",
  "num_parameters": 2344,
  "best_step": 9,
  "final_step": 10,
  "best_loss_total": 0.9799416870494639,
  "final_loss_total": 1.4412570995210428,
  "best_residual_rmse": 9.075799789473336e-11,
  "final_residual_rmse": 1.1148065977804082e-10,
  "best_relative_l2": 0.9999999999999447,
  "final_relative_l2": 0.9999999999999447
}
```

Importante: `reports/latest/summary.csv` contradice este run y pone `best_step=1` para el mismo run, porque el agregador vuelve a seleccionar la "mejor" fila usando `relative_l2` si existe. Como `relative_l2` es constante, elige la primera fila. Esto distorsiona `best_loss_total` en el reporte agregado.

## 6. Por que los reports actuales muestran solo un punto y `loss_curves_best_runs.png` sale vacio

`reports/latest/summary.csv` contiene 2 runs, pero ambos son repeticiones del mismo experimento:

```text
problem: heat2d_laplace
architecture: coordinate_branch_mps
feature_map: fourier_sites
n_sites_total: 16
bond_dim: 8
```

Por eso las figuras agregadas tienen solo un punto/categoria visible: no hay diversidad real en arquitectura, bond dimension, numero de sites o problema dentro de los runs analizados.

`loss_curves_best_runs.png` sale vacio por una razon directa de codigo. En `src/tnpinn/results/analysis.py` lineas 137-140 se crea una figura, se hace `tight_layout()`, se guarda y se cierra, pero no se dibuja ninguna curva:

```python
plt.figure(figsize=(6, 4))
plt.tight_layout()
plt.savefig(figures_dir / "loss_curves_best_runs.png", dpi=150)
plt.close()
```

El fichero existe, pero es una imagen en blanco generada por placeholder funcional.

## 7. Verificacion de `relative_l2`, `residual_rmse`, `boundary_rmse` y `best_loss`

### `relative_l2`

No es un default fijo para `heat2d_laplace`. Se calcula con predicciones reales del modelo contra la solucion de referencia:

- `trainer.py:73-84`: crea grid de evaluacion, llama `problem.reference_solution(coords)`, calcula `prediction = model(coords)`, `error = prediction - reference`, y `rel = error.norm() / reference.norm()`.
- Si el problema no tiene referencia, devuelve `NaN`.

En el run revisado vale aproximadamente `1.0` en todos los steps. Eso no significa que sea un default; significa que el modelo entrenado 10 pasos sigue prediciendo algo muy cercano a cero frente a una referencia no nula.

### `residual_rmse`

Se calcula como:

```python
residual_rmse = sqrt(loss_residual)
```

`loss_residual` viene de `compute_losses(...)`, que llama `problem.residual(model, collocation)`. Por tanto, usa predicciones reales y autograd. En este run es extremadamente pequeno porque una salida casi constante/casi cero tiene Laplaciano casi cero para Laplace, aunque no cumpla bien frontera.

### `boundary_rmse`

Se calcula como:

```python
boundary_rmse = sqrt(loss_boundary)
```

`loss_boundary` viene de `problem.boundary_loss(model, boundary_batch)`, que compara `model(coords)` contra los targets de frontera. Es real.

### `best_loss`

En `summary.json`, `best_loss_total` se calcula como el minimo de `loss_total` observado durante el entrenamiento (`trainer.py:127` y `trainer.py:227-230`). En el run revisado, el mejor step real es el 9.

En `reports/latest/summary.csv`, el agregador recalcula best usando `relative_l2` cuando existe (`results/aggregate.py:42-49`). Como `relative_l2` es constante, el reporte agregado elige el step 1, no el step de menor loss. Esto es un bug de analisis.

## 8. Verificacion del backend TensorKrowch

Evidencia de uso real de TensorKrowch de bajo nivel:

- `src/tnpinn/tn/tensorkrowch_backend.py:19`: crea `tk.TensorNetwork()`.
- `src/tnpinn/tn/tensorkrowch_backend.py:28-34`: crea `tk.ParamNode(...)`.
- `src/tnpinn/tn/tensorkrowch_backend.py:50`: crea `tk.Node(...)` para datos de entrada.
- `src/tnpinn/tn/tensorkrowch_backend.py:51`: conecta aristas con `data["phys"] ^ core["phys"]`.
- `src/tnpinn/tn/tensorkrowch_backend.py:52`: contrae con `tk.contract_between(data, core)`.
- Busqueda de `tk.models`, `MPSLayer`, `MPS(` y `PEPS` en `src/tnpinn/tn` y `src/tnpinn/models.py`: no aparecen usos de capas prehechas.

Pero el backend no cumple completamente la intencion fuerte de "toda la TN con nodos/aristas/contracciones TensorKrowch":

- `src/tnpinn/tn/tensorkrowch_backend.py:69`: los bonds MPS se contraen con `torch.einsum`.
- `src/tnpinn/tn/tensorkrowch_backend.py:75`: los merges se contraen con `torch.einsum`.
- `src/tnpinn/tn/tensorkrowch_backend.py:80`: el readout usa `hidden @ readout_tensor`.
- `src/tnpinn/tn/tensorkrowch_backend.py:57-61`: hay un metodo `_contract_projected_mps` no implementado.

Conclusion: no es un placeholder puro, porque si usa `ParamNode`, `Node`, aristas `^` y `tk.contract_between` para proyectar cada site. Pero la contraccion principal de la red despues de cada proyeccion esta hecha con PyTorch (`einsum`/matmul). Es una implementacion hibrida, no una contraccion TensorKrowch completa de la arquitectura.

## 9. Bugs encontrados por prioridad

### P1 - `fixed:` en benchmark/sweep se ignora

`configs/benchmark_smoke.yaml` define `fixed:` para reducir steps y tamanos, pero `load_benchmark_jobs` solo expande `grid` (`experiments.py:68-75`). Si se ejecuta el smoke real, no aplicaria `training.steps=10`, `n_collocation=64`, etc. Esto puede lanzar entrenamientos mucho mas largos de lo esperado.

### P1 - Backend TensorKrowch solo es parcialmente TensorKrowch

El backend principal usa nodos/aristas para proyectar sites, pero los bonds, merges y readout se hacen con `torch.einsum`/matmul. Esto incumple parcialmente la restriccion principal del briefing.

### P1 - Agregado de reportes recalcula mal el mejor step/loss

`summary.json` del run revisado dice `best_step=9`, pero `reports/latest/summary.csv` dice `best_step=1` para el mismo run. La causa es que `aggregate.py` ordena por `relative_l2` si existe, aunque sea constante, y luego sobrescribe los valores de best loss/residual.

### P2 - `tnpinn --help` falla si la `.venv` no esta activada

El comando literal `tnpinn --help` no esta disponible en la shell actual. Funciona como `.\.venv\Scripts\tnpinn.exe --help`. Para el usuario esto puede parecer una instalacion rota si no activa la `.venv`.

### P2 - Falta `config.yaml` en runs

El run guarda `resolved_config.yaml`, pero la auditoria pidio revisar `config.yaml` y algunos specs pueden esperar ese nombre. Falta alias o documentacion clara.

### P2 - `loss_curves_best_runs.png` es una figura vacia

La funcion de analisis crea y guarda la figura sin plotear datos. Es un placeholder visible en el reporte.

### P2 - `eval_every`, `save_figures_every` y `resample_every` no se respetan

El bucle de entrenamiento remuestrea batches y calcula metricas de evaluacion en cada step. En runs grandes esto puede ser costoso y no refleja la config.

### P2 - Visualizaciones incompletas

Solo hay `viz/heat.py` y `viz/training.py`. Faltan visualizaciones especificas de Helmholtz, Lorenz y comparativas completas.

### P2 - Soporte de datos supervisados incompleto

`loss_data` se registra, pero siempre vale cero. No hay sampler ni batch de datos.

### P3 - `pass` al guardar Parquet oculta errores

`results/analysis.py` ignora cualquier excepcion al escribir Parquet. Puede ser aceptable porque Parquet era opcional, pero no deja diagnostico.

### P3 - Tests pasan, pero son todavia smoke tests

Hay cobertura para configs, feature maps, derivadas, problemas, modelos y CLI smoke. Sin embargo, no validan benchmarks reales, `fixed:`, cumplimiento completo de TensorKrowch, ni reportes con multiples arquitecturas.

