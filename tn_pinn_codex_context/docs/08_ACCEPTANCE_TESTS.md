# 08. Acceptance tests

## Tests de instalación

```bash
python -m pip install -e ".[dev]"
python -c "import tnpinn; print(tnpinn.__version__)"
```

## Tests unitarios

```bash
pytest -q
```

Debe cubrir:

### Config

- carga de YAML;
- resolución de defaults;
- overrides tipo dotlist;
- error claro cuando falta una clave importante.

### Feature maps

- `fourier_sites` produce `[B, S, d]`;
- coordenadas diferentes producen features diferentes;
- el output conserva dtype y device;
- las features son diferenciables respecto a coords.

### Derivadas

- primera derivada de función simple;
- segunda derivada de función simple;
- laplaciano de:

$$
u(x,y)=\sin(\pi x)\sin(\pi y)
$$

debe aproximar:

$$
-2\pi^2\sin(\pi x)\sin(\pi y)
$$

### Modelos TN

Para cada arquitectura:

- forward con batch pequeño;
- output shape `[B, q]`;
- parámetros reciben gradiente;
- coords reciben gradiente;
- segunda derivada funciona.

### Problemas

- `heat2d_laplace.reference_solution` satisface aproximadamente el residual;
- sampling genera puntos dentro del dominio;
- boundary sampling genera puntos en frontera;
- `lorenz.residual` devuelve `[B, 3]`.

### CLI

Debe funcionar:

```bash
tnpinn train --config configs/heat2d_laplace.yaml --set training.steps=2 --set training.n_collocation=16 --set training.n_boundary=16
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml --dry-run --limit 3
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 3
tnpinn analyze --runs runs --out reports/test
```

## Criterios de salida

Un run smoke debe crear:

```text
resolved_config.yaml
metrics.csv
summary.json
checkpoint_last.pt
figures/
```

`metrics.csv` debe tener al menos tantas filas como evaluaciones o logs realizados.

`summary.json` debe contener:

```json
{
  "status": "completed",
  "problem": "...",
  "architecture": "...",
  "best_loss_total": 0.0
}
```

## Criterio de no regresión

Antes de considerar finalizado cualquier cambio, Codex debe ejecutar:

```bash
pytest -q
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 5
```
