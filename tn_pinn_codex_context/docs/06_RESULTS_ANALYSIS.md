# 06. Results analysis

## Objetivo

El módulo de análisis debe convertir muchos runs en tablas y figuras comparativas.

## Entrada

```bash
tnpinn analyze --runs runs --out reports/latest
```

Debe escanear recursivamente:

```text
runs/**/summary.json
runs/**/metrics.csv
```

## Salidas

```text
reports/latest/
├── summary.csv
├── summary.parquet
├── best_by_problem.csv
├── best_by_architecture.csv
├── failed_runs.csv
├── figures/
│   ├── rel_l2_by_architecture.png
│   ├── residual_by_architecture.png
│   ├── rel_l2_vs_bond_dim.png
│   ├── rel_l2_vs_n_sites.png
│   ├── runtime_vs_n_params.png
│   ├── heatmap_architecture_problem.png
│   └── loss_curves_best_runs.png
└── report.md
```

Parquet puede ser opcional si no se quiere depender de `pyarrow`.

## Report markdown

`report.md` debe incluir:

- fecha;
- número total de runs;
- número de runs completados/fallidos;
- mejor run por problema;
- mejor arquitectura media por problema;
- observaciones automáticas simples;
- enlaces relativos a figuras.

## Agrupaciones

Calcular agregados por:

```text
problem
architecture
feature_map
bond_dim
n_sites_total
model_family
```

Para cada grupo:

```text
count
mean_best_relative_l2
median_best_relative_l2
std_best_relative_l2
mean_best_residual_rmse
mean_time_total_s
mean_num_parameters
```

## Best run

Seleccionar mejor run por problema usando prioridad:

1. menor `best_relative_l2` si existe;
2. menor `best_residual_rmse`;
3. menor `best_loss_total`.

## Figuras

Usar matplotlib. Una figura por archivo. No usar seaborn.

### Error vs bond dimension

Eje x:

```text
bond_dim
```

Eje y:

```text
best_relative_l2
```

Agrupar por arquitectura.

### Error vs sites

Eje x:

```text
n_sites_total
```

Eje y:

```text
best_relative_l2
```

Agrupar por arquitectura.

### Runtime vs parámetros

Eje x:

```text
num_parameters
```

Eje y:

```text
time_total_s
```

Agrupar por arquitectura.

## Datos corruptos

Si falta un archivo, no fallar el análisis entero. Marcar el run como incompleto.
