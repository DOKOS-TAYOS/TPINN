# 00. Project brief

## Objetivo

Crear un repositorio Python para estudiar **PINNs basadas en tensor networks**. El repositorio debe permitir hacer pruebas rápidas, comparar arquitecturas tensoriales y obtener métricas físicas y tensoriales.

La pregunta de investigación práctica es:

> ¿Puede una tensor network jerárquica actuar como aproximador funcional útil dentro de una PINN para problemas sencillos de PDEs/ODEs?

## Principio básico

En una PINN clásica:

$$
u_\theta(z) = \operatorname{MLP}_\theta(z)
$$

En este repositorio:

$$
u_\theta(z) =
\operatorname{TN}_\theta
\left(
\phi_1(z),
\dots,
\phi_S(z)
\right)
$$

donde:

- `z` son coordenadas físicas;
- `phi_s` son sites de una feature map diferenciable;
- `S` es el número de sites;
- `TN_theta` es una tensor network entrenable;
- `u_theta` es el campo físico predicho.

## Problemas objetivo

1. Laplace/calor estacionario 2D.
2. Calor transitorio 2D.
3. Helmholtz 2D tipo antena.
4. Lorenz.

## Experimentos objetivo

El usuario debe poder variar por CLI o config:

- número de sites;
- dimensión local de cada site;
- tipo de feature map;
- arquitectura tensorial;
- bond dimension;
- pesos de pérdidas PINN;
- número de puntos de colocación;
- optimizador y scheduler;
- baseline MLP/SIREN/TN.

## Comando final esperado

Debe existir un comando que lance todos los benchmarks:

```bash
tnpinn benchmark --suite configs/benchmark_all.yaml
```

Y otro que analice resultados:

```bash
tnpinn analyze --runs runs --out reports/latest
```

## Filosofía de implementación

Primero correcto y reproducible. Después rápido.

Las contracciones tensoriales deben estar suficientemente encapsuladas para que, si más adelante se necesita optimizar el rendimiento, pueda reemplazarse solo el backend tensorial.
