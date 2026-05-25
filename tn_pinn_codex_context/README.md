# TN-PINNs Codex Context Pack

Este paquete está pensado para inicializar un repositorio Python llamado `tn-pinns` con modelos **physics-informed neural networks** cuyo aproximador funcional principal sea una **tensor network** construida con TensorKrowch.

La intención no es que Codex improvise la arquitectura, sino que implemente un repositorio reproducible con:

- modelos tensoriales basados en sites de feature maps;
- arquitecturas MPS, MPS ramificadas y TTN jerárquicas;
- problemas PINN de prueba: Laplace/calor 2D, calor transitorio 2D, Helmholtz 2D tipo antena y Lorenz;
- baselines MLP/SIREN;
- CLI para entrenar, barrer hiperparámetros, lanzar benchmarks completos y analizar resultados;
- métricas, figuras y artefactos por ejecución.

## Cómo usar este paquete con Codex

1. Crea un repositorio vacío, por ejemplo:

```bash
mkdir tn-pinns
cd tn-pinns
git init
```

2. Copia estos archivos dentro del repositorio. Especialmente importante:

```text
AGENTS.md
docs/
prompts/
configs/
templates/
codex/
```

3. Abre Codex en la raíz del repo. El archivo `AGENTS.md` debe quedar en la raíz para que Codex lo use como instrucción de proyecto.

4. Pega primero el prompt:

```text
prompts/00_master_prompt.md
```

5. Si prefieres trabajar por etapas, pega después los prompts:

```text
prompts/01_bootstrap_repo.md
prompts/02_feature_maps_and_tn_models.md
prompts/03_pinn_problems_and_derivatives.md
prompts/04_training_cli_configs.md
prompts/05_benchmarks_and_sweeps.md
prompts/06_visualization_and_analysis.md
prompts/07_tests_refactor_hardening.md
```

## Resultado esperado

Al final, tras la implementación, deberían funcionar comandos como:

```bash
tnpinn train --config configs/heat2d_laplace.yaml
tnpinn train --config configs/heat2d_laplace.yaml --set model.feature_map.n_sites_per_coord.x=12 --set model.tensor_network.bond_dim=16
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml
tnpinn benchmark --suite configs/benchmark_all.yaml
tnpinn analyze --runs runs --out reports/latest
```

También deberían funcionar variantes sin instalar el paquete:

```bash
python -m tnpinn.cli.train --config configs/heat2d_laplace.yaml
python -m tnpinn.cli.benchmark --suite configs/benchmark_all.yaml
python -m tnpinn.cli.analyze --runs runs --out reports/latest
```

## Principio de diseño

El modelo debe seguir la composición:

$$
z
\longrightarrow
\{\phi_s(z)\}_{s=1}^{S}
\longrightarrow
\operatorname{TN}_\theta
\longrightarrow
u_\theta(z)
\longrightarrow
\mathcal N[u_\theta](z)
\longrightarrow
\mathcal L(\theta)
$$

donde `z` son coordenadas físicas, `S` es el número de sites, `TN` es una tensor network entrenable, y `N[u]` es el residual físico de la ecuación diferencial.
