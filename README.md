# tnpinn

`tnpinn` is a Python package for small physics-informed neural network experiments
where the main approximator is a differentiable tensor network built from
feature-map sites.

The expected workflow is:

```bash
python -m pip install -e ".[dev]"
pytest -q
tnpinn train --config configs/heat2d_laplace.yaml --set training.steps=10 --set training.n_collocation=64 --set training.n_boundary=64
tnpinn sweep --config configs/sweeps/heat2d_bond_sites_arch.yaml --dry-run --limit 5
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 10
tnpinn analyze --runs runs --out reports/latest
```

