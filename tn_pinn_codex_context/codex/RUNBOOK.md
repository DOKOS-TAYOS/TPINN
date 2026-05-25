# Runbook

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Check TensorKrowch

```bash
python - <<'PY'
import tensorkrowch as tk
print("TensorKrowch:", getattr(tk, "__version__", "unknown"))
print("TensorNetwork:", hasattr(tk, "TensorNetwork"))
print("Node:", hasattr(tk, "Node"))
print("ParamNode:", hasattr(tk, "ParamNode"))
PY
```

## Tests

```bash
pytest -q
```

## Smoke train

```bash
tnpinn train \
  --config configs/heat2d_laplace.yaml \
  --set training.steps=10 \
  --set training.n_collocation=64 \
  --set training.n_boundary=64 \
  --set eval.grid_size=16 \
  --set eval.eval_every=5
```

## Dry-run benchmark

```bash
tnpinn benchmark --suite configs/benchmark_all.yaml --dry-run --limit 10
```

## Smoke benchmark

```bash
tnpinn benchmark --suite configs/benchmark_smoke.yaml
```

## Analysis

```bash
tnpinn analyze --runs runs --out reports/latest
```
