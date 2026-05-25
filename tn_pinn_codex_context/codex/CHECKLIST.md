# Codex implementation checklist

## Bootstrap

- [ ] `pyproject.toml`
- [ ] package under `src/tnpinn`
- [ ] console script `tnpinn`
- [ ] config loader
- [ ] override parser
- [ ] dry-run CLI

## Tensor networks

- [ ] `fourier_sites`
- [ ] `polynomial_sites`
- [ ] `rbf_sites`
- [ ] `tensorkrowch_nodes` backend
- [ ] `torch_einsum_reference` backend
- [ ] `global_mps`
- [ ] `coordinate_branch_mps`
- [ ] `binary_ttn`
- [ ] `branched_mps`
- [ ] tensor diagnostics

## PINN

- [ ] derivative utilities
- [ ] losses
- [ ] sampling
- [ ] heat2d_laplace
- [ ] heat2d_transient
- [ ] helmholtz_antenna2d
- [ ] lorenz

## Training

- [ ] trainer
- [ ] evaluator
- [ ] checkpoints
- [ ] metrics.csv
- [ ] summary.json
- [ ] predictions.npz
- [ ] figures

## CLI

- [ ] train
- [ ] sweep
- [ ] benchmark
- [ ] evaluate
- [ ] analyze
- [ ] plot

## Tests

- [ ] config tests
- [ ] feature map tests
- [ ] derivative tests
- [ ] TN shape tests
- [ ] TN gradient tests
- [ ] problem residual tests
- [ ] CLI smoke tests
- [ ] training smoke tests
- [ ] analysis tests
