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

## License

This project is released under the [MIT License](LICENSE). Third-party
dependencies keep their own licenses; see [NOTICE](NOTICE).

## Citing and acknowledgments

If you use this software in scholarly work, cite this repository
([`CITATION.cff`](CITATION.cff)) and, as applicable, the methods below.

**SIREN baseline.** The optional `siren` baseline is an independent
reimplementation of the SIREN architecture (periodic activations / related
initialization), not a port or copy of upstream SIREN source code:

- V. Sitzmann, J. N. P. Martel, A. W. Bergman, D. B. Lindell, and G. Wetzstein,
  “Implicit Neural Representations with Periodic Activation Functions,”
  NeurIPS 2020. https://arxiv.org/abs/2006.09661

**Tensor networks.** The default TN backend depends on TensorKrowch
(https://github.com/joserapa98/tensorkrowch). Runtime also uses PyTorch and the
usual scientific Python stack listed in `NOTICE`.

