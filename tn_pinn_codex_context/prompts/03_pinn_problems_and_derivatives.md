# Prompt 03: Problemas PINN y derivadas

Implementa los problemas físicos y utilidades de derivación siguiendo:

- `docs/01_MATH_MODEL.md`
- `docs/03_PINN_PROBLEMS.md`

Objetivo:

1. Implementar `pinn/derivatives.py`:
   - `grad`
   - `partial`
   - `second_partial`
   - `laplacian`
   - `jacobian`
2. Implementar `problems/base.py`.
3. Implementar:
   - `heat2d_laplace`
   - `heat2d_transient`
   - `helmholtz_antenna2d`
   - `lorenz`
4. Implementar sampling interior, frontera e inicial.
5. Implementar soluciones de referencia donde sea posible.
6. Implementar losses físicas por problema.
7. Añadir tests.

Tests requeridos:

```bash
pytest -q tests/test_derivatives.py tests/test_heat_residual.py tests/test_lorenz_residual.py tests/test_problem_sampling.py
```

Criterios:

- Las derivadas deben funcionar con `torch.float64`.
- `heat2d_laplace.reference_solution` debe tener residual Laplace cercano a cero en puntos interiores.
- `lorenz.residual(model, batch)` debe devolver shape `[B, 3]`.
- `helmholtz_antenna2d` debe devolver dos canales reales.
