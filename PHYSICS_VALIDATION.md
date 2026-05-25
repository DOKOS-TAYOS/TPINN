# Physics Validation

Date: 2026-05-25

## Tests Added

- `tests/test_metrics.py`
  - `test_relative_l2_is_zero_for_identical_prediction_and_reference`
  - `test_relative_l2_is_one_for_zero_prediction_and_nonzero_reference`
  - `test_relative_l2_is_nan_without_reference`
- `tests/test_physics_validation.py`
  - `test_laplace_analytic_solution_residual_rmse_is_small_float64`
  - `test_hard_constraint_laplace_satisfies_all_boundaries_and_zero_boundary_loss`
  - `test_laplacian_identity_requires_grad_and_no_detach_in_residual_loss`
  - `test_helmholtz_manufactured_solution_has_small_residual`
  - `test_minimal_soft_laplace_training_improves_loss_boundary_and_relative_l2`

## Formulas Verified

Laplace analytic solution:

```text
u(x, y) = sinh(pi * (1 - x)) / sinh(pi) * sin(pi y)
R = -(u_xx + u_yy)
```

The test verifies `residual_rmse < 1e-6` in `float64`.

Relative L2:

```text
relative_l2 = ||prediction - reference||_2 / (||reference||_2 + 1e-12)
```

The tests verify:

- identical prediction/reference gives relative L2 below `1e-10`;
- zero prediction with nonzero reference gives approximately `1`;
- missing reference returns `NaN`.

Hard Laplace boundary constraint:

```text
u_theta(x, y) =
  (1 - x) sin(pi y)
  + x (1 - x) y (1 - y) T_theta(x, y)
```

The tests verify:

- `u_theta(0, y) = sin(pi y)`;
- `u_theta(1, y) = 0`;
- `u_theta(x, 0) = 0`;
- `u_theta(x, 1) = 0`;
- `loss_boundary = 0` in hard mode unless soft boundary enforcement is explicitly requested.

Derivative identity:

```text
f(x, y) = sin(pi x) sin(pi y)
laplacian(f) = -2 pi^2 f
```

The tests also verify that residual evaluation passes coordinates with
`requires_grad=True` into the model and that gradients still reach both model
parameters and input coordinates.

Helmholtz manufactured residual:

For `source_kind: manufactured`, the reference field is:

```text
u_R = sin(pi x) sin(pi y)
u_I = cos(pi x) sin(pi y)
source = (k^2 - 2 pi^2) * u
```

This makes:

```text
laplacian(u) + k^2 u - source = 0
```

The test verifies residual RMSE below `1e-6`.

Minimum Laplace training:

- Runs 250 steps on a small soft-boundary MLP Laplace configuration.
- Verifies final `loss_total` is below the first logged value.
- Verifies final `boundary_rmse` is below the first logged value.
- Verifies final `relative_l2` is below the first logged value.

## Bugs Found

1. `tnpinn.pinn.metrics.relative_l2` did not exist as a reusable metric function.
2. `boundary_mode: hard` did not activate the hard Laplace output transform.
3. Baseline models did not support the hard Laplace output transform.
4. Hard Laplace mode still used soft boundary loss unless the old
   `use_hard_constraints` path was manually combined with other settings.
5. `compute_losses` evaluated boundary and initial losses even when their weights were
   zero, which can introduce unnecessary model calls and obscures gradient checks.
6. Helmholtz had no manufactured reference/source pair for residual validation.

## Corrections Applied

- Added `tnpinn.pinn.metrics.relative_l2` and used it in trainer evaluation.
- Added `OutputTransformedModel` so hard constraints work for baselines as well as TNs.
- Added `boundary_mode: hard` support for Laplace and preserved compatibility with
  `use_hard_constraints: true`.
- Made hard Laplace boundary loss return numerical zero by default; soft boundary
  enforcement can be explicitly enabled with `soft_boundary_with_hard: true`.
- Made `compute_losses` skip boundary, initial, and regularization computations when
  their weights are zero.
- Added Helmholtz `source_kind: manufactured` with a matching reference solution.

## Results

Targeted validation command:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\test_derivatives.py tests\test_problems.py tests\test_metrics.py tests\test_physics_validation.py
```

Result:

```text
13 passed
```
