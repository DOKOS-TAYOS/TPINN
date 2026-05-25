# Prompt 02: Feature maps y modelos tensoriales

Implementa feature maps y modelos tensoriales siguiendo:

- `docs/01_MATH_MODEL.md`
- `docs/02_TENSOR_NETWORK_ARCHITECTURES.md`
- `docs/07_TENSORKROWCH_IMPLEMENTATION_NOTES.md`

Objetivo:

1. Implementar `fourier_sites`, `polynomial_sites`, `rbf_sites`.
2. Implementar backend principal `tensorkrowch_nodes`.
3. Implementar backend `torch_einsum_reference` para tests.
4. Implementar arquitecturas:
   - `global_mps`
   - `coordinate_branch_mps`
   - `binary_ttn`
   - `branched_mps`
5. Implementar factory de modelos desde config.
6. Añadir diagnósticos: número de parámetros, normas de cores y rangos efectivos aproximados.
7. Añadir tests de shapes y gradientes.

Restricciones:

- No uses `tk.models.MPSLayer` ni capas prehechas de TensorKrowch como implementación principal.
- Usa TensorKrowch con nodos/aristas/nodos parametrizables en el backend principal.
- Si necesitas validar fórmulas, usa el backend `torch_einsum_reference`, pero no lo dejes como default.

Tests requeridos:

```bash
pytest -q tests/test_feature_maps.py tests/test_tn_shapes.py tests/test_tn_gradients.py
```

Smoke manual requerido:

```bash
python - <<'PY'
import torch
from tnpinn.config.loading import load_config
from tnpinn.tn.models import build_model

cfg = load_config("configs/heat2d_laplace.yaml")
model = build_model(cfg)
coords = torch.rand(8, 2, dtype=torch.float64, requires_grad=True)
u = model(coords)
print(u.shape)
du = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
print(du.shape)
PY
```

Asegúrate de que `u.shape == (8, 1)` y `du.shape == (8, 2)`.
