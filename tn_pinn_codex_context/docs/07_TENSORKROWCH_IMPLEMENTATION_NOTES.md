# 07. TensorKrowch implementation notes

## Requisito principal

El backend principal debe usar TensorKrowch con nodos y aristas de bajo nivel. No se deben usar capas prehechas como implementación principal.

## Patrón recomendado

Cada arquitectura tensorial debe estar encapsulada en un `torch.nn.Module`.

```python
class TensorNetworkPINN(torch.nn.Module):
    def __init__(self, feature_map, tn_backend, output_transform=None):
        super().__init__()
        self.feature_map = feature_map
        self.tn_backend = tn_backend
        self.output_transform = output_transform

    def forward(self, coords):
        features = self.feature_map(coords)
        u_raw = self.tn_backend(features)
        if self.output_transform is not None:
            return self.output_transform(coords, u_raw)
        return u_raw
```

El `tn_backend` debe ser una clase que construya TensorKrowch nodes.

## Validación de API

Codex debe inspeccionar la API local de TensorKrowch antes de implementar detalles finos:

```bash
python - <<'PY'
import tensorkrowch as tk
print("TensorKrowch:", getattr(tk, "__version__", "unknown"))
for name in ["TensorNetwork", "Node", "ParamNode", "randn", "zeros", "ones"]:
    print(name, hasattr(tk, name))
print([x for x in dir(tk) if "Node" in x or "Network" in x or "contract" in x])
PY
```

## Nodos de parámetros

Crear cores entrenables como nodos parametrizables.

Ejemplo conceptual:

```python
net = tk.TensorNetwork()
A = tk.randn(
    shape=(bond_left, site_dim, bond_right),
    axes_names=("left", "phys", "right"),
    network=net,
    param_node=True,
)
```

Si la API soporta `tk.ParamNode`, también es aceptable.

## Nodos de entrada

Los sites del batch deben entrar como nodos no parametrizables. La shape conceptual de un site es:

```text
[B, d]
```

Debe existir un eje batch compartido y un eje físico que contrae con el core correspondiente.

Ejemplo conceptual:

```text
feature_node axes: ("batch", "phys")
core_node axes: ("left", "phys", "right")
```

La contracción sobre `phys` debe preservar `batch`.

## MPS branch

Una rama MPS recibe:

```text
features_coord: [B, S_coord, d]
```

y devuelve:

```text
h_coord: [B, chi]
```

La contracción conceptual:

$$
h^r
=
\sum_{i_1,\dots,i_S}
\sum_{\rho_1,\dots,\rho_{S-1}}
A_1^{i_1,\rho_1}
A_2^{\rho_1,i_2,\rho_2}
\cdots
A_S^{\rho_{S-1},i_S,r}
\prod_{s=1}^{S}
\phi_s^{i_s}
$$

## Merge binario

Tensor entrenable:

```text
M: [chi_left, chi_right, chi_out]
```

Operación:

$$
h_{\mathrm{out}}^c
=
\sum_{a,b}
M^{a,b,c}
h_{\mathrm{left}}^a
h_{\mathrm{right}}^b
$$

En pruebas de referencia puede validarse con:

```python
torch.einsum("ba,abc,bb->bc", ...)
```

Cuidado: no usar dos veces la letra `b` en einsum. Forma correcta:

```python
torch.einsum("bi,ijk,bj->bk", h_left, M, h_right)
```

## Readout

Tensor entrenable:

```text
R: [chi_root, output_dim]
```

Operación:

$$
u^\alpha
=
\sum_r R^{r,\alpha} h^r
$$

## Contracciones y autograd

Verificar que:

```python
coords.requires_grad_(True)
u = model(coords)
du_dx = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
```

funciona.

También verificar segunda derivada.

## Diagnósticos tensoriales

Implementar:

```python
def core_norms(model) -> dict: ...
def parameter_count(model) -> int: ...
def effective_ranks(model) -> dict: ...
```

Para rangos efectivos, se pueden matricizar cores o merges y calcular SVD con `torch.linalg.svdvals`.

## Fallback de referencia

Implementar un backend `torch_einsum_reference` con los mismos parámetros lógicos, aunque no use TensorKrowch. Sirve para:

- comparar shapes;
- depurar;
- testear fórmulas;
- aislar errores de API de TensorKrowch.

Pero en configs principales usar:

```yaml
model:
  tensor_network:
    backend: tensorkrowch_nodes
```
