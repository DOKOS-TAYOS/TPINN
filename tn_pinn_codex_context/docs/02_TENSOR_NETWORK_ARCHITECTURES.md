# 02. Tensor network architectures

## Shapes comunes

Entrada física:

```text
coords: [B, p]
```

Feature map:

```text
features: [B, S, d]
```

o por coordenada:

```text
features_by_coord["x"]: [B, S_x, d]
features_by_coord["y"]: [B, S_y, d]
features_by_coord["t"]: [B, S_t, d]
```

Salida:

```text
u: [B, q]
```

## Arquitectura 1: global MPS

Todos los sites se ordenan en una cadena.

```mermaid
flowchart LR
    P1["phi_1"] --> A1["A1"]
    P2["phi_2"] --> A2["A2"]
    P3["phi_3"] --> A3["A3"]
    P4["phi_4"] --> A4["A4"]
    A1 -- bond --> A2
    A2 -- bond --> A3
    A3 -- bond --> A4
    A4 --> R["readout"]
    R --> U["u_theta"]
```

Modelo:

$$
u_\theta^\alpha(z)
=
\sum_{i_1,\dots,i_S}
\sum_{r_1,\dots,r_{S-1}}
A_1^{i_1,r_1}
A_2^{r_1,i_2,r_2}
\cdots
A_S^{r_{S-1},i_S,r_S}
R^{r_S,\alpha}
\prod_{s=1}^S
\phi_s^{i_s}(z)
$$

Parámetros aproximados:

$$
O(S d \chi^2 + \chi q)
$$

Uso:

- baseline tensorial simple;
- útil para comprobar que feature maps y derivadas funcionan.

## Arquitectura 2: coordinate_branch_mps

Una rama MPS por coordenada física.

```mermaid
flowchart TD
    X["x"] --> FX["sites x"]
    Y["y"] --> FY["sites y"]
    T["t"] --> FT["sites t"]

    FX --> MPSX["MPS_x"]
    FY --> MPSY["MPS_y"]
    FT --> MPST["MPS_t"]

    MPSX --> HX["h_x"]
    MPSY --> HY["h_y"]
    MPST --> HT["h_t"]

    HX --> MXY["merge xy"]
    HY --> MXY
    MXY --> HXY["h_xy"]

    HXY --> MXYT["merge xyt"]
    HT --> MXYT

    MXYT --> HROOT["h_root"]
    HROOT --> R["readout"]
    R --> U["u_theta"]
```

Rama por coordenada:

$$
h_x(x)
=
\operatorname{MPS}_x
\left(
\phi_{x,1}(x),
\dots,
\phi_{x,S_x}(x)
\right)
$$

Merge binario:

$$
h_{ab}^{c}
=
\sum_{i,j}
M^{i,j,c}
h_a^i
h_b^j
$$

Readout:

$$
u^\alpha
=
\sum_r
R^{r,\alpha}
h_{\mathrm{root}}^r
$$

Ventajas:

- interpretable;
- natural para PDEs separadas por coordenadas;
- permite controlar sites por coordenada.

## Arquitectura 3: binary_ttn

Árbol binario sobre todos los sites.

```mermaid
flowchart TD
    P1["phi_1"] --> L1["leaf 1"]
    P2["phi_2"] --> L2["leaf 2"]
    P3["phi_3"] --> L3["leaf 3"]
    P4["phi_4"] --> L4["leaf 4"]

    L1 --> M12["merge 1-2"]
    L2 --> M12
    L3 --> M34["merge 3-4"]
    L4 --> M34

    M12 --> ROOT["root merge"]
    M34 --> ROOT
    ROOT --> R["readout"]
    R --> U["u_theta"]
```

Leaf:

$$
v_s^a(z)
=
\sum_i
L_s^{i,a}
\phi_s^i(z)
$$

Merge:

$$
v_p^c(z)
=
\sum_{a,b}
M_p^{a,b,c}
v_l^a(z)
v_r^b(z)
$$

Readout:

$$
u^\alpha(z)=\sum_c R^{c,\alpha}v_{\mathrm{root}}^c(z)
$$

Ventajas:

- jerarquía explícita;
- fácil analizar rangos por cortes del árbol;
- natural para agrupaciones multiescala.

## Arquitectura 4: branched_mps

Arquitectura de MPS locales que se ramifican en MPS superiores.

```mermaid
flowchart TD
    XL["x low sites"] --> MXL["MPS x low"]
    XH["x high sites"] --> MXH["MPS x high"]
    YL["y low sites"] --> MYL["MPS y low"]
    YH["y high sites"] --> MYH["MPS y high"]

    MXL --> MXS["upper MPS x"]
    MXH --> MXS

    MYL --> MYS["upper MPS y"]
    MYH --> MYS

    MXS --> G["global MPS/merge"]
    MYS --> G

    G --> R["readout"]
    R --> U["u_theta"]
```

Uso:

- probar la hipótesis de que features por escala pueden fusionarse jerárquicamente;
- más cercana a “MPS que se ramifican en otros MPS”.

## Control de bond dimension

Cada arquitectura debe aceptar:

```yaml
model:
  tensor_network:
    bond_dim: 8
```

Opcionalmente, permitir bond dimensions por zona:

```yaml
model:
  tensor_network:
    bond_dims:
      branch: 8
      merge: 8
      root: 16
```

## Backend TensorKrowch

La implementación principal debe crear nodos parametrizables manuales.

El diseño interno recomendado:

```text
TensorNetworkModel(torch.nn.Module)
├── feature_map
├── backend
│   ├── build_graph()
│   ├── set_input_features(features)
│   ├── contract()
│   └── diagnostics()
└── forward(coords)
```

`forward(coords)` debe:

1. calcular sites;
2. crear o actualizar nodos de entrada;
3. contraer la red;
4. devolver `[B, q]`.

## Backend de referencia

Debe existir un backend alternativo `torch_einsum_reference` para tests de forma y depuración. Puede usar `torch.einsum` directamente.

No debe ser el backend por defecto.
