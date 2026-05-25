# TensorKrowch Backend Status

Fecha: 2026-05-25

## Backends disponibles

### `torch_reference`

- Backend de referencia en PyTorch puro.
- No importa ni usa TensorKrowch.
- Usa `torch.einsum` y multiplicaciones matriciales para las contracciones.
- Existe para depuracion, comparaciones de forma y tests.
- El alias legacy `torch_einsum_reference` sigue aceptado por compatibilidad, pero el
  diagnostico canonico es `torch_reference`.

### `tensorkrowch_hybrid`

- Backend TensorKrowch activo por defecto.
- Usa `tk.TensorNetwork`, `tk.ParamNode`, `tk.Node`, aristas con `^` y
  `tk.contract_between` para proyectar cada site fisico.
- Despues de las proyecciones locales, la contraccion de bonds MPS, merges TTN y
  readout se hace con tensores PyTorch (`torch.einsum` y `@`).
- Por tanto, es explicitamente hibrido. Su diagnostico reporta:
  - `tn_backend: tensorkrowch_hybrid`
  - `contraction_kind: hybrid_site_nodes_torch_bonds`

### `tensorkrowch_full`

- Nombre reservado para una implementacion completa con TensorKrowch de bajo nivel.
- No esta seleccionada por defecto.
- Actualmente falla de forma explicita con `NotImplementedError`.
- No contiene una contraccion principal con `torch.einsum`.

Falta para completarlo:

- Representar los bonds MPS como aristas TensorKrowch que conserven correctamente el eje batch.
- Representar los merges de TTN/branched MPS como nodos/aristas TensorKrowch y contraerlos sin
  convertir la parte central a `torch.einsum`.
- Representar el readout final como nodo TensorKrowch y contraerlo con operaciones TensorKrowch.
- Verificar gradientes, orden de ejes y equivalencia numerica contra `torch_reference`.

## Cambios realizados

- El backend antes llamado `tensorkrowch_nodes` fue renombrado a `tensorkrowch_hybrid`.
- `tensorkrowch_nodes` ahora falla con un error claro indicando el nombre correcto.
- Las configs principales usan `backend: tensorkrowch_hybrid`.
- `DEFAULT_CONFIG` usa `backend: tensorkrowch_hybrid`.
- Se anadieron tests que verifican:
  - `torch_reference` es PyTorch puro.
  - `tensorkrowch_hybrid` usa TensorKrowch para proyecciones de site y PyTorch para bonds/merges.
  - `tensorkrowch_full` esta reservado, no usa `torch.einsum` en su esqueleto y no se presenta como listo.
  - La config por defecto no usa el nombre ambiguo `tensorkrowch_nodes`.

## Verificacion

Ejecutado en esta iteracion:

- `.venv\Scripts\python.exe -m compileall src tests` -> OK.
- `ruff check . --fix` -> OK.
- `ruff format .` -> OK, sin cambios pendientes.
- `.venv\Scripts\python.exe -m pytest -q` -> 38 passed.
- `pyright` -> 0 errors, 0 warnings.
