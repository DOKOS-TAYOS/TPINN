# 03. PINN problems

## Interfaz común

Cada problema debe implementar una clase que cumpla:

```python
class PhysicsProblem:
    name: str
    input_dim: int
    output_dim: int
    coordinates: tuple[str, ...]

    def sample_collocation(self, n: int, device, dtype) -> dict: ...
    def sample_boundary(self, n: int, device, dtype) -> dict: ...
    def sample_initial(self, n: int, device, dtype) -> dict: ...
    def residual(self, model, batch: dict) -> torch.Tensor: ...
    def boundary_loss(self, model, batch: dict) -> torch.Tensor: ...
    def initial_loss(self, model, batch: dict) -> torch.Tensor: ...
    def reference_solution(self, coords: torch.Tensor) -> torch.Tensor | None: ...
    def make_eval_grid(self, n: int, device, dtype) -> dict: ...
```

## heat2d_laplace

Dominio:

$$
(x,y)\in[0,1]^2
$$

Ecuación:

$$
-\Delta u=0
$$

Condiciones:

$$
u(0,y)=\sin(\pi y)
$$

$$
u(1,y)=0
$$

$$
u(x,0)=0
$$

$$
u(x,1)=0
$$

Solución analítica:

$$
u(x,y)
=
\frac{\sinh(\pi(1-x))}{\sinh(\pi)}
\sin(\pi y)
$$

Residual:

$$
R_\theta(x,y)
=
-
\left(
u_{xx}+u_{yy}
\right)
$$

Métricas:

- `residual_rmse`;
- `boundary_rmse`;
- `relative_l2`;
- `max_abs_error`.

Visualizaciones:

- superficie 3D de `u_pred`;
- superficie 3D de `u_ref`;
- heatmap de error;
- heatmap del residual.

## heat2d_transient

Dominio:

$$
(x,y,t)\in[0,1]^2\times[0,T]
$$

Ecuación:

$$
u_t-\alpha(u_{xx}+u_{yy})=0
$$

Condiciones de frontera:

$$
u(0,y,t)=\sin(\pi y)
$$

$$
u(1,y,t)=0
$$

$$
u(x,0,t)=0
$$

$$
u(x,1,t)=0
$$

Condición inicial:

$$
u(x,y,0)=0
$$

Residual:

$$
R_\theta(x,y,t)
=
u_t-\alpha(u_{xx}+u_{yy})
$$

## helmholtz_antenna2d

Dominio:

$$
(x,y)\in[-1,1]^2
$$

Ecuación escalar compleja:

$$
\Delta u+k^2u=s(x,y)
$$

Representación real:

$$
u=u_R+i u_I
$$

Salida del modelo:

```text
u: [B, 2]
u[:, 0] = u_R
u[:, 1] = u_I
```

Residual:

$$
R_R
=
\Delta u_R+k^2u_R-s_R
$$

$$
R_I
=
\Delta u_I+k^2u_I-s_I
$$

Fuente sugerida para prueba:

$$
s_R(x,y)=\exp\left(-\frac{x^2+y^2}{2\sigma_s^2}\right)
$$

$$
s_I(x,y)=0
$$

Condición de frontera inicial sugerida para MVP:

- Dirichlet homogénea en el borde, o
- Sommerfeld simplificada si se implementa bien.

Para MVP, aceptar Dirichlet homogénea como primer caso. Añadir Sommerfeld como opción:

$$
\partial_n u - iku = 0
$$

En componentes:

$$
\partial_n u_R + k u_I=0
$$

$$
\partial_n u_I - k u_R=0
$$

El signo depende de la convención temporal; documentarlo en config.

Métricas:

- residual real;
- residual imaginario;
- error de amplitud si hay referencia;
- error de fase si hay referencia.

## lorenz

Entrada:

$$
t\in[0,T]
$$

Salida:

$$
u_\theta(t)=
(x_\theta(t),y_\theta(t),z_\theta(t))
$$

Sistema:

$$
\dot{x}=\sigma(y-x)
$$

$$
\dot{y}=x(\rho-z)-y
$$

$$
\dot{z}=xy-\beta z
$$

Residual:

$$
R_x=\dot{x}_\theta-\sigma(y_\theta-x_\theta)
$$

$$
R_y=\dot{y}_\theta-x_\theta(\rho-z_\theta)+y_\theta
$$

$$
R_z=\dot{z}_\theta-x_\theta y_\theta+\beta z_\theta
$$

Condición inicial:

$$
u_\theta(0)=u_0
$$

Métricas:

- residual_rmse;
- initial_rmse;
- short_horizon_mse;
- relative_l2 en intervalo corto;
- error final;
- métricas de nube del atractor si hay horizonte largo.

Visualizaciones:

- trayectoria 3D;
- componentes temporales;
- error temporal;
- residual temporal.
