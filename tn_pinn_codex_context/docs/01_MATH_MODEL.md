# 01. Mathematical model

## Modelo funcional

Sea:

$$
z \in \Omega \subset \mathbb R^p
$$

donde `p` es la dimensión de entrada física. Por ejemplo:

- Laplace 2D: $$z=(x,y)$$
- Calor 2D transitorio: $$z=(x,y,t)$$
- Helmholtz 2D: $$z=(x,y)$$
- Lorenz: $$z=t$$

Buscamos:

$$
u_\theta : \Omega \to \mathbb R^q
$$

donde `q` es el número de canales físicos:

- calor: $$q=1$$
- Helmholtz complejo separado: $$q=2$$
- Lorenz: $$q=3$$

## Feature map por sites

La entrada física se convierte en `S` sites:

$$
\phi_s(z) \in \mathbb R^d,
\qquad
s=1,\dots,S
$$

donde `d` es la dimensión local de site.

El producto tensorial implícito es:

$$
\Phi(z)
=
\phi_1(z)
\otimes
\phi_2(z)
\otimes
\cdots
\otimes
\phi_S(z)
$$

En componentes:

$$
\Phi_{i_1,\dots,i_S}(z)
=
\prod_{s=1}^S
\phi_s^{i_s}(z)
$$

con:

$$
i_s \in \{1,\dots,d\}
$$

## Expansión densa ideal

El modelo ideal sería:

$$
u_\theta^\alpha(z)
=
\sum_{i_1,\dots,i_S}
W_\theta^{\alpha}{}_{i_1,\dots,i_S}
\prod_{s=1}^S
\phi_s^{i_s}(z)
$$

donde:

$$
\alpha \in \{1,\dots,q\}
$$

El tensor denso tendría:

$$
q d^S
$$

coeficientes, por lo que no se almacena directamente.

## Tensor network como parametrización comprimida

En lugar de almacenar:

$$
W_\theta
$$

de forma densa, se representa con una tensor network:

$$
W_\theta
=
\operatorname{TN}
(A_1,\dots,A_n)
$$

Por tanto:

$$
u_\theta(z)
=
\operatorname{Contract}
\left[
A_1,\dots,A_n,
\phi_1(z),\dots,\phi_S(z)
\right]
$$

## PINN loss general

Una PDE u ODE se escribe como:

$$
\mathcal N[u](z)=0
$$

con condiciones de contorno:

$$
\mathcal B[u](z)=g(z)
$$

y, si aplica, condiciones iniciales:

$$
\mathcal I[u](z)=u_0(z)
$$

La loss general es:

$$
\mathcal L(\theta)
=
\lambda_r \mathcal L_r
+
\lambda_b \mathcal L_b
+
\lambda_i \mathcal L_i
+
\lambda_d \mathcal L_d
+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}
$$

con:

$$
\mathcal L_r
=
\frac{1}{N_r}
\sum_{j=1}^{N_r}
\left\|
\mathcal N[u_\theta](z_j)
\right\|^2
$$

$$
\mathcal L_b
=
\frac{1}{N_b}
\sum_{j=1}^{N_b}
\left\|
\mathcal B[u_\theta](z_j^b)-g(z_j^b)
\right\|^2
$$

$$
\mathcal L_i
=
\frac{1}{N_i}
\sum_{j=1}^{N_i}
\left\|
\mathcal I[u_\theta](z_j^i)-u_0(z_j^i)
\right\|^2
$$

## Derivadas

Las derivadas se calculan con autograd respecto a las coordenadas `z`.

Ejemplo para calor transitorio:

$$
R_\theta(x,y,t)
=
\frac{\partial u_\theta}{\partial t}
-
\alpha
\left(
\frac{\partial^2 u_\theta}{\partial x^2}
+
\frac{\partial^2 u_\theta}{\partial y^2}
\right)
$$

Ejemplo para Helmholtz:

$$
R_\theta(x,y)
=
\frac{\partial^2 u_\theta}{\partial x^2}
+
\frac{\partial^2 u_\theta}{\partial y^2}
+
k^2 u_\theta
-
s(x,y)
$$

Ejemplo para Lorenz:

$$
R_x
=
\dot{x}_\theta
-
\sigma
(y_\theta-x_\theta)
$$

$$
R_y
=
\dot{y}_\theta
-
x_\theta(\rho-z_\theta)
+
y_\theta
$$

$$
R_z
=
\dot{z}_\theta
-
x_\theta y_\theta
+
\beta z_\theta
$$

## Hard constraints opcionales

Para Laplace 2D con:

$$
u(0,y)=\sin(\pi y),
\qquad
u(1,y)=0,
\qquad
u(x,0)=u(x,1)=0
$$

puede usarse:

$$
u_\theta(x,y)
=
(1-x)\sin(\pi y)
+
x(1-x)y(1-y)
T_\theta(x,y)
$$

donde `T_theta` es la salida libre de la TN.

Esto elimina o reduce la necesidad de penalizar frontera.
