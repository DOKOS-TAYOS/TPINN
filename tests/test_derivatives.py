import math

import torch

from tnpinn.pinn.derivatives import gradient, laplacian, second_partial


def test_first_and_second_derivative_of_simple_function() -> None:
    coords = torch.tensor([[0.2, 0.4], [0.7, 0.3]], dtype=torch.float64, requires_grad=True)
    values = (coords[:, :1] ** 3) + (coords[:, 1:2] ** 2)

    grad = gradient(values, coords)
    d2x = second_partial(values, coords, 0)

    assert torch.allclose(grad[:, 0], 3.0 * coords[:, 0] ** 2)
    assert torch.allclose(grad[:, 1], 2.0 * coords[:, 1])
    assert torch.allclose(d2x.squeeze(-1), 6.0 * coords[:, 0])


def test_laplacian_of_sine_product() -> None:
    coords = torch.tensor([[0.25, 0.5], [0.4, 0.7]], dtype=torch.float64, requires_grad=True)
    values = torch.sin(math.pi * coords[:, :1]) * torch.sin(math.pi * coords[:, 1:2])

    got = laplacian(values, coords, dims=(0, 1))
    expected = -2.0 * math.pi**2 * values

    assert torch.allclose(got, expected, atol=1.0e-8, rtol=1.0e-8)
