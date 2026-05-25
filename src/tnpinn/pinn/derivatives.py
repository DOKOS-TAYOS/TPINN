from __future__ import annotations

import torch


def ensure_requires_grad(coords: torch.Tensor) -> torch.Tensor:
    if coords.requires_grad:
        return coords
    return coords.detach().clone().requires_grad_(True)


def gradient(values: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    if not values.requires_grad:
        return torch.zeros_like(coords)
    grad = torch.autograd.grad(
        values.sum(),
        coords,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        return torch.zeros_like(coords)
    return grad


def partial_derivative(values: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
    return gradient(values, coords)[:, dim : dim + 1]


def second_partial(values: torch.Tensor, coords: torch.Tensor, dim: int) -> torch.Tensor:
    first = partial_derivative(values, coords, dim)
    return partial_derivative(first, coords, dim)


def laplacian(values: torch.Tensor, coords: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    result = torch.zeros_like(values)
    for dim in dims:
        result = result + second_partial(values, coords, dim)
    return result
