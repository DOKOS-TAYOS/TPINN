from __future__ import annotations

from typing import Any

import torch

from tnpinn.problems.base import PhysicsProblem


def regularization_loss(model: torch.nn.Module) -> torch.Tensor:
    params = [param.square().mean() for param in model.parameters() if param.requires_grad]
    if not params:
        return torch.zeros(())
    return torch.stack(params).mean()


def compute_losses(
    model: torch.nn.Module,
    problem: PhysicsProblem,
    batches: dict[str, dict[str, torch.Tensor]],
    weights: dict[str, Any],
) -> dict[str, torch.Tensor]:
    collocation = batches["collocation"]
    boundary_weight = float(weights.get("boundary", 1.0))
    initial_weight = float(weights.get("initial", 1.0))
    data_weight = float(weights.get("data", 0.0))
    regularization_weight = float(weights.get("regularization", 0.0))
    residual_values = problem.residual(model, collocation)
    loss_residual = residual_values.square().mean()
    loss_boundary = (
        problem.boundary_loss(model, batches["boundary"])
        if boundary_weight != 0.0
        else torch.zeros_like(loss_residual)
    )
    loss_initial = (
        problem.initial_loss(model, batches["initial"])
        if initial_weight != 0.0
        else torch.zeros_like(loss_residual)
    )
    loss_data = torch.zeros_like(loss_residual)
    loss_regularization = (
        regularization_loss(model).to(loss_residual.device, loss_residual.dtype)
        if regularization_weight != 0.0
        else torch.zeros_like(loss_residual)
    )
    loss_total = (
        float(weights.get("residual", 1.0)) * loss_residual
        + boundary_weight * loss_boundary
        + initial_weight * loss_initial
        + data_weight * loss_data
        + regularization_weight * loss_regularization
    )
    return {
        "loss_total": loss_total,
        "loss_residual": loss_residual,
        "loss_boundary": loss_boundary,
        "loss_initial": loss_initial,
        "loss_data": loss_data,
        "loss_regularization": loss_regularization,
    }
