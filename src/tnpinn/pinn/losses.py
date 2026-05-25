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
    residual_values = problem.residual(model, collocation)
    loss_residual = residual_values.square().mean()
    loss_boundary = problem.boundary_loss(model, batches["boundary"])
    loss_initial = problem.initial_loss(model, batches["initial"])
    loss_data = torch.zeros_like(loss_residual)
    loss_regularization = regularization_loss(model).to(loss_residual.device, loss_residual.dtype)
    loss_total = (
        float(weights.get("residual", 1.0)) * loss_residual
        + float(weights.get("boundary", 1.0)) * loss_boundary
        + float(weights.get("initial", 1.0)) * loss_initial
        + float(weights.get("data", 0.0)) * loss_data
        + float(weights.get("regularization", 0.0)) * loss_regularization
    )
    return {
        "loss_total": loss_total,
        "loss_residual": loss_residual,
        "loss_boundary": loss_boundary,
        "loss_initial": loss_initial,
        "loss_data": loss_data,
        "loss_regularization": loss_regularization,
    }
