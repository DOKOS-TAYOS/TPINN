from __future__ import annotations

import math

import torch


def relative_l2(prediction: torch.Tensor, reference: torch.Tensor | None) -> float:
    if reference is None:
        return math.nan
    error = prediction - reference
    return float((error.norm() / (reference.norm() + 1.0e-12)).detach().cpu())
