from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    metrics: dict[str, Any],
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "metrics": metrics,
    }
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path, model: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model_state"])
    return payload
