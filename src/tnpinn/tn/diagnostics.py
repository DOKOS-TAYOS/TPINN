from __future__ import annotations

import torch


def parameter_count(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def tensor_diagnostics(model: torch.nn.Module) -> dict[str, float]:
    tensors = [param.detach() for param in model.parameters() if param.ndim >= 2]
    if not tensors:
        return {"core_norm_mean": 0.0, "core_norm_max": 0.0, "effective_rank_mean": 0.0}
    norms = torch.stack([tensor.norm() for tensor in tensors])
    ranks = []
    for tensor in tensors:
        matrix = tensor.reshape(tensor.shape[0], -1)
        singular_values = torch.linalg.svdvals(matrix.float())
        if singular_values.numel() == 0 or singular_values.sum() <= 0:
            ranks.append(torch.tensor(0.0))
            continue
        probs = singular_values / singular_values.sum()
        entropy = -(probs * torch.log(probs + 1.0e-12)).sum()
        ranks.append(torch.exp(entropy))
    rank_values = torch.stack(ranks)
    return {
        "core_norm_mean": float(norms.mean().cpu()),
        "core_norm_max": float(norms.max().cpu()),
        "effective_rank_mean": float(rank_values.mean().cpu()),
    }
