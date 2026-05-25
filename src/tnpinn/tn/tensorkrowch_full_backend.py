from __future__ import annotations

import torch


class TensorKrowchFullBackend(torch.nn.Module):
    """Reserved backend for complete low-level TensorKrowch contractions."""

    def __init__(self, architecture: str, *args: object, **kwargs: object) -> None:
        super().__init__()
        raise NotImplementedError(
            "tensorkrowch_full is not implemented yet. The current TensorKrowch "
            "implementation is tensorkrowch_hybrid: site projections use ParamNode/Node "
            "and TensorKrowch contractions, while MPS bonds, TTN merges, and readout still "
            "use torch tensor operations."
        )
