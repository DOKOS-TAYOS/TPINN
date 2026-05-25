from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

METRIC_FIELDS = [
    "step",
    "loss_total",
    "loss_residual",
    "loss_boundary",
    "loss_initial",
    "loss_data",
    "loss_regularization",
    "residual_rmse",
    "boundary_rmse",
    "initial_rmse",
    "relative_l2",
    "max_abs_error",
    "grad_norm",
    "num_parameters",
    "forward_time_s",
    "backward_time_s",
    "step_time_s",
    "bond_dim",
    "n_sites_total",
    "site_dim",
    "architecture",
    "feature_map",
    "tn_backend",
    "core_norm_mean",
    "core_norm_max",
    "effective_rank_mean",
]


class MetricsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        self._writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        full_row = {field: row.get(field, "") for field in METRIC_FIELDS}
        self._writer.writerow(full_row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
