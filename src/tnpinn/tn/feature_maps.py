from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FeatureMapSpec:
    kind: str
    site_dim: int
    n_sites_per_coord: dict[str, int]


class SiteFeatureMap(torch.nn.Module):
    def __init__(self, config: dict[str, Any], coordinates: tuple[str, ...]) -> None:
        super().__init__()
        self.kind = str(config.get("kind", "fourier_sites"))
        self.site_dim = int(config.get("site_dim", 2))
        self.coordinates = coordinates
        raw_sites = config.get("n_sites_per_coord") or {}
        self.n_sites_per_coord = {coord: int(raw_sites.get(coord, 1)) for coord in self.coordinates}
        self.frequency_scale = float(config.get("frequency_scale", torch.pi))
        self.frequency_mode = str(config.get("frequency_mode", "powers_of_two"))
        self.rbf_gamma = float(config.get("rbf_gamma", 25.0))
        self.site_slices = self._make_site_slices()

    @property
    def n_sites_total(self) -> int:
        return sum(self.n_sites_per_coord.values())

    def _make_site_slices(self) -> dict[str, slice]:
        start = 0
        slices: dict[str, slice] = {}
        for coord in self.coordinates:
            stop = start + self.n_sites_per_coord[coord]
            slices[coord] = slice(start, stop)
            start = stop
        return slices

    def _site_frequency(self, site: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self.frequency_mode == "powers_of_two":
            value = self.frequency_scale * (2.0**site)
        else:
            value = self.frequency_scale * float(site + 1)
        return torch.tensor(value, dtype=dtype, device=device)

    def _fourier_site(self, x: torch.Tensor, site: int) -> torch.Tensor:
        freq = self._site_frequency(site, x.dtype, x.device)
        channels = []
        for channel in range(self.site_dim):
            harmonic = channel // 2 + 1
            angle = harmonic * freq * x
            channels.append(torch.sin(angle) if channel % 2 == 0 else torch.cos(angle))
        return torch.cat(channels, dim=1)

    def _polynomial_site(self, x: torch.Tensor, site: int) -> torch.Tensor:
        scaled = (site + 1) * x
        channels = [torch.ones_like(x)]
        channels.extend(scaled.pow(power) for power in range(1, self.site_dim))
        return torch.cat(channels[: self.site_dim], dim=1)

    def _rbf_site(self, x: torch.Tensor, site: int, n_sites: int) -> torch.Tensor:
        centers = torch.linspace(
            0.0, 1.0, steps=n_sites * self.site_dim, dtype=x.dtype, device=x.device
        )
        start = site * self.site_dim
        site_centers = centers[start : start + self.site_dim].reshape(1, -1)
        return torch.exp(-self.rbf_gamma * (x - site_centers).square())

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 2 or coords.shape[1] < len(self.coordinates):
            expected = f"[B, {len(self.coordinates)}]"
            raise ValueError(f"Expected coords with shape {expected}, got {tuple(coords.shape)}")
        sites: list[torch.Tensor] = []
        for coord_index, coord_name in enumerate(self.coordinates):
            x = coords[:, coord_index : coord_index + 1]
            n_sites = self.n_sites_per_coord[coord_name]
            for site in range(n_sites):
                if self.kind == "fourier_sites":
                    sites.append(self._fourier_site(x, site))
                elif self.kind == "polynomial_sites":
                    sites.append(self._polynomial_site(x, site))
                elif self.kind == "rbf_sites":
                    sites.append(self._rbf_site(x, site, n_sites))
                else:
                    raise ValueError(f"Unknown feature map kind: {self.kind}")
        return torch.stack(sites, dim=1)

    def split_by_coord(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return {coord: features[:, site_slice, :] for coord, site_slice in self.site_slices.items()}


def build_feature_map(config: dict[str, Any], coordinates: tuple[str, ...]) -> SiteFeatureMap:
    return SiteFeatureMap(config, coordinates)
