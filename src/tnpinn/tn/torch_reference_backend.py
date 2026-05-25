from __future__ import annotations

from typing import cast

import torch


class TorchReferenceBase(torch.nn.Module):
    def __init__(
        self, site_dim: int, output_dim: int, bond_dim: int, init_std: float = 0.05
    ) -> None:
        super().__init__()
        self.site_dim = site_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.init_std = init_std

    def _param(self, shape: tuple[int, ...]) -> torch.nn.Parameter:
        return torch.nn.Parameter(torch.randn(*shape) * self.init_std)

    @staticmethod
    def _run_mps(features: torch.Tensor, cores: torch.nn.ParameterList) -> torch.Tensor:
        hidden = torch.einsum("bd,dk->bk", features[:, 0, :], cores[0])
        for site, core in enumerate(cores[1:], start=1):
            projected = torch.einsum("bd,ldk->blk", features[:, site, :], core)
            hidden = torch.einsum("bl,blk->bk", hidden, projected)
        return hidden

    @staticmethod
    def _merge(left: torch.Tensor, right: torch.Tensor, merge: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ijk,bj->bk", left, merge, right)

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "bond_dim": self.bond_dim,
            "site_dim": self.site_dim,
            "tn_backend": "torch_einsum_reference",
        }


class TorchReferenceGlobalMPS(TorchReferenceBase):
    def __init__(
        self,
        n_sites: int,
        site_dim: int,
        output_dim: int,
        bond_dim: int,
        init_std: float = 0.05,
    ) -> None:
        super().__init__(site_dim, output_dim, bond_dim, init_std)
        self.n_sites = n_sites
        self.cores = torch.nn.ParameterList([self._param((site_dim, bond_dim))])
        for _ in range(1, n_sites):
            self.cores.append(self._param((bond_dim, site_dim, bond_dim)))
        self.readout = self._param((bond_dim, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self._run_mps(features, self.cores) @ self.readout

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update({"architecture": "global_mps", "n_sites_total": self.n_sites})
        return values


class TorchReferenceCoordinateBranchMPS(TorchReferenceBase):
    def __init__(
        self,
        site_slices: dict[str, slice],
        site_dim: int,
        output_dim: int,
        bond_dim: int,
        init_std: float = 0.05,
    ) -> None:
        super().__init__(site_dim, output_dim, bond_dim, init_std)
        self.site_slices = site_slices
        self.branch_cores = torch.nn.ModuleDict()
        for coord, site_slice in site_slices.items():
            n_sites = site_slice.stop - site_slice.start
            cores = torch.nn.ParameterList([self._param((site_dim, bond_dim))])
            for _ in range(1, n_sites):
                cores.append(self._param((bond_dim, site_dim, bond_dim)))
            self.branch_cores[coord] = cores
        self.merges = torch.nn.ParameterList(
            [
                self._param((bond_dim, bond_dim, bond_dim))
                for _ in range(max(len(site_slices) - 1, 0))
            ]
        )
        self.readout = self._param((bond_dim, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branches = [
            self._run_mps(
                features[:, site_slice, :],
                cast(torch.nn.ParameterList, self.branch_cores[coord]),
            )
            for coord, site_slice in self.site_slices.items()
        ]
        hidden = branches[0]
        for merge, branch in zip(self.merges, branches[1:], strict=False):
            hidden = self._merge(hidden, branch, merge)
        return hidden @ self.readout

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update(
            {
                "architecture": "coordinate_branch_mps",
                "n_sites_total": sum(
                    site_slice.stop - site_slice.start for site_slice in self.site_slices.values()
                ),
            }
        )
        return values


class TorchReferenceBinaryTTN(TorchReferenceBase):
    def __init__(
        self,
        n_sites: int,
        site_dim: int,
        output_dim: int,
        bond_dim: int,
        init_std: float = 0.05,
    ) -> None:
        super().__init__(site_dim, output_dim, bond_dim, init_std)
        self.n_sites = n_sites
        self.leaves = torch.nn.ParameterList(
            [self._param((site_dim, bond_dim)) for _ in range(n_sites)]
        )
        self.merges = torch.nn.ParameterList(
            [self._param((bond_dim, bond_dim, bond_dim)) for _ in range(max(n_sites - 1, 0))]
        )
        self.readout = self._param((bond_dim, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        nodes = [
            torch.einsum("bd,dk->bk", features[:, site, :], leaf)
            for site, leaf in enumerate(self.leaves)
        ]
        merge_index = 0
        while len(nodes) > 1:
            next_level = []
            for idx in range(0, len(nodes), 2):
                if idx + 1 >= len(nodes):
                    next_level.append(nodes[idx])
                    continue
                next_level.append(self._merge(nodes[idx], nodes[idx + 1], self.merges[merge_index]))
                merge_index += 1
            nodes = next_level
        return nodes[0] @ self.readout

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update({"architecture": "binary_ttn", "n_sites_total": self.n_sites})
        return values


class TorchReferenceBranchedMPS(TorchReferenceBase):
    def __init__(
        self,
        n_sites: int,
        site_dim: int,
        output_dim: int,
        bond_dim: int,
        branch_factor: int = 2,
        init_std: float = 0.05,
    ) -> None:
        super().__init__(site_dim, output_dim, bond_dim, init_std)
        self.n_sites = n_sites
        self.branch_factor = max(branch_factor, 1)
        self.groups = [
            (start, min(start + self.branch_factor, n_sites))
            for start in range(0, n_sites, self.branch_factor)
        ]
        self.local_cores = torch.nn.ModuleList()
        for start, stop in self.groups:
            cores = torch.nn.ParameterList([self._param((site_dim, bond_dim))])
            for _ in range(start + 1, stop):
                cores.append(self._param((bond_dim, site_dim, bond_dim)))
            self.local_cores.append(cores)
        self.upper_cores = torch.nn.ParameterList()
        if len(self.groups) > 1:
            self.upper_cores.append(self._param((bond_dim, bond_dim)))
            for _ in range(1, len(self.groups)):
                self.upper_cores.append(self._param((bond_dim, bond_dim, bond_dim)))
        self.readout = self._param((bond_dim, output_dim))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latents = [
            self._run_mps(features[:, start:stop, :], cast(torch.nn.ParameterList, cores))
            for (start, stop), cores in zip(self.groups, self.local_cores, strict=True)
        ]
        hidden = (
            latents[0]
            if len(latents) == 1
            else self._run_mps(
                torch.stack(latents, dim=1),
                cast(torch.nn.ParameterList, self.upper_cores),
            )
        )
        return hidden @ self.readout

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update({"architecture": "branched_mps", "n_sites_total": self.n_sites})
        return values
