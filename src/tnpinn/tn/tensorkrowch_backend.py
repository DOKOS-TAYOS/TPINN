from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import tensorkrowch as tk
import torch


class TensorKrowchBackendBase(torch.nn.Module):
    def __init__(
        self, site_dim: int, output_dim: int, bond_dim: int, init_std: float = 0.05
    ) -> None:
        super().__init__()
        self.site_dim = site_dim
        self.output_dim = output_dim
        self.bond_dim = bond_dim
        self.init_std = init_std
        self.network = tk.TensorNetwork()
        self.param_nodes: list[tk.ParamNode] = []

    def _new_param_node(
        self,
        shape: Sequence[int],
        axes_names: Sequence[str],
        name: str,
    ) -> tk.ParamNode:
        node = tk.ParamNode(
            shape=tuple(shape),
            axes_names=tuple(axes_names),
            name=name,
            network=self.network,
            init_method="randn",
        )
        assert node.tensor is not None
        with torch.no_grad():
            node.tensor.mul_(self.init_std)
        self.param_nodes.append(node)
        return node

    @staticmethod
    def _tensor_with_axes(node: tk.Node, axes: Sequence[str]) -> torch.Tensor:
        perm = [node.axes_names.index(axis) for axis in axes]
        tensor = node.tensor
        assert tensor is not None
        return tensor.permute(*perm)

    def _project_site(self, features: torch.Tensor, core_node: tk.ParamNode) -> torch.Tensor:
        core = core_node.copy(share_tensor=True)
        data = tk.Node(tensor=features, axes_names=("batch", "phys"), data=True)
        _ = data["phys"] ^ core["phys"]
        out = tk.contract_between(data, core)
        if "left" in out.axes_names:
            return self._tensor_with_axes(out, ("batch", "left", "right"))
        return self._tensor_with_axes(out, ("batch", "right"))

    @staticmethod
    def _contract_projected_mps(
        features: torch.Tensor, core_nodes: list[tk.ParamNode], projector: object
    ) -> torch.Tensor:
        raise NotImplementedError

    def _run_mps(self, features: torch.Tensor, core_nodes: list[tk.ParamNode]) -> torch.Tensor:
        if features.shape[1] != len(core_nodes):
            raise ValueError(f"Expected {len(core_nodes)} sites, got {features.shape[1]}")
        hidden = self._project_site(features[:, 0, :], core_nodes[0])
        for site_index, node in enumerate(core_nodes[1:], start=1):
            projected = self._project_site(features[:, site_index, :], node)
            hidden = torch.einsum("bl,blr->br", hidden, projected)
        return hidden

    @staticmethod
    def _merge(left: torch.Tensor, right: torch.Tensor, merge_node: tk.ParamNode) -> torch.Tensor:
        merge_tensor = cast(torch.Tensor, merge_node.tensor)
        return torch.einsum("bi,ijk,bj->bk", left, merge_tensor, right)

    @staticmethod
    def _readout(hidden: torch.Tensor, readout_node: tk.ParamNode) -> torch.Tensor:
        readout_tensor = cast(torch.Tensor, readout_node.tensor)
        return hidden @ readout_tensor

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "bond_dim": self.bond_dim,
            "site_dim": self.site_dim,
            "tn_backend": "tensorkrowch_nodes",
        }


class GlobalMPSBackend(TensorKrowchBackendBase):
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
        self.cores: list[tk.ParamNode] = [
            self._new_param_node((site_dim, bond_dim), ("phys", "right"), "mps_core_0")
        ]
        for site in range(1, n_sites):
            self.cores.append(
                self._new_param_node(
                    (bond_dim, site_dim, bond_dim),
                    ("left", "phys", "right"),
                    f"mps_core_{site}",
                )
            )
        self.readout = self._new_param_node((bond_dim, output_dim), ("hidden", "out"), "readout")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self._run_mps(features, self.cores)
        return self._readout(hidden, self.readout)

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update({"architecture": "global_mps", "n_sites_total": self.n_sites})
        return values


class CoordinateBranchMPSBackend(TensorKrowchBackendBase):
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
        self.branch_cores: dict[str, list[tk.ParamNode]] = {}
        for coord, site_slice in site_slices.items():
            n_sites = site_slice.stop - site_slice.start
            cores = [
                self._new_param_node((site_dim, bond_dim), ("phys", "right"), f"{coord}_core_0")
            ]
            for site in range(1, n_sites):
                cores.append(
                    self._new_param_node(
                        (bond_dim, site_dim, bond_dim),
                        ("left", "phys", "right"),
                        f"{coord}_core_{site}",
                    )
                )
            self.branch_cores[coord] = cores
        self.merge_nodes = [
            self._new_param_node(
                (bond_dim, bond_dim, bond_dim), ("left", "right_in", "out"), f"merge_{idx}"
            )
            for idx in range(max(len(site_slices) - 1, 0))
        ]
        self.readout = self._new_param_node((bond_dim, output_dim), ("hidden", "out"), "readout")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        branches = []
        for coord, site_slice in self.site_slices.items():
            branches.append(self._run_mps(features[:, site_slice, :], self.branch_cores[coord]))
        hidden = branches[0]
        for merge_node, branch in zip(self.merge_nodes, branches[1:], strict=False):
            hidden = self._merge(hidden, branch, merge_node)
        return self._readout(hidden, self.readout)

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


class BinaryTTNBackend(TensorKrowchBackendBase):
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
        self.leaves = [
            self._new_param_node((site_dim, bond_dim), ("phys", "right"), f"leaf_{site}")
            for site in range(n_sites)
        ]
        self.merge_nodes = [
            self._new_param_node(
                (bond_dim, bond_dim, bond_dim), ("left", "right_in", "out"), f"merge_{idx}"
            )
            for idx in range(max(n_sites - 1, 0))
        ]
        self.readout = self._new_param_node((bond_dim, output_dim), ("hidden", "out"), "readout")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        nodes = [
            self._project_site(features[:, site, :], leaf) for site, leaf in enumerate(self.leaves)
        ]
        merge_index = 0
        while len(nodes) > 1:
            next_level: list[torch.Tensor] = []
            for idx in range(0, len(nodes), 2):
                if idx + 1 >= len(nodes):
                    next_level.append(nodes[idx])
                    continue
                next_level.append(
                    self._merge(nodes[idx], nodes[idx + 1], self.merge_nodes[merge_index])
                )
                merge_index += 1
            nodes = next_level
        return self._readout(nodes[0], self.readout)

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update({"architecture": "binary_ttn", "n_sites_total": self.n_sites})
        return values


class BranchedMPSBackend(TensorKrowchBackendBase):
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
        self.local_cores: list[list[tk.ParamNode]] = []
        for group_index, (start, stop) in enumerate(self.groups):
            cores = [
                self._new_param_node(
                    (site_dim, bond_dim), ("phys", "right"), f"local_{group_index}_0"
                )
            ]
            for local_site in range(start + 1, stop):
                cores.append(
                    self._new_param_node(
                        (bond_dim, site_dim, bond_dim),
                        ("left", "phys", "right"),
                        f"local_{group_index}_{local_site - start}",
                    )
                )
            self.local_cores.append(cores)
        self.upper_cores: list[tk.ParamNode] = []
        if len(self.groups) > 1:
            self.upper_cores.append(
                self._new_param_node((bond_dim, bond_dim), ("phys", "right"), "upper_0")
            )
            for site in range(1, len(self.groups)):
                self.upper_cores.append(
                    self._new_param_node(
                        (bond_dim, bond_dim, bond_dim),
                        ("left", "phys", "right"),
                        f"upper_{site}",
                    )
                )
        self.readout = self._new_param_node((bond_dim, output_dim), ("hidden", "out"), "readout")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        latents = []
        for (start, stop), cores in zip(self.groups, self.local_cores, strict=True):
            latents.append(self._run_mps(features[:, start:stop, :], cores))
        if len(latents) == 1:
            hidden = latents[0]
        else:
            latent_sites = torch.stack(latents, dim=1)
            hidden = self._run_mps(latent_sites, self.upper_cores)
        return self._readout(hidden, self.readout)

    def diagnostics(self) -> dict[str, float | int | str]:
        values = super().diagnostics()
        values.update({"architecture": "branched_mps", "n_sites_total": self.n_sites})
        return values
