# This code mainly inspired by the InfGCN by cheng
# https://github.com/ccr-cheng/infgcn-pytorch?tab=readme-ov-file

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_scatter import scatter
from torch_geometric.nn import radius_graph, radius
from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import FullyConnectedNet, Extract, Activation

from source.models.GPWNO_utils import *
from source.models.PWNO_utils import *
from source.models.interface_with_log import interface
from source.models.orbital import GaussianOrbital


def pbc_vec(vec, cell):
    """
    Apply periodic boundary condition to the vector
    :param vec: original vector of (N, K, 3)
    :param cell: cell frame of (N, 3, 3)
    :return: shortest vector of (N, K, 3)
    """
    coord = vec @ torch.linalg.inv(cell)
    coord = coord - torch.round(coord)
    pbc_vec = coord @ cell
    return pbc_vec.detach(), coord.detach()


class PWNO(nn.Module):
    def __init__(
        self,
        modes1,
        modes2,
        modes3,
        width,
        num_fourier_time,
        padding=0,
        num_layers=2,
        using_ff=False,
    ):
        super(PWNO, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 timesteps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.num_fourier_time = num_fourier_time
        self.padding = padding  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(self.num_fourier_time + 3, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        self.num_layers = num_layers
        if using_ff:
            self.conv = nn.ModuleList(
                [
                    SpectralConv3d_FFNO(
                        self.width, self.width, self.modes1, self.modes2, self.modes3
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            self.conv = nn.ModuleList(
                [
                    SpectralConv3d(
                        self.width, self.width, self.modes1, self.modes2, self.modes3
                    )
                    for _ in range(self.num_layers)
                ]
            )
        self.w = nn.ModuleList(
            [nn.Conv3d(self.width, self.width, 1) for _ in range(self.num_layers)]
        )
        self.bn = nn.ModuleList(
            [torch.nn.BatchNorm3d(self.width) for _ in range(self.num_layers)]
        )

    def forward(self, residue, fourier_grid):
        x = torch.cat([residue, fourier_grid], dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, H, X, Y, Z]
        if self.padding != 0:
            x = F.pad(
                x, [0, self.padding, 0, self.padding, 0, self.padding]
            )  # pad the domain if input is non-periodic

        for i in range(self.num_layers):
            x1 = self.conv[i](x)
            x2 = self.w[i](x)
            x = x1 + x2
            # x = self.bn[i](x)
            if i != self.num_layers - 1:
                x = F.gelu(x)

        if self.padding != 0:
            x = x[..., :, : -self.padding, : -self.padding, : -self.padding]
        x = x.permute(0, 2, 3, 4, 1)  # pad the domain if input is non-periodic
        return x


class GPWNO(interface):
    def __init__(
        self,
        n_atom_type,
        num_radial,
        num_spherical,
        radial_embed_size,
        radial_hidden_size,
        num_fourier=64,
        num_fourier_time=4,
        num_radial_layer=2,
        num_gcn_layer=3,
        cutoff=3.0,
        grid_cutoff=3.0,
        probe_cutoff=3.0,
        is_fc=True,
        gauss_start=0.5,
        gauss_end=5.0,
        activation="norm",
        residual=True,
        pbc=False,
        width=20,
        padding=6,
        use_max_cell=True,
        max_cell_size=22,
        equivariant_frame=True,
        probe_and_node=False,
        product=False,
        normalize=True,
        model_sharing=True,
        num_infgcn_layer=3,
        input_infgcn=False,
        using_ff=True,
        scalar_mask=True,
        use_detach=True,
        mask_cutoff=3.0,
        scalar_inv=False,
        num_spherical_RNO=None,
        positive_output=False,
        atomic_gauss_dist=True,
        input_dist=True,
        atom_info=None,
        fourier_mode=0,
        *args,
        **kwargs,
    ):
        """
        Implement the GPWNO model for electron density estimation
        :param n_atom_type: number of atom types
        :param num_radial: number of radial basis
        :param num_spherical: maximum number of spherical harmonics for each radial basis,
                number of spherical basis will be (num_spherical + 1)^2
        :param radial_embed_size: embedding size of the edge length
        :param radial_hidden_size: hidden size of the radial network
        :param num_radial_layer: number of hidden layers in the radial network
        :param num_gcn_layer: number of GPWNO layers
        :param cutoff: cutoff distance for building the molecular graph
        :param grid_cutoff: cutoff distance for building the grid-atom graph
        :param is_fc: whether the GPWNO layer should use fully connected tensor product
        :param gauss_start: start coefficient of the Gaussian radial basis
        :param gauss_end: end coefficient of the Gaussian radial basis
        :param activation: activation type for the GPWNO layer, can be ['scalar', 'norm']
        :param residual: whether to use the residue prediction layer
        :param pbc: whether the data satisfy the periodic boundary condition
        """
        super(GPWNO, self).__init__(*args, **kwargs)
        self.n_atom_type = n_atom_type
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.radial_embed_size = radial_embed_size
        self.radial_hidden_size = radial_hidden_size
        self.num_radial_layer = num_radial_layer
        self.num_gcn_layer = num_gcn_layer
        self.cutoff = cutoff
        self.grid_cutoff = grid_cutoff
        self.is_fc = is_fc
        self.gauss_start = gauss_start
        self.gauss_end = gauss_end
        self.activation = activation
        self.residual = residual
        self.pbc = pbc
        self.num_fourier = num_fourier
        self.num_fourier_time = num_fourier_time
        self.probe_cutoff = probe_cutoff
        self.use_max_cell = use_max_cell
        self.max_cell_size = max_cell_size
        self.equivariant_frame = equivariant_frame
        self.probe_and_node = probe_and_node
        self.normalize = normalize
        self.model_sharing = model_sharing
        self.num_infgcn_layer = num_infgcn_layer
        self.input_infgcn = input_infgcn
        self.scalar_mask = scalar_mask
        self.use_detach = use_detach
        self.mask_cutoff = mask_cutoff
        self.scalar_inv = scalar_inv
        self.positive_output = positive_output
        self.atomic_gauss_dist = atomic_gauss_dist
        self.input_dist = input_dist
        self.atom_info = atom_info
        self.fourier_mode = fourier_mode

        self.num_spherical_RNO = num_spherical_RNO
        if model_sharing == True or num_spherical_RNO is None:
            self.num_spherical_RNO = num_spherical

        self.product = product

        assert activation in ["scalar", "norm"]

        self.embedding = nn.Embedding(n_atom_type, num_radial)
        self.irreps_sh = o3.Irreps.spherical_harmonics(num_spherical, p=1)
        self.irreps_feat = (self.irreps_sh * num_radial).sort().irreps.simplify()

        self.irreps_sh_RNO = o3.Irreps.spherical_harmonics(num_spherical_RNO, p=1)
        self.irreps_feat_RNO = (
            (self.irreps_sh_RNO * num_radial).sort().irreps.simplify()
        )

        self.gcns_RNO = nn.ModuleList(
            [
                GCNLayer(
                    (f"{num_radial}x0e" if i == 0 else self.irreps_feat_RNO),
                    self.irreps_feat_RNO,
                    self.irreps_sh_RNO,
                    radial_embed_size,
                    num_radial_layer,
                    radial_hidden_size,
                    is_fc=is_fc,
                    **kwargs,
                )
                for i in range(num_gcn_layer)
            ]
        )
        if self.activation == "scalar":
            self.act = ScalarActivation(
                self.irreps_feat, torch.nn.functional.silu, torch.sigmoid
            )
            self.act_RNO = ScalarActivation(
                self.irreps_feat_RNO, torch.nn.functional.silu, torch.sigmoid
            )
        else:
            self.act = NormActivation(self.irreps_feat)
            self.act_RNO = NormActivation(self.irreps_feat_RNO)

        if self.model_sharing == False:
            self.infgcns = nn.ModuleList(
                [
                    GCNLayer(
                        (f"{num_radial}x0e" if i == 0 else self.irreps_feat),
                        self.irreps_feat,
                        self.irreps_sh,
                        radial_embed_size,
                        num_radial_layer,
                        radial_hidden_size,
                        is_fc=is_fc,
                        **kwargs,
                    )
                    for i in range(num_infgcn_layer)
                ]
            )

        self.residue = None
        if self.residual:
            self.residue = GCNLayer(
                self.irreps_feat,
                "0e",
                self.irreps_sh,
                radial_embed_size,
                num_radial_layer,
                radial_hidden_size,
                is_fc=True,
                use_sc=False,
                **kwargs,
            )

        self.probe_gcn = GCNLayer(
            self.irreps_feat_RNO,
            f"{num_fourier_time}x0e",
            self.irreps_sh_RNO,
            radial_embed_size,
            num_radial_layer,
            radial_hidden_size,
            is_fc=True,
            use_sc=False,
            **kwargs,
        )
        self.act_probe = NormActivation(f"{num_fourier_time}x0e")

        self.mode = (
            (self.num_fourier // 2) if self.fourier_mode == 0 else self.fourier_mode
        )
        self.width = width
        if self.input_infgcn:
            self.num_fourier_time += 1
        if self.input_dist:
            self.num_fourier_time += 1
        if self.atomic_gauss_dist:
            self.num_fourier_time += 10

        self.PWNO = PWNO(
            modes1=self.mode,
            modes2=self.mode,
            modes3=self.mode,
            width=self.width,
            num_fourier_time=self.num_fourier_time,
            padding=padding,
            using_ff=using_ff,
        )

        self.scalar_field_gcn = GCNLayer(
            f"{self.width}x0e",
            f"0e",
            self.irreps_sh_RNO,
            radial_embed_size,
            num_radial_layer,
            radial_hidden_size,
            is_fc=True,
            use_sc=False,
            **kwargs,
        )

        self.orbital = GaussianOrbital(
            gauss_start, gauss_end, num_radial, num_spherical
        )

        if self.atomic_gauss_dist and atom_info is not None:
            import json

            with open(atom_info) as f:
                atom_info = json.load(f)
            atom_radius = [info["radius"] for info in atom_info]
            self.atom_radius = [radius for idx, radius in enumerate(atom_radius)]
            self.atom_radius = torch.FloatTensor(self.atom_radius)

    def forward(self, atom_types, atom_coord, grid, batch, infos, mode="train"):
        """
        Network forward
        :param atom_types: atom types of (N,)
        :param atom_coord: atom coordinates of (N, 3)
        :param grid: coordinates at grid points of (G, K, 3)
        :param batch: batch index for each node of (N,)
        :param infos: list of dictionary containing additional information
        :return: predicted value at each grid point of (G, K)
        """
        # Embedding
        cell = torch.stack([info["cell"] for info in infos], dim=0).to(batch.device)
        atom_types = atom_types.to(batch.device)
        feat = self.embedding(atom_types)
        n_graph, n_sample = grid.size(0), grid.size(1)

        probe_len = infos[0]["probe"].reshape(-1, 3).size(0)
        probe = torch.cat([info["probe"] for info in infos], dim=0).to(batch.device)
        probe_flat = probe.reshape(-1, 3)
        probe_batch = torch.arange(grid.size(0), device=grid.device).repeat_interleave(
            probe_len
        )

        probe_edge = torch.cat([info["probe_edge"] for info in infos], dim=0).to(
            batch.device
        )
        probe_dst = torch.cat([info["probe_dst"] for info in infos], dim=0).to(
            batch.device
        )
        probe_src = torch.cat([info["probe_src"] for info in infos], dim=0).to(
            batch.device
        )

        probe_pad = []
        atom_pad = []
        probe_pad_num = 0
        atom_pad_num = 0
        for i in range(grid.size(0)):
            probe_src_num = infos[i]["probe_src"].size(0)
            pad = torch.zeros(probe_src_num).to(batch.device) + probe_pad_num
            probe_pad.append(pad)
            probe_pad_num += probe_len

            atom_num = batch[batch == i].size(0)
            pad = torch.zeros(probe_src_num).to(batch.device) + atom_pad_num
            atom_pad.append(pad)
            atom_pad_num += atom_num
        probe_dst = probe_dst + torch.cat(probe_pad, dim=0).long()
        probe_src = probe_src + torch.cat(atom_pad, dim=0).long()

        if self.pbc:
            super_probe_len = infos[0]["super_probe"].reshape(-1, 3).size(0)
            super_probe = torch.cat([info["super_probe"] for info in infos], dim=0).to(
                batch.device
            )
            super_probe_flat = super_probe.reshape(-1, 3)
            super_probe_batch = torch.arange(
                grid.size(0), device=grid.device
            ).repeat_interleave(super_probe_len)
            super_probe_idx = torch.cat(
                [info["super_probe_idx"] for info in infos], dim=0
            ).to(batch.device)
            super_probe_idx_batch = torch.arange(
                grid.size(0), device=grid.device
            ).repeat_interleave(
                len(super_probe_idx)
            ) * probe_len + super_probe_idx.repeat(
                grid.size(0)
            )

        # normalize the atom coordinates
        if self.use_max_cell and self.equivariant_frame:
            atom_center = scatter(atom_coord, batch, dim=0, reduce="mean")
            atom_coord = atom_coord - atom_center[batch]
            grid -= atom_center.unsqueeze(1)

        probe_log = probe.cpu()
        if self.use_max_cell and self.equivariant_frame:
            probe_log += atom_center.reshape(-1, 1, 1, 1, 3).cpu()

        # GCN input
        edge_index = radius_graph(atom_coord, self.cutoff, batch, loop=False)
        src, dst = edge_index
        edge_vec = atom_coord[src] - atom_coord[dst]
        edge_len = edge_vec.norm(dim=-1) + 1e-8
        edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical + 1)),
            edge_vec / edge_len[..., None],
            normalize=False,
            normalization="integral",
        )
        edge_embed = soft_one_hot_linspace(
            edge_len,
            start=0.0,
            end=self.cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ).mul(self.radial_embed_size**0.5)

        edge_feat_RNO = o3.spherical_harmonics(
            list(range(self.num_spherical_RNO + 1)),
            edge_vec / edge_len[..., None],
            normalize=False,
            normalization="integral",
        )

        # GCN
        if self.model_sharing == False:
            infgcn_feat = feat
            for i, gcn in enumerate(self.infgcns):
                infgcn_feat = gcn(
                    edge_index,
                    infgcn_feat,
                    edge_feat,
                    edge_embed,
                    dim_size=atom_types.size(0),
                )
                if i != self.num_infgcn_layer - 1:
                    infgcn_feat = self.act(infgcn_feat)

        for i, gcn in enumerate(self.gcns_RNO):
            feat = gcn(
                edge_index, feat, edge_feat_RNO, edge_embed, dim_size=atom_types.size(0)
            )
            if i != self.num_gcn_layer - 1:
                feat = self.act_RNO(feat)

        avg_probe_degree = (
            probe_dst.size(0) / probe_flat.size(0) if probe_flat.size(0) != 0 else 0
        )
        avg_probe_exist_degree = (
            probe_dst.size(0) / probe_dst.unique().size(0)
            if probe_dst.unique().size(0) != 0
            else 0
        )
        avg_probe_exist_rate = (
            probe_dst.unique().size(0) / probe_flat.size(0)
            if probe_dst.unique().size(0) != 0
            else 0
        )

        probe_len = torch.norm(probe_edge, dim=-1) + 1e-8
        probe_edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical_RNO + 1)),
            probe_edge / (probe_len[..., None] + 1e-8),
            normalize=False,
            normalization="integral",
        )
        probe_edge_embed = soft_one_hot_linspace(
            probe_len,
            start=0.0,
            end=self.probe_cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ).mul(self.radial_embed_size**0.5)

        # feat: (probe_src.size(0), sp_feat(4-> 400))
        # probe_feat: (b,f*f*f,c)

        probe_feat = self.probe_gcn.forward_mean(
            (probe_src, probe_dst),
            feat,
            probe_edge_feat,
            probe_edge_embed,
            dim_size=probe_flat.size(0),
        )

        probe_feat = self.act_probe(probe_feat)

        # probe_feat: (b,f,f,f,c)
        probe_feat = probe_feat.reshape(
            grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, -1
        )

        # Potential
        minimal_dist_grid = torch.zeros(grid.size(0), grid.size(1))
        for batch_idx in range(grid.size(0)):
            if self.pbc:
                minimal_dist_grid[batch_idx] = torch.min(
                    torch.min(
                        torch.norm(
                            grid[batch_idx].unsqueeze(1).unsqueeze(1)  # (G,1,1,3)
                            - atom_coord[batch == batch_idx]
                            .unsqueeze(0)
                            .unsqueeze(0)  # (1,1,N,3)
                            + torch.stack(
                                [
                                    i * cell[batch_idx][0]
                                    + j * cell[batch_idx][1]
                                    + k * cell[batch_idx][2]
                                    for i in [-1, 0, 1]
                                    for j in [-1, 0, 1]
                                    for k in [-1, 0, 1]
                                ],
                                dim=0,
                            ).reshape(
                                1, 27, 1, 3
                            ),  # (1,27,1,3)
                            dim=-1,
                        ),
                        dim=-2,
                    )[0],
                    dim=-1,
                )[0]
            else:
                minimal_dist_grid[batch_idx] = torch.min(
                    torch.norm(
                        grid[batch_idx].unsqueeze(1)  # (G,1,3)
                        - atom_coord[batch == batch_idx].unsqueeze(0),  # (1,N,3)
                        dim=-1,
                    ),
                    dim=-1,
                )[0]
        mask = minimal_dist_grid > self.mask_cutoff

        # atomic_dist_probe = torch.zeros(grid.size(0), self.num_fourier**3).to(
        #     batch.device)
        atomic_dist_probe = torch.zeros(grid.size(0), self.num_fourier**3, 10).to(
            batch.device
        )
        alpha = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]).to(
            batch.device
        )
        alpha = alpha.reshape(1, 1, 10)
        minimal_dist_probe = torch.zeros(grid.size(0), self.num_fourier**3).to(
            batch.device
        )

        for batch_idx in range(grid.size(0)):
            if self.pbc:
                pr = probe.reshape(grid.size(0), -1, 3)
                dist_min = torch.min(
                    torch.norm(
                        pr[batch_idx].unsqueeze(1).unsqueeze(1)  # (G,1,1,3)
                        - atom_coord[batch == batch_idx]
                        .unsqueeze(0)
                        .unsqueeze(0)  # (1,1,N,3)
                        + torch.stack(
                            [
                                i * cell[batch_idx][0]
                                + j * cell[batch_idx][1]
                                + k * cell[batch_idx][2]
                                for i in [-1, 0, 1]
                                for j in [-1, 0, 1]
                                for k in [-1, 0, 1]
                            ],
                            dim=0,
                        ).reshape(
                            1, 27, 1, 3
                        ),  # (1,27,1,3)
                        dim=-1,
                    ),
                    dim=-2,
                )[0]
                if self.atomic_gauss_dist and self.atom_info is not None:
                    self.atom_radius = self.atom_radius.to(batch.device)
                    atomic_dist_probe[batch_idx] = torch.sum(
                        torch.exp(
                            -alpha
                            * (
                                (
                                    dist_min
                                    / (
                                        self.atom_radius[
                                            atom_types[batch == batch_idx]
                                        ].unsqueeze(0)
                                    )
                                )
                            ).unsqueeze(-1)
                        ),
                        dim=-2,
                    )

                minimal_dist_probe[batch_idx] = torch.min(
                    dist_min,
                    dim=-1,
                )[0]

        if self.input_infgcn:
            probe_vec = probe.reshape(grid.size(0), -1, 3)[
                batch
            ] - atom_coord.unsqueeze(-2)

            if self.pbc:
                probe_vec, frac = pbc_vec(probe_vec, cell[batch])

            density_res = torch.zeros(grid.size(0), self.num_fourier**3, 1).to(
                batch.device
            )
            for batch_idx in range(grid.size(0)):
                if self.model_sharing == False:
                    density_res[batch_idx] = (
                        (
                            self.orbital(probe_vec[batch_idx])
                            * infgcn_feat[batch_idx].unsqueeze(0)
                        ).sum(dim=-1)
                    ).reshape(-1, 1)
                else:
                    density_res[batch_idx] = (
                        (
                            self.orbital(probe_vec[batch_idx])
                            * feat[batch_idx].unsqueeze(0)
                        ).sum(dim=-1)
                    ).reshape(-1, 1)
            density_res = density_res.reshape(
                grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, -1
            )
            if self.model_sharing == True:
                infgcn_feat = feat

            if self.residual and self.input_infgcn:
                res_feat = infgcn_feat
                probe_edge_feat_res = o3.spherical_harmonics(
                    list(range(self.num_spherical + 1)),
                    probe_edge / (probe_len[..., None] + 1e-8),
                    normalize=False,
                    normalization="integral",
                )

                residue = self.residue(
                    (probe_src, probe_dst),
                    res_feat,
                    probe_edge_feat_res,
                    probe_edge_embed,
                    dim_size=probe_flat.size(0),
                )
                density_res = density_res + residue.reshape(
                    grid.size(0),
                    self.num_fourier,
                    self.num_fourier,
                    self.num_fourier,
                    -1,
                )

            probe_feat = torch.cat([probe_feat, density_res], dim=-1)

        if self.input_dist:
            minimal_dist_probe = torch.exp(-minimal_dist_probe * 0.5).unsqueeze(-1)
            minimal_dist_probe = minimal_dist_probe.reshape(
                grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, 1
            )

            probe_feat = torch.cat([probe_feat, minimal_dist_probe], dim=-1)

        if self.atomic_gauss_dist:
            atomic_dist_probe = atomic_dist_probe.reshape(
                grid.size(0), self.num_fourier, self.num_fourier, self.num_fourier, 10
            )
            # atomic_dist_probe = torch.exp(-atomic_dist_probe * 0.5).unsqueeze(-1)

            probe_feat = torch.cat([probe_feat, atomic_dist_probe], dim=-1)

        fourier_grid = self.get_grid(probe_feat.shape, probe_feat.device)
        probe_feat = self.PWNO(probe_feat, fourier_grid)
        probe_feat = probe_feat.reshape(-1, probe_feat.size(-1))

        # sample node
        sample_flat = grid.reshape(-1, 3)
        sample_batch = (
            torch.arange(n_graph, device=grid.device)
            .repeat_interleave(n_sample)
            .detach()
        )

        probe_and_node = probe_flat
        probe_and_node_batch = probe_batch

        if self.pbc:
            sample_dst, super_probe_and_node_src = radius(
                super_probe_flat,
                sample_flat,
                self.grid_cutoff,
                super_probe_batch,
                sample_batch,
            )
            probe_and_node_src = super_probe_idx_batch[super_probe_and_node_src]
        else:
            sample_dst, probe_and_node_src = radius(
                probe_and_node,
                sample_flat,
                self.grid_cutoff,
                probe_and_node_batch,
                sample_batch,
            )
        # Logging the degree of the graph
        avg_sample_degree = (
            sample_dst.size(0) / sample_flat.size(0) if sample_flat.size(0) != 0 else 0
        )
        avg_sample_exist_degree = (
            sample_dst.size(0) / sample_dst.unique().size(0)
            if sample_dst.unique().size(0) != 0
            else 0
        )
        avg_sample_exist_rate = (
            sample_dst.unique().size(0) / sample_flat.size(0)
            if sample_dst.unique().size(0) != 0
            else 0
        )

        if self.pbc:
            sample_edge = (
                sample_flat[sample_dst] - super_probe_flat[super_probe_and_node_src]
            )
        else:
            sample_edge = sample_flat[sample_dst] - probe_flat[probe_and_node_src]
        sample_len = torch.norm(sample_edge, dim=-1) + 1e-8

        sample_edge_feat = o3.spherical_harmonics(
            list(range(self.num_spherical_RNO + 1)),
            sample_edge / (sample_len[..., None] + 1e-8),
            normalize=False,
            normalization="integral",
        )
        sample_edge_embed = soft_one_hot_linspace(
            sample_len,
            start=0.0,
            end=self.grid_cutoff,
            number=self.radial_embed_size,
            basis="gaussian",
            cutoff=False,
        ).mul(self.radial_embed_size**0.5)

        # if self.probe_and_node:
        #     probe_and_node_feat = torch.cat([probe_feat, feat], dim=0)
        probe_and_node_feat = probe_feat
        scalar_field_gcn = self.scalar_field_gcn.forward_mean(
            (probe_and_node_src, sample_dst),
            probe_and_node_feat,
            sample_edge_feat,
            sample_edge_embed,
            dim_size=sample_flat.size(0),
        )

        # torch.min(torch.norm(grid[batch_idx].unsqueeze(1).unsqueeze(1) - atom_coord[batch == batch_idx].unsqueeze(0).unsqueeze(0) + torch.stack([i * cell[batch_idx][0]+ j * cell[batch_idx][1]+ k * cell[batch_idx][2] for i in [-1, 0, 1]for j in [-1, 0, 1]for k in [-1, 0, 1]],dim=0,).reshape(1,27,1,3), dim=-1,),dim=[-1, -2],)[0]> self.mask_cutoff

        if self.residual:
            grid_flat = grid.view(-1, 3)
            grid_batch = torch.arange(n_graph, device=grid.device).repeat_interleave(
                n_sample
            )
            grid_dst, node_src = radius(
                atom_coord, grid_flat, self.cutoff, batch, grid_batch
            )
            grid_edge = grid_flat[grid_dst] - atom_coord[node_src]
            grid_len = torch.norm(grid_edge, dim=-1) + 1e-8
            grid_edge_feat = o3.spherical_harmonics(
                list(range(self.num_spherical + 1)),
                grid_edge / (grid_len[..., None] + 1e-8),
                normalize=False,
                normalization="integral",
            )
            grid_edge_embed = soft_one_hot_linspace(
                grid_len,
                start=0.0,
                end=self.cutoff,
                number=self.radial_embed_size,
                basis="gaussian",
                cutoff=False,
            ).mul(self.radial_embed_size**0.5)
            res_feat = feat
            if self.model_sharing == False:
                res_feat = infgcn_feat

            residue = self.residue(
                (node_src, grid_dst),
                res_feat,
                grid_edge_feat,
                grid_edge_embed,
                dim_size=grid_flat.size(0),
            )
        else:
            residue = 0.0

        # Orbital
        sample_vec = grid[batch] - atom_coord.unsqueeze(-2)
        if self.pbc:
            sample_vec, frac = pbc_vec(sample_vec, cell[batch])
        orbital = self.orbital(sample_vec)
        # (1) Coefficients of the orbital
        if self.model_sharing == False:
            density = (orbital * infgcn_feat.unsqueeze(1)).sum(dim=-1)
        else:
            density = (orbital * feat.unsqueeze(1)).sum(dim=-1)
        density = scatter(density, batch, dim=0, reduce="sum")
        # (2) Scalar fields
        scalar_field = scalar_field_gcn.reshape(grid.size(0), grid.size(1)).real
        if self.residual:
            density = density + residue.view(*density.size())

        # density_threshold = density.max() * 0.01
        if self.scalar_mask:
            scalar_field = scalar_field * mask.cuda()
            if self.scalar_inv:
                density = density * (1 - mask.cuda())

        # (3) Sum up / Product
        if self.positive_output:
            density_res = density + scalar_field
            density_res[(density_res < 0) & (density > 0) & (scalar_field < 0)] = 0
            scalar_field[(density_res < 0) & (density > 0) & (scalar_field < 0)] = 0
            density = density_res
            # density[density > 0 & scalar_field <0] = 0
        else:
            density = density + scalar_field
        # if self.positive_output:
        #     density[density < 0] = 0

        scalar_field_influence = (scalar_field.abs().sum() / density.sum()).detach()

        log_dict = {}
        log_dict["avg_probe_degree"] = avg_probe_degree
        log_dict["avg_probe_exist_degree"] = avg_probe_exist_degree
        log_dict["avg_probe_exist_rate"] = avg_probe_exist_rate

        log_dict["avg_sample_degree"] = avg_sample_degree
        log_dict["avg_sample_exist_degree"] = avg_sample_exist_degree
        log_dict["avg_sample_exist_rate"] = avg_sample_exist_rate

        log_dict["density"] = density

        log_dict["scalar_field_influence"] = scalar_field_influence
        log_dict["coefficient_field"] = density - scalar_field
        log_dict["scalar_field"] = scalar_field
        log_dict["probe"] = probe_log

        self.draw_hist = True

        return log_dict

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1]
        )
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1]
        )
        gridz = torch.linspace(0, 1, size_z)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1]
        )
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)
