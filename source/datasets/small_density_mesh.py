import os

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np

# 0 = C, 1 = H, 2 = O

ATOM_TYPES = {
    "benzene": torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
    "ethanol": torch.LongTensor([0, 0, 2, 1, 1, 1, 1, 1, 1]),
    "phenol": torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 1, 1]),
    "resorcinol": torch.LongTensor([0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1]),
    "ethane": torch.LongTensor([0, 0, 1, 1, 1, 1, 1, 1]),
    "malonaldehyde": torch.LongTensor([2, 0, 0, 0, 2, 1, 1, 1, 1]),
}


class SmallDensityDataset(Dataset):
    def __init__(
        self,
        root,
        mol_name,
        split,
        temp_folder,
        num_fourier,
        use_max_cell,
        equivariant_frame,
        probe_cutoff,
        model_pbc,
        *args,
        **kwargs,
    ):
        """
        Density dataset for small molecules in the MD datasets.
        Note that the validation and test splits are the same.
        :param root: data root
        :param mol_name: name of the molecule
        :param split: data split, can be 'train', 'validation', 'test'
        """
        super(SmallDensityDataset, self).__init__()
        assert mol_name in (
            "benzene",
            "ethanol",
            "phenol",
            "resorcinol",
            "ethane",
            "malonaldehyde",
        )
        self.root = root
        self.mol_name = mol_name
        self.split = split
        if split == "validation":
            split = "test"

        self.n_grid = 50  # number of grid points along each dimension
        self.grid_size = 20.0  # box size in Bohr
        self.data_path = os.path.join(root, mol_name, f"{mol_name}_{split}")

        self.atom_type = ATOM_TYPES[mol_name]
        self.atom_coords = torch.FloatTensor(
            np.load(os.path.join(self.data_path, "structures.npy"))
        )
        self.densities = self._convert_fft(
            np.load(os.path.join(self.data_path, "dft_densities.npy"))
        )
        self.grid_coord = self._generate_grid()

        self.mesh_temp_folder = os.path.join(root, "../GPWNO_mesh_calc/" + temp_folder)
        self.probe_cutoff = probe_cutoff
        self.num_fourier = num_fourier
        self.use_max_cell = use_max_cell
        self.equivariant_frame = equivariant_frame

    def _convert_fft(self, fft_coeff):
        # The raw data are stored in Fourier basis, we need to convert them back.
        print(f"Precomputing {self.split} density from FFT coefficients ...")
        fft_coeff = torch.FloatTensor(fft_coeff).to(torch.complex64)
        d = fft_coeff.view(-1, self.n_grid, self.n_grid, self.n_grid)
        hf = self.n_grid // 2
        # first dimension
        d[:, :hf] = (d[:, :hf] - d[:, hf:] * 1j) / 2
        d[:, hf:] = torch.flip(d[:, 1 : hf + 1], [1]).conj()
        d = torch.fft.ifft(d, dim=1)
        # second dimension
        d[:, :, :hf] = (d[:, :, :hf] - d[:, :, hf:] * 1j) / 2
        d[:, :, hf:] = torch.flip(d[:, :, 1 : hf + 1], [2]).conj()
        d = torch.fft.ifft(d, dim=2)
        # third dimension
        d[..., :hf] = (d[..., :hf] - d[..., hf:] * 1j) / 2
        d[..., hf:] = torch.flip(d[..., 1 : hf + 1], [3]).conj()
        d = torch.fft.ifft(d, dim=3)
        return torch.flip(d.real.view(-1, self.n_grid**3), [-1]).detach()

    def _generate_grid(self):
        x = torch.linspace(self.grid_size / self.n_grid, self.grid_size, self.n_grid)
        return (
            torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1)
            .view(-1, 3)
            .detach()
        )

    def __getitem__(self, item):
        info = {
            "cell": torch.eye(3) * self.grid_size,
            "shape": [self.n_grid, self.n_grid, self.n_grid],
        }

        mesh_save_file = os.path.join(
            self.mesh_temp_folder, f"{self.mol_name}_{item}.pt"
        )
        os.makedirs(self.mesh_temp_folder, exist_ok=True)
        if not os.path.exists(mesh_save_file):
            if self.model_pbc:
                (
                    probe,
                    probe_dst,
                    probe_src,
                    probe_edge,
                    super_probe,
                    super_probe_idx,
                ) = self.make_grid(g.pos, info["cell"])
                torch.save(
                    (
                        probe,
                        probe_dst,
                        probe_src,
                        probe_edge,
                        super_probe,
                        super_probe_idx,
                    ),
                    mesh_save_file,
                )
            else:
                probe, probe_dst, probe_src, probe_edge = self.make_grid(
                    g.pos, info["cell"]
                )
                torch.save((probe, probe_dst, probe_src, probe_edge), mesh_save_file)
        else:
            if self.model_pbc:
                (
                    probe,
                    probe_dst,
                    probe_src,
                    probe_edge,
                    super_probe,
                    super_probe_idx,
                ) = torch.load(mesh_save_file)

            else:
                probe, probe_dst, probe_src, probe_edge = torch.load(mesh_save_file)

        info["probe"] = probe
        info["probe_dst"] = probe_dst
        info["probe_src"] = probe_src
        info["probe_edge"] = probe_edge
        if self.model_pbc:
            info["super_probe"] = super_probe
            info["super_probe_idx"] = super_probe_idx

        return (
            Data(x=self.atom_type, pos=self.atom_coords[item]),
            self.densities[item],
            self.grid_coord,
            info,
        )

    def make_grid(self, atom_coord, cell):
        num_batch = 1
        batch = torch.zeros(atom_coord.shape[0], dtype=torch.long).to(atom_coord.device)
        cell = cell.unsqueeze(0)
        bins_lin = torch.linspace(0, 1, self.num_fourier).to(cell.device)
        if self.model_pbc:
            half = len(bins_lin) // 2
            super_bins = torch.cat(
                [bins_lin[half:-1] - 1, bins_lin, bins_lin[1:half] + 1]
            ).to(cell.device)
            super_bins = torch.meshgrid(super_bins, super_bins, super_bins)
            super_probe = torch.stack(super_bins, dim=-1).to(cell.device)

            bins_idx = torch.arange(len(bins_lin))
            bins_idx[-1] = 0

            super_bins_idx = torch.cat(
                [bins_idx[half:-1], bins_idx, bins_idx[1:half]]
            ).to(cell.device)

            smart_idx = (
                torch.arange((len(bins_idx) - 1) ** 3)
                .reshape(len(bins_idx) - 1, len(bins_idx) - 1, len(bins_idx) - 1)
                .to(cell.device)
            )
            super_probe_idx_help = torch.stack(
                torch.meshgrid(super_bins_idx, super_bins_idx, super_bins_idx), dim=-1
            ).reshape(-1, 3)
            super_probe_idx = smart_idx[
                super_probe_idx_help[:, 0],
                super_probe_idx_help[:, 1],
                super_probe_idx_help[:, 2],
            ].reshape(-1)
            del super_probe_idx_help

        if self.use_max_cell and self.equivariant_frame:
            bins_lin -= 0.5
        bins = torch.meshgrid(bins_lin, bins_lin, bins_lin)

        probe = torch.stack(bins, dim=-1).to(cell.device)

        cell_inp = cell
        if self.use_max_cell:
            if self.max_cell_size is None:
                self.max_cell_size = 20

            if self.equivariant_frame:
                new_cell = []
                for i in range(num_batch):
                    atom_bat = atom_coord[batch == i]
                    atoms_centered = atom_bat - atom_bat.mean(dim=0)
                    R = torch.matmul(atoms_centered.t(), atoms_centered)
                    _, vec = torch.linalg.eigh(R)
                    vec /= torch.linalg.norm(
                        vec, dim=0
                    )  # orthogonal frame / equivariant
                    new_cell.append(vec)
                new_cell = torch.stack(new_cell, dim=0)
                max_cell = new_cell * self.max_cell_size
            else:
                max_cell_tensor = np.eyes(3) * self.max_cell_size
                max_cell = torch.FloatTensor(self.max_cell_size).to(cell.device)
                max_cell = max_cell.unsqueeze(0).repeat(num_batch, 1, 1)
            cell_inp = max_cell

        probe = torch.einsum("ijkl,blm->bijkm", probe, cell_inp).detach()  # (N,f,f,f,3)
        probe_log = probe.cpu()
        if self.use_max_cell and self.equivariant_frame:
            probe_log += atom_center.reshape(-1, 1, 1, 1, 3).cpu()
        probe_flat = probe.reshape(-1, 3)
        probe_batch = torch.arange(num_batch, device=cell.device).repeat_interleave(
            len(bins_lin) ** 3
        )
        if self.model_pbc:
            super_probe = torch.einsum(
                "ijkl,blm->bijkm", super_probe, cell_inp
            ).detach()
            super_probe_flat = super_probe.reshape(-1, 3)
            super_probe_batch = torch.arange(
                num_batch, device=cell.device
            ).repeat_interleave(len(super_bins_idx) ** 3)
            super_probe_idx_batch = torch.arange(
                num_batch, device=cell.device
            ).repeat_interleave(len(super_probe_idx)) * len(
                bins_lin
            ) ** 3 + super_probe_idx.repeat(
                num_batch
            )

        if self.model_pbc:
            super_probe_dst, probe_src = radius(
                atom_coord,
                super_probe_flat,
                self.probe_cutoff,
                batch,
                super_probe_batch,
            )
            probe_dst = super_probe_idx_batch[super_probe_dst]
        else:
            probe_dst, probe_src = radius(
                atom_coord, probe_flat, self.probe_cutoff, batch, probe_batch
            )  # probe_cutoff: (1 ~ 3), check connectivity

        if self.model_pbc:
            probe_edge = super_probe_flat[super_probe_dst] - atom_coord[probe_src]
        else:
            probe_edge = probe_flat[probe_dst] - atom_coord[probe_src]

        if self.model_pbc:
            return probe, probe_dst, probe_src, probe_edge, super_probe, super_probe_idx
        else:
            return probe, probe_dst, probe_src, probe_edge

    def __len__(self):
        return self.atom_coords.shape[0]
