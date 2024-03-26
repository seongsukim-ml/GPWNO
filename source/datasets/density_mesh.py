import json
import os
import time

import numpy as np
import torch
from e3nn import o3
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import radius


Bohr = 0.529177  # Bohr radius in angstrom


def pbc_expand(atom_type, atom_coord):
    """
    Expand the atoms by periodic boundary condition to eight directions in the neighboring cells.
    :param atom_type: atom types, tensor of shape (n_atom,)
    :param atom_coord: atom coordinates, tensor of shape (n_atom, 3)
    :return: expanded atom types and coordinates
    """
    exp_type, exp_coord = [], []
    exp_direction = torch.FloatTensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
    )
    for a_type, a_coord in zip(atom_type, atom_coord):
        for direction in exp_direction:
            new_coord = a_coord + direction
            if (new_coord <= 1).all():
                exp_type.append(a_type)
                exp_coord.append(new_coord)
    return torch.LongTensor(exp_type), torch.stack(exp_coord, dim=0)


def rotate_voxel(shape, cell, density, rotated_grid):
    """
    Rotate the volumetric data using trilinear interpolation.
    :param shape: voxel shape, tensor of shape (3,)
    :param cell: cell vectors, tensor of shape (3, 3)
    :param density: original density, tensor of shape (n_grid,)
    :param rotated_grid: rotated grid coordinates, tensor of shape (n_grid, 3)
    :return: rotated density, tensor of shape (n_grid,)
    """
    density = density.view(1, 1, *shape)
    rotated_grid = rotated_grid.view(1, *shape, 3)
    shape = torch.FloatTensor(shape)
    grid_cell = cell / shape.view(3, 1)
    normalized_grid = (2 * rotated_grid @ torch.linalg.inv(grid_cell) - shape + 1) / (
        shape - 1
    )
    return F.grid_sample(
        density, torch.flip(normalized_grid, [-1]), mode="bilinear", align_corners=False
    ).view(-1)


class DensityDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        split_file,
        atom_file,
        temp_folder,
        num_fourier,
        use_max_cell,
        equivariant_frame,
        probe_cutoff,
        model_pbc,
        extension="CHGCAR",
        compression="lz4",
        rotate=False,
        pbc=False,
        num_samples=None,
        reverse_order=False,
        *args,
        **kwargs,
    ):
        """
        The density dataset contains volumetric data of molecules.
        :param root: data root
        :param split: data split, can be 'train', 'validation', 'test'
        :param split_file: the data split file containing file names of the split
        :param atom_file: atom information file
        :param extension: raw data file extension, can be 'CHGCAR', 'cube', 'json'
        :param compression: raw data compression, can be 'lz4', 'xz', or None (no compression)
        :param rotate: whether to rotate the molecule and the volumetric data
        :param pbc: whether the data satisfy the periodic boundary condition
        """
        super(DensityDataset, self).__init__()
        self.root = root
        self.mesh_temp_folder = os.path.join(root, "../GPWNO_mesh_calc/" + temp_folder)
        self.split = split
        self.extension = extension
        self.compression = compression
        self.rotate = rotate
        self.pbc = pbc
        self.model_pbc = model_pbc
        self.reverse_order = reverse_order

        self.probe_cutoff = probe_cutoff
        self.num_fourier = num_fourier
        self.use_max_cell = use_max_cell
        self.equivariant_frame = equivariant_frame

        self.file_pattern = f".{extension}"
        if compression is not None:
            self.file_pattern += f".{compression}"
        with open(os.path.join(split_file)) as f:
            # reverse the order so that larger molecules are tested first
            self.file_list = list(reversed(json.load(f)[split]))
            if reverse_order:
                self.file_list = self.file_list[::-1]
            if num_samples is not None:
                self.file_list = self.file_list[:num_samples]
        with open(atom_file) as f:
            atom_info = json.load(f)
        atom_list = [info["name"] for info in atom_info]
        self.atom_name2idx = {name: idx for idx, name in enumerate(atom_list)}
        self.atom_name2idx.update(
            {name.encode(): idx for idx, name in enumerate(atom_list)}
        )
        self.atom_num2idx = {
            info["atom_num"]: idx for idx, info in enumerate(atom_info)
        }
        self.idx2atom_num = {
            idx: info["atom_num"] for idx, info in enumerate(atom_info)
        }

        if extension == "CHGCAR" or extension == "chgcar":
            self.read_func = self.read_chgcar
        elif extension == "cube":
            self.read_func = self.read_cube
        elif extension == "json":
            self.read_func = self.read_json
        else:
            raise TypeError(f"Unknown extension {extension}")

        if compression == "lz4":
            import lz4.frame

            self.open = lz4.frame.open
        elif compression == "xz":
            import lzma

            self.open = lzma.open
        else:
            self.open = open

    def __getitem__(self, item):
        if self.compression == "lz4":
            file_name = f"{(self.file_list[item]+1):06}{self.file_pattern}"
        else:
            file_name = f"{(self.file_list[item])}{self.file_pattern}"

        try:
            with self.open(os.path.join(self.root, file_name)) as f:
                g, density, grid_coord, info = self.read_func(f)
        except EOFError:
            print("EOFError")
            print(f"Error reading {file_name} in {self.split} set, try again")
        except RuntimeError:
            print(f"Error reading {file_name} in {self.split} set")
            raise

        info["file_name"] = file_name

        if self.rotate:
            rot = o3.rand_matrix()
            center = info["cell"].sum(dim=0) / 2
            g.pos = (g.pos - center) @ rot.t() + center
            rotated_grid = (grid_coord - center) @ rot + center
            density = rotate_voxel(info["shape"], info["cell"], density, rotated_grid)
            info["rot"] = rot

        os.makedirs(self.mesh_temp_folder, exist_ok=True)
        mesh_save_file = os.path.join(self.mesh_temp_folder, f"{file_name}.pt")
        if not os.path.exists(mesh_save_file):
            if self.model_pbc:
                (
                    probe,
                    probe_dst,
                    probe_src,
                    probe_edge,
                    super_probe,
                    super_probe_dst,
                    super_probe_idx,
                ) = self.make_grid(g.pos, info["cell"])
                torch.save(
                    (
                        probe,
                        probe_dst,
                        probe_src,
                        probe_edge,
                        super_probe,
                        super_probe_dst,
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
                    super_probe_dst,
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
            info["super_probe_dst"] = super_probe_dst

        return g, density, grid_coord, info

    def __len__(self):
        return len(self.file_list)

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
                max_cell = torch.FloatTensor(self.max_cell_size).to(cell.device)
                max_cell = max_cell.unsqueeze(0).repeat(num_batch, 1, 1)
            cell_inp = max_cell

        probe = torch.einsum("ijkl,blm->bijkm", probe, cell_inp).detach()  # (N,f,f,f,3)
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
            return (
                probe,
                probe_dst,
                probe_src,
                probe_edge,
                super_probe,
                super_probe_dst,
                super_probe_idx,
            )
        else:
            return probe, probe_dst, probe_src, probe_edge

    def read_chgcar(self, fileobj):
        """Read atoms and data from CHGCAR file."""
        readline = fileobj.readline
        readline()  # the first comment line
        scale = float(readline())  # the scaling factor (lattice constant)

        # the upcoming three lines contain the cell information
        cell = torch.empty(3, 3, dtype=torch.float)
        for i in range(3):
            cell[i] = torch.FloatTensor([float(s) for s in readline().split()])
        cell = cell * scale

        # the sixth line specifies the constituting elements
        elements = readline().split()
        # the seventh line supplies the number of atoms per atomic species
        n_atoms = [int(s) for s in readline().split()]
        # the eighth line is always "Direct" in our application
        readline()

        tot_atoms = sum(n_atoms)
        atom_type = torch.empty(tot_atoms, dtype=torch.long)
        atom_coord = torch.empty(tot_atoms, 3, dtype=torch.float)
        # the upcoming lines contains the atomic positions in fractional coordinates
        idx = 0
        for elem, n in zip(elements, n_atoms):
            atom_type[idx : idx + n] = self.atom_name2idx[elem]
            for _ in range(n):
                atom_coord[idx] = torch.FloatTensor(
                    [float(s) for s in readline().split()]
                )
                idx += 1
        if self.pbc:
            atom_type, atom_coord = pbc_expand(atom_type, atom_coord)
        # the coordinates are fractional, convert them to cartesian
        atom_coord = atom_coord @ cell
        g = Data(x=atom_type, pos=atom_coord)

        readline()  # an empty line
        shape = [int(s) for s in readline().split()]  # grid size
        n_grid = shape[0] * shape[1] * shape[2]
        # the grids are corner-aligned
        x_coord = (
            torch.linspace(0, shape[0] - 1, shape[0]).unsqueeze(-1) / shape[0] * cell[0]
        )
        y_coord = (
            torch.linspace(0, shape[1] - 1, shape[1]).unsqueeze(-1) / shape[1] * cell[1]
        )
        z_coord = (
            torch.linspace(0, shape[2] - 1, shape[2]).unsqueeze(-1) / shape[2] * cell[2]
        )
        grid_coord = (
            x_coord.view(-1, 1, 1, 3)
            + y_coord.view(1, -1, 1, 3)
            + z_coord.view(1, 1, -1, 3)
        )
        grid_coord = grid_coord.view(-1, 3)

        # the augmented occupancies are ignored
        density = torch.FloatTensor([float(s) for s in fileobj.read().split()[:n_grid]])
        # the value stored is the charge within a grid instead of the charge density
        # divide the charge by the grid volume to get the density
        volume = torch.linalg.det(cell).abs()
        density = density / volume
        # CHGCAR file stores the density as Z-Y-X, convert them to X-Y-Z
        density = (
            density.view(shape[2], shape[1], shape[0])
            .transpose(0, 2)
            .contiguous()
            .view(-1)
        )
        return g, density, grid_coord, {"shape": shape, "cell": cell}
