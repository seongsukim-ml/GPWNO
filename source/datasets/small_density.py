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
        return (
            Data(x=self.atom_type, pos=self.atom_coords[item]),
            self.densities[item],
            self.grid_coord,
            info,
        )

    def __len__(self):
        return self.atom_coords.shape[0]
