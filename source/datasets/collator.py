import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch_geometric.data import Batch


class DensityCollator:
    def __init__(self, n_samples=None):
        self.n_samples = n_samples

    def __call__(self, batch):
        g, densities, grid_coord, infos = zip(*batch)
        g = Batch.from_data_list(g)

        if self.n_samples is None:
            densities = pad_sequence(densities, batch_first=True, padding_value=-1)
            grid_coord = pad_sequence(grid_coord, batch_first=True, padding_value=0.0)
            return g, densities, grid_coord, infos

        sampled_density, sampled_grid = [], []
        for d, coord in zip(densities, grid_coord):
            idx = random.sample(range(d.size(0)), self.n_samples)
            sampled_density.append(d[idx])
            sampled_grid.append(coord[idx])
        sampled_density = torch.stack(sampled_density, dim=0)
        sampled_grid = torch.stack(sampled_grid, dim=0)
        return g, sampled_density, sampled_grid, infos


# fmt: off
class DensityVoxelCollator:
    def __call__(self, batch):
        g, densities, grid_coord, infos = zip(*batch)
        g = Batch.from_data_list(g)
        shapes = [info['shape'] for info in infos]
        max_shape = np.array(shapes).max(0)

        padded_density, padded_grid = [], []
        for den, grid, shape in zip(densities, grid_coord, shapes):
            padded_density.append(F.pad(den.view(*shape), (
                0, max_shape[2] - shape[2],
                0, max_shape[1] - shape[1],
                0, max_shape[0] - shape[0]
            ), value=-1))
            padded_grid.append(F.pad(grid.view(*shape, 3), (
                0, 0,
                0, max_shape[2] - shape[2],
                0, max_shape[1] - shape[1],
                0, max_shape[0] - shape[0]
            ), value=0.))
        densities = torch.stack(padded_density, dim=0)
        grid_coord = torch.stack(padded_grid, dim=0)
        return g, densities, grid_coord, infos
