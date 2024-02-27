from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from source.common.visualize import draw_stack, draw_stack_probe
from lightning.pytorch.loggers import WandbLogger
from source.models.utils import BaseModule
import wandb
import numpy as np
from typing import List, Dict, Optional
import math
import ase
import source.baseline.deepdft_layer as layer
from source.baseline.deepdft_layer import ShiftedSoftplus
from source.baseline.deepdft import *

from source.datasets.density_deepdft import probes_to_graph
from source.datasets.density_deepdft import CollateFuncAtoms, DensityGridIterator
from source.datasets.density_deepdft import collate_list_of_dicts
from mpire import WorkerPool


class deepDFT(BaseModule):
    def __init__(
        self,
        num_interactions,
        hidden_state_size,
        cutoff,
        gaussian_expansion_step=0.1,
        use_painn_model=False,
        distance_embedding_size=30,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cutoff = cutoff
        self.use_painn_model = use_painn_model
        if self.use_painn_model:
            self.atom_model = PainnAtomRepresentationModel(
                num_interactions,
                hidden_state_size,
                cutoff,
                distance_embedding_size,
            )

            self.probe_model = PainnProbeMessageModel(
                num_interactions,
                hidden_state_size,
                cutoff,
                distance_embedding_size,
            )
        else:
            self.atom_model = AtomRepresentationModel(
                num_interactions,
                hidden_state_size,
                cutoff,
                gaussian_expansion_step,
            )

            self.probe_model = ProbeMessageModel(
                num_interactions,
                hidden_state_size,
                cutoff,
                gaussian_expansion_step,
            )

        self.rotate = False
        self.no_log = ["scalar_field", "coefficient_field", "probe", "min_dist"]
        self.draw_hist = False

    def forward(self, input_dict):
        """
        Network forward
        :param atom_types: atom types of (N,)
        :param atom_coord: atom coordinates of (N, 3)
        :param grid: coordinates at grid points of (G, K, 3)
        :param batch: batch index for each node of (N,)
        :param infos: list of dictionary containing additional information
        :return: predicted value at each grid point of (G, K)
        """
        if self.use_painn_model:
            atom_representation_scalar, atom_representation_vector = self.atom_model(
                input_dict
            )
            probe_result = self.probe_model(
                input_dict, atom_representation_scalar, atom_representation_vector
            )
        else:
            atom_representation = self.atom_model(input_dict)
            probe_result = self.probe_model(input_dict, atom_representation)
        return probe_result

    """
    'nodes', 'atom_edges', 'atom_edges_displacement',
    'num_atom_edges', 'atom_xyz', 'num_nodes',
    'probe_edges', 'probe_edges_displacement',
    'num_probe_edges', 'probe_xyz', 'num_probes', 'cell'
    """

    def training_step(self, batch, batch_idx):
        batch_size = batch["num_nodes"].size(0)
        result = self(batch)
        pred = result
        densities = batch["probe_target"]
        loss = nn.MSELoss()(pred, densities)
        mae = torch.abs(pred.detach() - densities).sum() / densities.sum()
        mae_abs = torch.abs(pred.detach() - densities).sum() / abs(densities).sum()
        self.log_dict(
            {"train/loss": loss, "train/mae": mae, "train/mae_abs": mae_abs},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        # import pdb;
        # pdb.set_trace()
        batch_size = batch["num_nodes"].size(0)
        result = self(batch)
        pred = result
        densities = batch["probe_target"]
        loss = nn.MSELoss()(pred, densities)
        mae = torch.abs(pred.detach() - densities).sum() / densities.sum()
        mae_abs = torch.abs(pred.detach() - densities).sum() / abs(densities).sum()
        self.log_dict(
            {"val/loss": loss, "val/mae": mae, "val/mae_abs": mae_abs},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    # def on_validation_epoch_end(self, batch, outs):
    #     # outs is a list of whatever you returned in `validation_step`
    #     self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, inf_samples=4096):
        rot = "_rot" if self.rotate else ""
        # import pdb;
        # pdb.set_trace()
        batch_size = 1
        # print(batch.keys())
        densities = batch["density"]  # "density", "atoms", "origin", "grid_position"

        # pred = self(g.x, g.pos, grid_coord, g.batch, infos)
        pred_density, error, rmse, mae_dft = self.inference_batch(
            batch, 4096, self.cutoff, set_pbc_to=False
        )
        mae = torch.abs(pred_density.detach() - densities).sum() / densities.sum()
        mae_abs = (
            torch.abs(pred_density.detach() - densities).sum() / abs(densities).sum()
        )

        self.log_dict(
            {
                f"test{rot}/dloss": rmse,
                f"test{rot}/dft_mae": mae_dft,
                f"test{rot}/dft_error": error,
                f"test{rot}/mae": mae,
                f"test{rot}/mae_abs": mae_abs,
            },
            batch_size=batch_size,
        )
        return rmse

    def inference_batch(
        self,
        densitydict,
        probe_count: int,
        cutoff: float,
        set_pbc_to: Optional[bool] = False,
    ):
        device = self.device
        num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        num_slices = int(math.ceil(num_positions / probe_count))

        if set_pbc_to is not None:
            new_atoms = densitydict["atoms"].copy()
            new_atoms.set_pbc(set_pbc_to)
        else:
            new_atoms = densitydict["atoms"]

        # meshgrid = densitydict["grid_position"]
        # with WorkerPool(n_jobs=16) as pool:
        #     sliced = pool.map(lambda i: get_slice(i, new_atoms, meshgrid, probe_count, cutoff), range(num_slices))

        density_iter = DensityGridIterator(densitydict, probe_count, self.cutoff, False)

        collate_fn = CollateFuncAtoms(
            cutoff=self.cutoff,
            n_samples=None,
            pin_memory=True,
            set_pbc_to=False,
        )
        graph_dict = collate_fn([densitydict])
        device_batch = {
            k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
        }

        if self.use_painn_model:
            atom_representation_scalar, atom_representation_vector = self.atom_model(
                device_batch
            )
        else:
            atom_representation = self.atom_model(device_batch)

        num_positions = np.prod(densitydict["grid_position"].shape[0:3])
        sum_abs_error = torch.tensor(0, dtype=torch.double, device=device)
        sum_squared_error = torch.tensor(0, dtype=torch.double, device=device)
        sum_target = torch.tensor(0, dtype=torch.double, device=device)
        density = []

        for slice_id, probe_graph_dict in enumerate(density_iter):
            flat_index = np.arange(
                slice_id * probe_count, min((slice_id + 1) * probe_count, num_positions)
            )
            pos_index = np.unravel_index(flat_index, densitydict["density"].shape[0:3])
            probe_target = torch.tensor(densitydict["density"][pos_index]).to(
                device=device, non_blocking=True
            )

            probe_dict = probe_graph_dict
            probe_dict = collate_list_of_dicts([probe_graph_dict])
            probe_dict = {
                k: v.to(device=device, non_blocking=True) for k, v in probe_dict.items()
            }

            # device_batch["probe_edges"] = torch.tensor(probe_dict["probe_edges"])
            # device_batch["probe_edges_displacement"] = torch.tensor(probe_dict["probe_edges_displacement"])
            # device_batch["probe_xyz"] = torch.tensor(probe_dict["probe_xyz"])
            # device_batch["num_probe_edges"] = torch.tensor(probe_dict["num_probe_edges"])
            # device_batch["num_probes"] = torch.tensor(probe_dict["num_probes"])
            device_batch["probe_edges"] = probe_dict["probe_edges"]
            device_batch["probe_edges_displacement"] = probe_dict[
                "probe_edges_displacement"
            ]
            device_batch["probe_xyz"] = probe_dict["probe_xyz"]
            device_batch["num_probe_edges"] = probe_dict["num_probe_edges"]
            device_batch["num_probes"] = probe_dict["num_probes"]

            if self.use_painn_model:
                res = self.probe_model(
                    device_batch, atom_representation_scalar, atom_representation_vector
                )
            else:
                res = self.probe_model(device_batch, atom_representation)
            error = probe_target - res
            sum_abs_error += torch.sum(torch.abs(error))
            sum_squared_error += torch.sum(torch.square(error))
            sum_target += torch.sum(probe_target)
            density.append(res.detach().cpu().numpy())

        voxel_volume = densitydict["atoms"].get_volume() / np.prod(
            densitydict["density"].shape
        )
        mae = sum_abs_error / num_positions
        rmse = torch.sqrt((sum_squared_error / num_positions))
        abserror_integral = sum_abs_error * voxel_volume
        total_integral = sum_target * voxel_volume
        error = abserror_integral / total_integral
        pred_density = np.concatenate(density, axis=1)
        pred_density = pred_density.reshape(densitydict["density"].shape)
        pred_density = torch.tensor(pred_density, dtype=torch.float32, device=device)

        return pred_density, error, rmse, mae


def get_slice(slice_index, atoms, meshgrid, probe_count, cutoff, neighborlist=None):
    num_positions = np.prod(meshgrid.shape[0:3])
    flat_index = np.arange(
        slice_index * probe_count, min((slice_index + 1) * probe_count, num_positions)
    )
    pos_index = np.unravel_index(flat_index, meshgrid.shape[0:3])
    probe_pos = meshgrid[pos_index]
    probe_edges, probe_edges_displacement = probes_to_graph(
        atoms, probe_pos, cutoff, neighborlist
    )

    if not probe_edges:
        probe_edges = [np.zeros((0, 2), dtype=np.int)]
        probe_edges_displacement = [np.zeros((0, 3), dtype=np.float32)]

    res = {
        "probe_edges": np.concatenate(probe_edges, axis=0),
        "probe_edges_displacement": np.concatenate(
            probe_edges_displacement, axis=0
        ).astype(np.float32),
    }
    res["num_probe_edges"] = res["probe_edges"].shape[0]
    res["num_probes"] = len(flat_index)
    res["probe_xyz"] = probe_pos.astype(np.float32)

    return res
