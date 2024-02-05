from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from source.common.visualize import draw_stack
from lightning.pytorch.loggers import WandbLogger
from source.models.utils import BaseModule


class InfGCN_interface(BaseModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        self.rotate = False
        super().__init__(*args, **kwargs)

    def forward(self, atom_types, atom_coord, grid, batch, infos):
        """
        Network forward
        :param atom_types: atom types of (N,)
        :param atom_coord: atom coordinates of (N, 3)
        :param grid: coordinates at grid points of (G, K, 3)
        :param batch: batch index for each node of (N,)
        :param infos: list of dictionary containing additional information
        :return: predicted value at each grid point of (G, K)
        """
        return

    def training_step(self, batch, batch_idx):
        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)
        pred = self(g.x, g.pos, grid_coord, g.batch, infos)
        loss = nn.MSELoss()(pred, densities)
        mae = torch.abs(pred.detach() - densities).sum() / densities.sum()
        self.log_dict(
            {"train/loss": loss, "train/mae": mae},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)

        pred = self(g.x, g.pos, grid_coord, g.batch, infos)
        loss = nn.MSELoss()(pred, densities)
        mae = torch.abs(pred.detach() - densities).sum() / densities.sum()
        self.log_dict(
            {"val/loss": loss, "val/mae": mae},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def test_step(self, batch, batch_idx, num_vis=2, inf_samples=4096):
        rot = "_rot" if self.rotate else ""

        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)

        # pred = self(g.x, g.pos, grid_coord, g.batch, infos)
        pred, loss, mae = self.inference_batch(
            g, densities, grid_coord, infos, grid_batch_size=inf_samples
        )

        if batch_idx == 0 and self.hparams.logging.draw_predictions == True:
            for vis_idx, (p, d, info) in enumerate(zip(pred, densities, infos)):
                if vis_idx >= num_vis:
                    break
                # import pdb; pdb.set_trace()
                shape = info["shape"]
                mask = g.batch == vis_idx
                atom_type, coord = g.x[mask], g.pos[mask]
                grid_cell = info["cell"] / torch.FloatTensor(shape).view(3, 1).cuda()
                coord = coord @ torch.linalg.inv(grid_cell)
                num_voxel = shape[0] * shape[1] * shape[2]
                d, p = d[:num_voxel].view(*shape), p[:num_voxel].view(*shape)
                self.logger.log_image(
                    key=f"inf{rot}/gt_{vis_idx}",
                    images=[draw_stack(d, atom_type, coord)],
                )
                self.logger.log_image(
                    key=f"inf{rot}/pred_{vis_idx}",
                    images=[draw_stack(p, atom_type, coord)],
                )
                self.logger.log_image(
                    key=f"inf{rot}/diff_{vis_idx}",
                    images=[draw_stack(d - p, atom_type, coord)],
                )

        loss = loss.mean()
        mae = mae.mean()

        self.log_dict(
            {
                f"test{rot}/loss": loss,
                f"test{rot}/mae": mae,
            },
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    def inference_batch(self, g, density, grid_coord, infos, grid_batch_size=None):
        if grid_batch_size is None:
            preds = self(g.x, g.pos, grid_coord, g.batch, infos)
        else:
            preds = []
            for grid in grid_coord.split(grid_batch_size, dim=1):
                preds.append(self(g.x, g.pos, grid.contiguous(), g.batch, infos))
            preds = torch.cat(preds, dim=1)
        # voxel
        mask = (density > 0).float()
        preds = preds * mask
        density = density * mask

        diff = torch.abs(preds - density)
        sum_idx = tuple(range(1, density.dim()))
        loss = diff.pow(2).sum(sum_idx) / mask.sum(sum_idx)
        mae = diff.sum(sum_idx) / density.sum(sum_idx)
        return preds, loss, mae

    def predict_step(self, batch, batch_idx, dataloader_idx=None, inf_samples=4096):
        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)

        # pred = self(g.x, g.pos, grid_coord, g.batch, infos)
        pred, loss, mae = self.inference_batch(
            g, densities, grid_coord, infos, grid_batch_size=inf_samples
        )

        # self.log_dict(
        #     {"pred/loss": loss, "pred/mae": mae},
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        #     batch_size=batch_size,
        # )
        return {
            "pred": pred,
            "density": densities,
            "atom_type": g.x,
            "atom_coord": g.pos,
            "grid_coord": grid_coord,
            "loss": loss,
            "mae": mae,
            "cell": [info["cell"] for info in infos],
        }