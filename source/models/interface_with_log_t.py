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


class interface(BaseModule):
    def __init__(
        self,
        criterion="mse",
        *args,
        **kwargs,
    ):
        self.rotate = False
        super().__init__(*args, **kwargs)
        self.no_log = ["scalar_field", "coefficient_field", "probe"]
        self.draw_hist = False
        self.criterion = nn.L1Loss()

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
        result_dict = self(g.x, g.pos, grid_coord, g.batch, infos)
        pred = result_dict["density"]
        loss = self.criterion(pred, densities)
        mae = torch.abs(pred.detach() - densities).sum() / densities.sum()
        mae_abs = torch.abs(pred.detach() - densities).sum() / abs(densities).sum()
        self.log_dict(
            {"train/loss": loss, "train/mae": mae, "train/mae_abs": mae_abs},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        result_dict.pop("density")
        for elem in self.no_log:
            if elem in result_dict:
                result_dict.pop(elem)

        self.log_dict(
            {f"train_stat/{k}": v for k, v in result_dict.items()},
            batch_size=batch_size,
        )
        del result_dict
        return loss

    def validation_step(self, batch, batch_idx):
        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)
        result_dict = self(g.x, g.pos, grid_coord, g.batch, infos)
        pred = result_dict["density"]
        loss = self.criterion(pred, densities)
        mae = torch.abs(pred.detach() - densities).sum() / densities.sum()
        mae_abs = torch.abs(pred.detach() - densities).sum() / abs(densities).sum()
        self.log_dict(
            {"val/loss": loss, "val/mae": mae, "val/mae_abs": mae_abs},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        result_dict.pop("density")
        for elem in self.no_log:
            if elem in result_dict:
                result_dict.pop(elem)

        self.log_dict(
            {f"val_stat/{k}": v for k, v in result_dict.items()},
            batch_size=batch_size,
        )
        del result_dict

        return loss

    # def on_validation_epoch_end(self, batch, outs):
    #     # outs is a list of whatever you returned in `validation_step`
    #     self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, num_vis=2, inf_samples=4096):
        rot = "_rot" if self.rotate else ""

        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)

        # pred = self(g.x, g.pos, grid_coord, g.batch, infos)
        pred, loss, mae, mae_abs, logs = self.inference_batch(
            g, densities, grid_coord, infos, grid_batch_size=inf_samples
        )

        if batch_idx == 0 and self.hparams.logging.draw_predictions == True:
            for vis_idx, (p, d, info) in enumerate(zip(pred, densities, infos)):
                if vis_idx >= num_vis:
                    break
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
                if "scalar_field" in logs:
                    scalar_field = logs["scalar_field"][vis_idx][:num_voxel].view(
                        *shape
                    )
                    if "probe" in logs:
                        probe = (
                            logs["probe"][vis_idx] @ torch.linalg.inv(grid_cell).cpu()
                        )
                        self.logger.log_image(
                            key=f"inf{rot}/scalar_field_{vis_idx}",
                            images=[
                                draw_stack_probe(
                                    scalar_field, atom_type, coord, probe=probe
                                )
                            ],
                        )
                    else:
                        self.logger.log_image(
                            key=f"inf{rot}/scalar_field_{vis_idx}",
                            images=[draw_stack(scalar_field, atom_type, coord)],
                        )
                if "scalar_field" in logs:
                    coeff_field = p - logs["scalar_field"][vis_idx][:num_voxel].view(
                        *shape
                    )
                    self.logger.log_image(
                        key=f"inf{rot}/coeff_field_{vis_idx}",
                        images=[draw_stack(coeff_field, atom_type, coord)],
                    )
                if self.draw_hist:
                    wandb_logger = self.logger.experiment
                    make_table = lambda x: wandb.Table(
                        columns=["density"], data=[[pt] for pt in x.flatten().cpu()]
                    )

                    wandb_logger.log(
                        {
                            f"inf{rot}/density_hist_{vis_idx}": wandb.plot.histogram(
                                make_table(d),
                                "density",
                                title=f"Density Distribution_{vis_idx}",
                            )
                        }
                    )
                    wandb_logger.log(
                        {
                            f"inf{rot}/pred_hist_{vis_idx}": wandb.plot.histogram(
                                make_table(p),
                                "density",
                                title=f"Prediction Distribution_{vis_idx}",
                            )
                        }
                    )
                    wandb_logger.log(
                        {
                            f"inf{rot}/diff_hist_{vis_idx}": wandb.plot.histogram(
                                make_table(d - p),
                                "density",
                                title=f"Difference Distribution_{vis_idx}",
                            )
                        }
                    )

                    if "scalar_field" in logs:
                        wandb_logger.log(
                            {
                                f"inf{rot}/scalar_field_hist_{vis_idx}": wandb.plot.histogram(
                                    make_table(scalar_field),
                                    "density",
                                    title=f"Scalar Field Distribution_{vis_idx}",
                                )
                            }
                        )

                    if "coefficient_field" in logs:
                        wandb_logger.log(
                            {
                                f"inf{rot}/coeff_field_hist_{vis_idx}": wandb.plot.histogram(
                                    make_table(coeff_field),
                                    "density",
                                    title=f"Coefficient Field Distribution_{vis_idx}",
                                )
                            }
                        )

        loss = loss.mean()
        mae = mae.mean()
        mae_abs = mae_abs.mean()

        self.log_dict(
            {
                f"test{rot}/loss": loss,
                f"test{rot}/mae": mae,
                f"test{rot}/mae_abs": mae_abs,
            },
            on_step=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def inference_batch(self, g, density, grid_coord, infos, grid_batch_size=None):
        rot = "_rot" if self.rotate else ""
        log_scalar_field = False
        log_coeff = False
        log_probe = False

        if grid_batch_size is None:
            results = self(g.x, g.pos, grid_coord, g.batch, infos)
            preds = results["density"]
            probes = [[] for _ in range(g.batch.max().item() + 1)]
            if "scalar_field" in results:
                log_scalar_field = True
                scalar_fields = results["scalar_field"]
            # if "coefficient_field" in results:
            #     log_coeff = True
            #     coeff_fields = results["coefficient_field"]
            if self.draw_hist:
                density_hist = results["density"]
                coeff_field_hist = results["coefficient_field"]
                scalar_field_hist = results["scalar_field"]
            if "probe" in results:
                log_probe = True
                for b in range(grid.size(0)):
                    probe = (results["probe"][b]).reshape(-1, 3)
                    probes[b] = probe
        else:
            preds = []
            scalar_fields = []
            coeff_fields = []
            density_hist = []
            coeff_field_hist = []
            scalar_field_hist = []
            probes = [[] for _ in range(g.batch.max().item() + 1)]
            for grid in grid_coord.split(grid_batch_size, dim=1):
                results = self(g.x, g.pos, grid.contiguous(), g.batch, infos)
                pred = results["density"]
                preds.append(pred)
                if "scalar_field" in results:
                    log_scalar_field = True
                    scalar_field = results["scalar_field"]
                    scalar_fields.append(scalar_field)
                # if "coefficient_field" in results:
                #     log_coeff=True
                #     coeff_field = results["coefficient_field"]
                #     coeff_fields.append(coeff_field)
                if self.draw_hist:
                    density_hist.append(results["density"])
                    coeff_field_hist.append(results["coefficient_field"])
                    scalar_field_hist.append(results["scalar_field"])
                if "probe" in results:
                    log_probe = True
                    for b in range(grid.size(0)):
                        probe = (results["probe"][b]).reshape(-1, 3)
                        probes[b].append(probe)
            preds = torch.cat(preds, dim=1)
            if log_scalar_field:
                scalar_fields = torch.cat(scalar_fields, dim=1)
            if log_coeff:
                coeff_fields = torch.cat(coeff_fields, dim=1)
            if log_probe:
                for b in range(len(probes)):
                    probes[b] = torch.mean(torch.stack(probes[b]), dim=0)
        # voxel
        mask = (density > 0).float()
        preds = preds * mask
        density = density * mask
        if log_scalar_field:
            scalar_fields = scalar_fields * mask
        if log_coeff:
            coeff_fields = coeff_fields * mask

        diff = torch.abs(preds - density)
        sum_idx = tuple(range(1, density.dim()))
        loss = diff.pow(2).sum(sum_idx) / mask.sum(sum_idx)
        mae = diff.sum(sum_idx) / density.sum(sum_idx)
        mae_abs = diff.sum(sum_idx) / abs(density).sum(sum_idx)

        logs = {}
        if log_scalar_field:
            logs["scalar_field"] = scalar_fields
        # if log_coeff:
        #     logs["coefficient_field"] = coeff_fields
        if log_probe:
            logs["probe"] = probes

        return preds, loss, mae, mae_abs, logs

    def predict_step(self, batch, batch_idx, dataloader_idx=None, inf_samples=4096):
        g, densities, grid_coord, infos = batch
        batch_size = grid_coord.size(0)

        pred, loss, mae, mae_abs, logs = self.inference_batch(
            g, densities, grid_coord, infos, grid_batch_size=inf_samples
        )

        res = {
            "density": densities,
            "pred": pred,
            "atom_type": g.x,
            "atom_coord": g.pos,
            "grid_coord": grid_coord,
            "loss": loss,
            "mae": mae,
            "mae_abs": mae_abs,
            "cell": [info["cell"] for info in infos],
        }
        if "scalar_field" in logs:
            res.update({"scalar_field": logs["scalar_field"]})
        return res
