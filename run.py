from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import omegaconf
import pytorch_lightning as pl


from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything, Callback
from pytorch_lightning.loggers import WandbLogger
import wandb
from source.common.utils import build_callbacks, log_hyperparameters, PROJECT_ROOT
import os


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """

    """ Set up the seeds """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    """ For debugging purposes """
    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.pl_trainer.fast_dev_run=}>. "
            f"Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        # cfg.train.pl_trainer.gpus = 0
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.logging.wandb.mode = "offline"

    """ Hydra run directory """
    if HydraConfig.get().mode.name == "MULTIRUN":
        hydra_dir = Path(
            HydraConfig.get().sweep.dir + "/" + HydraConfig.get().sweep.subdir
        )
        os.chdir("../" + HydraConfig.get().sweep.subdir)
    else:
        hydra_dir = Path(HydraConfig.get().run.dir)

    hydra.utils.log.info(f"Saving os.getcwd is <{os.getcwd()}>")
    hydra.utils.log.info(f"Saving hydra_dir is <{hydra_dir}>")

    """ Instantiate datamodule """
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )

    """ Instantiate model """
    hydra.utils.log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )

    """ Instantiate the callbacks """
    callbacks: List[Callback] = build_callbacks(cfg=cfg)

    """ Logger instantiation/configuration """
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            save_dir=hydra_dir,
            tags=cfg.core.tags,
        )
        hydra.utils.log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    """ Store the YaML config separately into the wandb dir """
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        accelerator="auto",
        default_root_dir=hydra_dir,
        logger=wandb_logger,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        # check_val_every_n_epoch=cfg.logging.val_check_interval,
        log_every_n_steps=1,
        **cfg.train.pl_trainer,  # max_steps 포함
    )

    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_data_loader = datamodule.val_dataloader()
    test_data_loader = datamodule.test_dataloader()

    hydra.utils.log.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_data_loader,
    )

    hydra.utils.log.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_data_loader, ckpt_path="best")

    if datamodule.rotate:
        model.rotate = True
        hydra.utils.log.info("Starting testing on rotated data!")
        test_rotate_dataloader = datamodule.test_rotate_dataloader()
        trainer.test(model=model, dataloaders=test_rotate_dataloader, ckpt_path="best")

    # Logger closing to release resources/avoid multi-run conflicts
    if wandb_logger is not None:
        wandb.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
