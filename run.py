from pathlib import Path
from typing import List

import os
import wandb

import omegaconf
from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.utils import instantiate, log
from hydra.core.hydra_config import HydraConfig

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything, Callback

from source.common.utils import build_callbacks, log_hyperparameters, PROJECT_ROOT

import warnings

warnings.simplefilter("ignore", UserWarning)


def run(cfg: DictConfig) -> None:
    """
    Generic train loop

    :param cfg: run configuration, defined by Hydra in /conf
    """

    """ Set up the seeds """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    """ Hydra run directory """
    if HydraConfig.get().mode.name == "MULTIRUN":
        hydra_dir = Path(
            HydraConfig.get().sweep.dir + "/" + HydraConfig.get().sweep.subdir
        )
        os.chdir("../" + HydraConfig.get().sweep.subdir)
    else:
        hydra_dir = Path(HydraConfig.get().run.dir)

    log.info(f"Saving os.getcwd is <{os.getcwd()}>")
    log.info(f"Saving hydra_dir is <{hydra_dir}>")

    """ Instantiate datamodule """
    log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = instantiate(
        cfg.data.datamodule,
        _recursive_=False,
        temp_folder=cfg.expname,
        num_fourier=cfg.model.num_fourier,
        use_max_cell=cfg.model.use_max_cell,
        equivariant_frame=cfg.model.equivariant_frame,
    )

    """ Instantiate model """
    log.info(f"Instantiating <{cfg.model._target_}>")
    model: pl.LightningModule = instantiate(
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
        log.info("Instantiating <WandbLogger>")
        wandb_config = cfg.logging.wandb
        wandb_logger = WandbLogger(
            **wandb_config,
            save_dir=hydra_dir,
            tags=cfg.core.tags,
        )
        log.info("W&B is now watching <{cfg.logging.wandb_watch.log}>!")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )

    """ Store the YaML config separately into the wandb dir """
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (hydra_dir / "hparams.yaml").write_text(yaml_conf)

    """ Trainer instantiation """
    log.info("Instantiating the Trainer")
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

    """ Data preparation """
    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_data_loader = datamodule.val_dataloader()
    test_data_loader = datamodule.test_dataloader()

    """ Train and test"""
    log.info("Starting training!")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_data_loader,
    )

    log.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_data_loader, ckpt_path="best")

    if datamodule.rotate:
        model.rotate = True
        log.info("Starting testing on rotated data!")
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
