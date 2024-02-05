import random

import yaml
from easydict import EasyDict
import numpy as np
import torch


def load_config(config_path):
    """Load configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return EasyDict(config)


def seed_all(seed):
    """Seed all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_optimizer(cfg, model):
    """Get optimizer from config."""
    if cfg.type == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=(cfg.beta1, cfg.beta2,)
        )
    else:
        raise NotImplementedError(f'Optimizer not supported: {cfg.type}')


def get_scheduler(cfg, optimizer):
    """Get scheduler from config."""
    if cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=cfg.factor,
            patience=cfg.patience,
            min_lr=cfg.min_lr,
        )
    elif cfg.type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.milestones,
            gamma=cfg.gamma,
        )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=cfg.gamma,
        )
    else:
        raise NotImplementedError(f'Scheduler not supported: {cfg.type}')


def count_parameters(model):
    """Count the number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())
