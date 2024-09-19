import torch.optim as optim


class PowerDecayScheduler(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, alpha=0.96, beta=1e5):
        scheduler_fn = lambda step: alpha ** (step / beta)
        super().__init__(optimizer=optimizer, lr_lambda=scheduler_fn)
