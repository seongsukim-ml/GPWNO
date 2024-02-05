import lightning.pytorch as pl

# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader

# from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig


class MDDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        collate_fn: DictConfig,
        pin_memory: bool = True,
        *args,
        **kwargs
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.collate_fn = collate_fn

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # Assign train/val datasets for use in dataloaders
        if stage is None or stage == "fit":
            self.dataset_train = hydra.utils.instantiate(self.datasets.train)
            self.dataset_val = hydra.utils.instantiate(self.datasets.val)
        # Assign test dataset for use in dataloader(s)
        if stage is None or stage == "test":
            self.dataset_test = hydra.utils.instantiate(self.datasets.test)

    def train_dataloader(self):
        train_col = hydra.utils.instantiate(
            self.collate_fn, n_samples=self.datasets.train.n_samples
        )
        return DataLoader(
            self.dataset_train,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size.train,
            shuffle=self.datasets.train.shuffle,
            # drop_last=self.datasets.train.drop_last,
            num_workers=self.num_workers.train,
            collate_fn=train_col,
        )

    def val_dataloader(self):
        val_col = hydra.utils.instantiate(
            self.collate_fn, n_samples=self.datasets.val.n_samples
        )
        return DataLoader(
            self.dataset_val,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size.val,
            shuffle=False,
            # drop_last=self.datasets.val.drop_last,
            num_workers=self.num_workers.val,
            collate_fn=val_col,
        )
    
    def inf_dataloader(self):
        val_col = hydra.utils.instantiate(
            self.collate_fn, n_samples=None
        )
        return DataLoader(
            self.dataset_val,
            pin_memory=self.pin_memory,
            batch_size=1,
            shuffle=False,
            # drop_last=self.datasets.val.drop_last,
            num_workers=self.num_workers.val,
            collate_fn=val_col,
        )

    def test_dataloader(self):
        test_col = hydra.utils.instantiate(
            self.collate_fn, n_samples=self.datasets.test.n_samples
        )
        return DataLoader(
            self.dataset_test,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size.test,
            shuffle=False,
            # drop_last=self.datasets.test.drop_last,
            num_workers=self.num_workers.test,
            collate_fn=test_col,
        )
