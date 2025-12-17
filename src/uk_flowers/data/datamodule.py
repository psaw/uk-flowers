from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from uk_flowers.data.dataset import TestDataset


class FlowerDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/flower_data",
        batch_size: int = 64,
        num_workers: int = 4,
        mean: list[float] = [0.485, 0.456, 0.406],
        std: list[float] = [0.229, 0.224, 0.225],
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mean = mean
        self.std = std

        self.train_dir = self.data_dir / "dataset" / "train"
        self.valid_dir = self.data_dir / "dataset" / "valid"
        self.test_dir = self.data_dir / "dataset" / "test"

    def prepare_data(self) -> None:
        """
        Download data if needed.
        This method is called only from a single GPU.
        Do not assign state here (e.g. self.x = y).
        """
        # Check if data exists, if not download via DVC
        if not self.data_dir.exists():
            print(f"Data directory {self.data_dir} not found. Downloading via DVC...")
            # Assuming DVC is configured and data is tracked
            # We can use dvc.api or subprocess to pull data
            # Here we use a simple subprocess call as it's robust for CLI tools
            import subprocess

            try:
                subprocess.run(["dvc", "pull"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error pulling data with DVC: {e}")
                # Fallback or raise error depending on requirements
                raise e
        else:
            print(f"Data directory {self.data_dir} exists.")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called on every GPU.
        Assignments are allowed here (self.x = y).
        """

        # Define transforms
        normalize = transforms.Normalize(mean=self.mean, std=self.std)

        self.train_transforms = transforms.Compose(
            [
                transforms.Pad(4, padding_mode="reflect"),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self.test_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.ImageFolder(str(self.train_dir), self.train_transforms)
            self.val_dataset = datasets.ImageFolder(str(self.valid_dir), self.test_transforms)

            # Store class mapping
            self.idx_to_class = {val: key for key, val in self.val_dataset.class_to_idx.items()}

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_dataset = TestDataset(self.test_dir, self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Test dataset usually processed one by one or small batches
            shuffle=False,
            num_workers=self.num_workers,
        )
