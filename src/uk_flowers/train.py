from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from uk_flowers.data.datamodule import FlowerDataModule
from uk_flowers.model.module import FlowerClassifier
from uk_flowers.utils.common import seed_everything


def train(cfg: DictConfig) -> None:
    """
    Training pipeline.
    """
    # Set seed
    seed_everything(cfg.project.seed)

    # DataModule
    datamodule = FlowerDataModule(
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Model
    model = FlowerClassifier(
        arch=cfg.model.arch,
        num_classes=cfg.model.num_classes,
        pretrained=cfg.model.pretrained,
        lr=cfg.train.optimizer.lr,
        momentum=cfg.train.optimizer.get("momentum", 0.9),
        weight_decay=cfg.train.optimizer.weight_decay,
        freeze_backbone=cfg.model.freeze_backbone,
        optimizer_name=cfg.train.optimizer.name,
        scheduler_name=cfg.train.get("scheduler", {}).get("name"),
        scheduler_params=cfg.train.get("scheduler"),
    )

    # Logger
    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.logger.experiment_name,
        tracking_uri=cfg.logger.tracking_uri,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.train.save_dir,
        filename=f"{cfg.model.arch}-{{epoch:02d}}-{{val_acc:.2f}}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator=cfg.project.device,
        devices=1,  # Assuming single GPU/CPU for now
        logger=mlflow_logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Note: We do not run trainer.test() here because the test set is unlabeled (for Kaggle).
    # Validation metrics during training are used for evaluation.
