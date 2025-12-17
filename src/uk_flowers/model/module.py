from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torchvision import models
import pytorch_lightning as pl
from torch.optim import Optimizer


class FlowerClassifier(pl.LightningModule):
    def __init__(
        self,
        arch: str = "resnet18",
        num_classes: int = 102,
        pretrained: bool = True,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        freeze_backbone: bool = True,
        optimizer_name: str = "sgd",
        scheduler_name: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model setup
        if arch == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.model = models.resnet18(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        elif arch == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.model = models.resnet50(weights=weights)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Any:
        # Optimizer
        optimizer_name = self.hparams.get("optimizer_name", "sgd")
        lr = self.hparams.get("lr", 1e-2)
        weight_decay = self.hparams.get("weight_decay", 1e-4)

        optimizer: Optimizer
        if optimizer_name == "sgd":
            momentum = self.hparams.get("momentum", 0.9)
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        scheduler_name = self.hparams.get("scheduler_name")
        scheduler_params = self.hparams.get("scheduler_params", {})

        if scheduler_name:
            scheduler: Any
            if scheduler_name == "step_lr":
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_params.get("step_size", 7),
                    gamma=scheduler_params.get("gamma", 0.1),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": scheduler,
                }
            elif scheduler_name == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode=scheduler_params.get("mode", "min"),
                    factor=scheduler_params.get("factor", 0.1),
                    patience=scheduler_params.get("patience", 10),
                )
                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "monitor": "val_loss",
                    },
                }
            else:
                # Default or no scheduler
                return {"optimizer": optimizer}

        return {"optimizer": optimizer}
