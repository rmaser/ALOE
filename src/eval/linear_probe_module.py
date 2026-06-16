import math
from typing import Literal, Optional, Sequence

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from loguru import logger

from src.models.hf.aloe.modules.bcos_core import (
    BcosUnnormedLinear,
    BcosUnnormedLinear_v2,
)
from src.modules.logit_layer import LogitLayer

BcosImpl = Literal["v1", "v2"]


class LinearProbeModule(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        b_classifier: float = 2.0,
        bcos_impl: BcosImpl = "v1",
        use_bcos_head: bool = True,
        loss_fn: Optional[nn.Module] = None,
        metric_prefix: str = "",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.metric_prefix = metric_prefix

        logger.info(
            "Initializing LinearProbeModule with {} features, {} classes, B={}, impl={}, bcos_head={}",
            feature_dim,
            num_classes,
            b_classifier,
            bcos_impl,
            use_bcos_head,
        )
        self.use_bcos_head = use_bcos_head
        if use_bcos_head:
            if bcos_impl == "v1":
                classifier_cls = BcosUnnormedLinear
            elif bcos_impl == "v2":
                classifier_cls = BcosUnnormedLinear_v2
            else:
                raise ValueError(f"bcos_impl must be 'v1' or 'v2', got {bcos_impl!r}")
            self.classifier = classifier_cls(feature_dim, num_classes, b=b_classifier)
        else:
            self.classifier = nn.Linear(feature_dim, num_classes, bias=False)

        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        logger.info("Using loss function {}", self.loss_fn.__class__.__name__)

        loss_name = self.loss_fn.__class__.__name__
        self.use_logit_layer = "BCE" in loss_name
        if self.use_logit_layer:
            self.logit_layer = LogitLayer(logit_bias=-math.log(num_classes - 1))
        else:
            self.logit_layer = nn.Identity()
        if self.use_bcos_head:
            logger.info("Using B-cos classifier head")
        else:
            logger.info("Using standard nn.Linear classifier head")
        logger.info("Using logit layer: {}", self.use_logit_layer)

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.classifier(x)
        if self.use_logit_layer:
            x = self.logit_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.train_acc(logits, y)
        prefix = self.metric_prefix
        self.log(f"{prefix}train_loss", loss, prog_bar=True)
        self.log(f"{prefix}train_acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        self.val_acc(logits, y)
        prefix = self.metric_prefix
        self.log(f"{prefix}val_loss", loss, prog_bar=True)
        self.log(f"{prefix}val_acc", self.val_acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class MultiplexedLinearProbeModule(pl.LightningModule):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr_values: Sequence[float],
        weight_decay: float = 1e-4,
        max_epochs: int = 30,
        b_classifier: float = 2.0,
        bcos_impl: BcosImpl = "v1",
        use_bcos_head: bool = True,
        loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.lr_values = [float(lr) for lr in lr_values]
        self.metric_prefixes = [f"lr{str(lr).replace('.', '_')}/" for lr in self.lr_values]

        logger.info(
            "Initializing MultiplexedLinearProbeModule with {} features, {} classes, {} LR heads, B={}, impl={}, bcos_head={}",
            feature_dim,
            num_classes,
            len(self.lr_values),
            b_classifier,
            bcos_impl,
            use_bcos_head,
        )

        self.use_bcos_head = use_bcos_head
        self.bcos_impl = bcos_impl
        self.b_classifier = b_classifier
        if use_bcos_head:
            if bcos_impl == "v1":
                classifier_cls = BcosUnnormedLinear
            elif bcos_impl == "v2":
                classifier_cls = BcosUnnormedLinear_v2
            else:
                raise ValueError(f"bcos_impl must be 'v1' or 'v2', got {bcos_impl!r}")
            self.classifiers = nn.ModuleList(
                [classifier_cls(feature_dim, num_classes, b=b_classifier) for _ in self.lr_values]
            )
        else:
            self.classifiers = nn.ModuleList(
                [nn.Linear(feature_dim, num_classes, bias=False) for _ in self.lr_values]
            )

        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
        logger.info("Using loss function {}", self.loss_fn.__class__.__name__)

        loss_name = self.loss_fn.__class__.__name__
        self.use_logit_layer = "BCE" in loss_name
        self.logit_bias = -math.log(num_classes - 1) if self.use_logit_layer else None
        self.logit_temperature = None
        logger.info("Using logit layer: {}", self.use_logit_layer)

        self.train_losses = nn.ModuleList([torchmetrics.MeanMetric() for _ in self.lr_values])
        self.val_losses = nn.ModuleList([torchmetrics.MeanMetric() for _ in self.lr_values])
        self.train_accs = nn.ModuleList(
            [torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) for _ in self.lr_values]
        )
        self.val_accs = nn.ModuleList(
            [torchmetrics.Accuracy(task="multiclass", num_classes=num_classes) for _ in self.lr_values]
        )
        self.register_buffer("best_val_accs", torch.zeros(len(self.lr_values), dtype=torch.float32))
        self.register_buffer(
            "best_epoch_indices", torch.full((len(self.lr_values),), -1, dtype=torch.long)
        )
        self.best_classifier_state_dicts: list[dict[str, torch.Tensor] | None] = [
            None for _ in self.lr_values
        ]

    def _stacked_weights(self) -> torch.Tensor:
        if self.use_bcos_head:
            return torch.stack([classifier.linear.weight for classifier in self.classifiers], dim=0)
        return torch.stack([classifier.weight for classifier in self.classifiers], dim=0)

    def _apply_vectorized_classifier(self, x: torch.Tensor) -> torch.Tensor:
        weights = self._stacked_weights()
        logits = torch.einsum("bd,ncd->nbc", x, weights)

        if not self.use_bcos_head or self.b_classifier == 1:
            return logits

        if self.bcos_impl == "v2":
            normed_x = F.normalize(x, p=2, dim=-1, eps=1e-12)
            normed_w = F.normalize(weights, p=2, dim=-1, eps=1e-12)
            scale = torch.einsum("bd,ncd->nbc", normed_x, normed_w).abs()
        else:
            norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).unsqueeze(0) + 1e-12
            scale = logits.abs() / norm

        if self.b_classifier != 2:
            scale = scale.clamp(min=1e-6).pow(self.b_classifier - 1)
        return scale * logits

    def forward(self, x):
        logits = self._apply_vectorized_classifier(x)
        if self.logit_temperature is not None:
            logits = logits * self.logit_temperature
        if self.logit_bias is not None:
            logits = logits + self.logit_bias
        return logits

    def _compute_losses(self, logits_per_lr: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.loss_fn(logits, y) for logits in logits_per_lr], dim=0)

    def _snapshot_best_state(self, idx: int, epoch: int, val_acc: torch.Tensor) -> None:
        self.best_val_accs[idx] = val_acc.detach().float()
        self.best_epoch_indices[idx] = int(epoch)
        self.best_classifier_state_dicts[idx] = {
            key: value.detach().cpu().clone()
            for key, value in self.classifiers[idx].state_dict().items()
        }

    def export_best_checkpoint(self, idx: int) -> dict[str, object] | None:
        best_state = self.best_classifier_state_dicts[idx]
        if best_state is None:
            return None

        state_dict = {f"classifier.{key}": value.clone() for key, value in best_state.items()}
        return {
            "state_dict": state_dict,
            "epoch": int(self.best_epoch_indices[idx].item()),
            "hyper_parameters": {
                "feature_dim": int(self.hparams.feature_dim),
                "num_classes": int(self.hparams.num_classes),
                "lr": float(self.lr_values[idx]),
                "weight_decay": float(self.hparams.weight_decay),
                "max_epochs": int(self.hparams.max_epochs),
                "b_classifier": float(self.hparams.b_classifier),
                "bcos_impl": self.hparams.bcos_impl,
                "use_bcos_head": bool(self.hparams.use_bcos_head),
                "best_val_acc": float(self.best_val_accs[idx].item()),
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits_per_lr = self(x)
        losses = self._compute_losses(logits_per_lr, y)
        total_loss = losses.sum()

        for idx, prefix in enumerate(self.metric_prefixes):
            self.train_losses[idx].update(losses[idx].detach())
            self.train_accs[idx].update(logits_per_lr[idx], y)
            self.log(f"{prefix}train_loss_step", losses[idx].detach(), on_step=True, on_epoch=False)

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits_per_lr = self(x)
        losses = self._compute_losses(logits_per_lr, y)

        for idx in range(len(self.lr_values)):
            self.val_losses[idx].update(losses[idx].detach())
            self.val_accs[idx].update(logits_per_lr[idx], y)

        return losses.sum()

    def on_train_epoch_end(self) -> None:
        for idx, prefix in enumerate(self.metric_prefixes):
            self.log(f"{prefix}train_loss", self.train_losses[idx].compute(), prog_bar=(idx == 0))
            self.log(f"{prefix}train_acc", self.train_accs[idx].compute(), prog_bar=(idx == 0))
            self.train_losses[idx].reset()
            self.train_accs[idx].reset()

    def on_validation_epoch_end(self) -> None:
        if self.trainer is not None and self.trainer.sanity_checking:
            for idx in range(len(self.lr_values)):
                self.val_losses[idx].reset()
                self.val_accs[idx].reset()
            return

        for idx, prefix in enumerate(self.metric_prefixes):
            val_loss = self.val_losses[idx].compute()
            val_acc = self.val_accs[idx].compute()
            if (
                self.best_classifier_state_dicts[idx] is None
                or val_acc.detach().float() >= self.best_val_accs[idx]
            ):
                epoch = self.current_epoch if self.trainer is not None else 0
                self._snapshot_best_state(idx=idx, epoch=epoch, val_acc=val_acc)
            self.log(f"{prefix}val_loss", val_loss, prog_bar=(idx == 0))
            self.log(f"{prefix}val_acc", val_acc, prog_bar=(idx == 0))
            self.log(f"{prefix}best_val_acc", self.best_val_accs[idx], prog_bar=(idx == 0))
            self.log(f"{prefix}best_epoch", self.best_epoch_indices[idx], prog_bar=False)
            self.val_losses[idx].reset()
            self.val_accs[idx].reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": classifier.parameters(),
                    "lr": lr,
                    "weight_decay": self.hparams.weight_decay,
                }
                for classifier, lr in zip(self.classifiers, self.lr_values, strict=True)
            ]
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr_values,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
