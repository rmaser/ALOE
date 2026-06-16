import time
import functools
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Sequence

import hydra
import torch
import torch.nn as nn
import lightning.pytorch as L
from lightning.pytorch.callbacks import ModelCheckpoint
from loguru import logger
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader, TensorDataset

from src.eval.linear_probe_module import LinearProbeModule
from src.eval.evaluator import Evaluator
from src.eval.linear_probe_module import MultiplexedLinearProbeModule

torch.set_float32_matmul_precision('high')

class LinearProbeEvaluator(Evaluator):
    def __init__(
        self,
        batch_size: int = 256,
        lr: Union[float, List[float], Sequence[float]] = 1e-3,
        weight_decay: float = 1e-4,
        b_classifier: float = 2.0,
        bcos_impl: str = "v1",
        use_bcos_head: bool = True,
        num_workers: int = 4,
        feature_cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        feature_device: str = "cpu",
        loss_fn: Optional[Any] = None,
        trainer_config: Optional[Dict[str, Any]] = None,
        callbacks_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            feature_cache_dir=feature_cache_dir,
            force_recompute=force_recompute,
            feature_device=feature_device,
        )
        self.batch_size = batch_size
        if isinstance(lr, (list, tuple, ListConfig)):
            self.lr_values = list(lr)
        else:
            self.lr_values = [lr]
        self.weight_decay = weight_decay
        self.b_classifier = b_classifier
        self.num_workers = num_workers
        self.bcos_impl = bcos_impl
        self.use_bcos_head = use_bcos_head
        self.loss_fn_factory = loss_fn
        self.trainer_config = trainer_config
        self.callbacks_config = callbacks_config

    def _build_probe_loaders(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
    ) -> tuple[DataLoader, DataLoader]:
        train_dataset = TensorDataset(train_features, train_labels)
        val_dataset = TensorDataset(val_features, val_labels)

        dataset_on_cuda = any(
            tensor.is_cuda for tensor in (train_features, train_labels, val_features, val_labels)
        )
        effective_num_workers, pin_memory = self._tensor_dataset_loader_runtime(
            dataset_on_cuda=dataset_on_cuda,
            requested_num_workers=self.num_workers,
            feature_device=self.feature_device,
        )
        probe_train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
        )
        probe_val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=effective_num_workers,
            pin_memory=pin_memory,
        )
        return probe_train_loader, probe_val_loader

    def _resolve_loss_fn(self, num_classes: int) -> Optional[Any]:
        loss_fn = None
        if isinstance(self.loss_fn_factory, (dict, DictConfig)):
            if "_target_" in self.loss_fn_factory:
                self.loss_fn_factory = hydra.utils.instantiate(self.loss_fn_factory)

        if self.loss_fn_factory is not None:
            if isinstance(self.loss_fn_factory, (type, functools.partial)) or callable(self.loss_fn_factory):
                try:
                    loss_fn = self.loss_fn_factory(num_classes=num_classes)
                except TypeError:
                    try:
                        loss_fn = self.loss_fn_factory()
                    except TypeError:
                        loss_fn = self.loss_fn_factory
            else:
                loss_fn = self.loss_fn_factory
        return loss_fn

    def _resolve_max_epochs(self) -> int:
        max_epochs = 100
        if self.trainer_config:
            if isinstance(self.trainer_config, (dict, DictConfig)):
                max_epochs = self.trainer_config.get("max_epochs", 100)
            else:
                max_epochs = getattr(self.trainer_config, "max_epochs", 100)
        return max_epochs

    def _build_callbacks(
        self,
        metric_prefix: Optional[str] = None,
        lr: Optional[float] = None,
        include_checkpoint_callbacks: bool = True,
    ) -> list[Any]:
        callbacks = []
        if not self.callbacks_config:
            return callbacks

        for _, cb_conf in self.callbacks_config.items():
            if isinstance(cb_conf, (dict, DictConfig)):
                if "_target_" not in cb_conf:
                    continue
                cb_conf_copy = cb_conf.copy()
                is_checkpoint = cb_conf_copy.get("_target_") in (
                    "pytorch_lightning.callbacks.ModelCheckpoint",
                    "lightning.pytorch.callbacks.ModelCheckpoint",
                )
                if is_checkpoint:
                    if not include_checkpoint_callbacks:
                        continue
                    monitor_key = cb_conf_copy.get("monitor", "val_acc")
                    if metric_prefix is not None:
                        cb_conf_copy["monitor"] = f"{metric_prefix}{monitor_key}"
                    if lr is not None:
                        filename = cb_conf_copy.get("filename", "best")
                        cb_conf_copy["filename"] = f"{filename}_lr={lr}"
                callbacks.append(hydra.utils.instantiate(cb_conf_copy))
            else:
                callbacks.append(cb_conf)
        return callbacks

    def _build_probe_trainer(
        self,
        trainer: L.Trainer,
        callbacks: list[Any],
        max_epochs: int,
    ) -> L.Trainer:
        if self.trainer_config:
            if isinstance(self.trainer_config, (dict, DictConfig)):
                return hydra.utils.instantiate(
                    self.trainer_config,
                    logger=trainer.logger,
                    callbacks=callbacks,
                )
            logger.warning("Trainer config is already instantiated. Callbacks and logger overriding might fail.")
            return self.trainer_config
        return L.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            logger=trainer.logger,
            callbacks=callbacks,
            enable_progress_bar=True,
        )

    @staticmethod
    def _tensor_dataset_loader_runtime(
        dataset_on_cuda: bool,
        requested_num_workers: int,
        feature_device: str,
    ) -> tuple[int, bool]:
        if dataset_on_cuda:
            if requested_num_workers != 0:
                logger.warning(
                    "Linear probe features are on CUDA; forcing num_workers=0 to avoid CUDA "
                    "initialization in DataLoader worker processes."
                )
            return 0, False
        return requested_num_workers, feature_device != "cuda"

    def evaluate(
        self,
        model: nn.Module,
        datamodule: L.LightningDataModule,
        trainer: L.Trainer,
    ) -> Dict[str, float]:

        device = next(model.parameters()).device

        train_features, train_labels = self.get_features(
            model, datamodule, "train", device
        )
        val_features, val_labels = self.get_features(
            model, datamodule, "val", device
        )

        probe_train_loader, probe_val_loader = self._build_probe_loaders(
            train_features, train_labels, val_features, val_labels
        )

        feature_dim = train_features.shape[1]
        num_classes = int(max(train_labels.max(), val_labels.max()) + 1)
        loss_fn = self._resolve_loss_fn(num_classes=num_classes)
        max_epochs = self._resolve_max_epochs()

        all_results: Dict[str, float] = {}
        best_acc = 0.0
        best_lr = self.lr_values[0]

        for lr in self.lr_values:
            logger.info(f"Training Linear Probe with lr={lr}...")

            lr_key = str(lr).replace(".", "_")
            metric_prefix = f"lr{lr_key}/"

            probe_module = LinearProbeModule(
                feature_dim=feature_dim,
                num_classes=num_classes,
                lr=lr,
                weight_decay=self.weight_decay,
                max_epochs=max_epochs,
                b_classifier=self.b_classifier,
                bcos_impl=self.bcos_impl,
                use_bcos_head=self.use_bcos_head,
                loss_fn=loss_fn,
                metric_prefix=metric_prefix,
            )

            callbacks = self._build_callbacks(metric_prefix=metric_prefix, lr=lr)
            probe_trainer = self._build_probe_trainer(
                trainer=trainer, callbacks=callbacks, max_epochs=max_epochs
            )

            start_time = time.time()
            probe_trainer.fit(probe_module, probe_train_loader, probe_val_loader)
            duration = time.time() - start_time

            val_acc_key = f"{metric_prefix}val_acc"
            best_val_acc = 0.0

            checkpoint_callback_found = False
            if probe_trainer.callbacks:
                for cb in probe_trainer.callbacks:
                    if isinstance(cb, ModelCheckpoint):
                        if cb.monitor == val_acc_key or cb.monitor == "val_acc":
                            if cb.best_model_score is not None:
                                best_val_acc = cb.best_model_score.item()
                                checkpoint_callback_found = True
                            break

            if not checkpoint_callback_found:
                val_acc = probe_trainer.callback_metrics.get(val_acc_key)
                if val_acc is None:
                    logger.warning(f"Could not retrieve {val_acc_key} for lr={lr}.")
                    val_acc = 0.0
                else:
                    val_acc = val_acc.item()
                best_val_acc = val_acc

            logger.info(f"Linear Probe (lr={lr}) finished in {duration:.2f}s. max Val Acc: {best_val_acc:.4f}")

            all_results[f"linear_probe/best_acc_lr{lr_key}"] = best_val_acc
            all_results[f"linear_probe/time_lr{lr_key}"] = duration

            if trainer.logger:
                trainer.logger.log_metrics({f"linear_probe/best_acc_lr{lr_key}": best_val_acc})

            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_lr = lr

        all_results["linear_probe/best_acc"] = best_acc
        all_results["linear_probe/best_lr"] = best_lr

        if trainer.logger:
            trainer.logger.log_metrics({
                "linear_probe/best_acc": best_acc,
                "linear_probe/best_lr": best_lr,
            })

        loss_name = loss_fn.__class__.__name__ if loss_fn else "Auto"
        logger.info(f"Best Linear Probe: lr={best_lr}, acc={best_acc:.4f}, loss={loss_name}")

        return all_results


class MultiplexedLinearProbeEvaluator(LinearProbeEvaluator):
    @staticmethod
    def _resolve_run_id(outer_trainer: L.Trainer) -> Optional[str]:
        """Return a stable run identifier (prefer W&B run id)."""
        logger_obj = getattr(outer_trainer, "logger", None)
        if logger_obj is None:
            return None

        experiment = getattr(logger_obj, "experiment", None)
        exp_id = getattr(experiment, "id", None)
        if exp_id:
            return str(exp_id)

        version = getattr(logger_obj, "version", None)
        if version is not None:
            return str(version)

        return None

    @staticmethod
    def _resolve_checkpoint_dir(outer_trainer: L.Trainer) -> Path:
        """Resolve run-scoped checkpoint dir for multiplexed LP snapshots."""
        base_dir: Path | None = None
        for cb in getattr(outer_trainer, "callbacks", []) or []:
            if isinstance(cb, ModelCheckpoint) and cb.dirpath is not None:
                base_dir = Path(cb.dirpath)
                break

        # Fallback: derive from the logger if no ModelCheckpoint is configured.
        if base_dir is None:
            logger_obj = getattr(outer_trainer, "logger", None)
            if logger_obj is not None:
                save_dir = getattr(logger_obj, "save_dir", None)
                if save_dir is not None:
                    project = getattr(logger_obj, "project", None)
                    version = getattr(logger_obj, "version", None)
                    if project and version:
                        base_dir = Path(save_dir) / str(project) / str(version) / "checkpoints"
                    else:
                        base_dir = Path(save_dir) / "linear_probe_checkpoints"

        if base_dir is None:
            default_root_dir = getattr(outer_trainer, "default_root_dir", ".")
            base_dir = Path(default_root_dir) / "checkpoints"

        run_id = MultiplexedLinearProbeEvaluator._resolve_run_id(outer_trainer)
        if run_id:
            # Keep per-run snapshots isolated even when callback dirpath is shared.
            if base_dir.name == run_id:
                return base_dir
            if base_dir.parent.name == run_id and base_dir.name == "checkpoints":
                return base_dir
            return base_dir / run_id
        return base_dir

    def _write_multiplexed_best_checkpoints(
        self,
        probe_module: MultiplexedLinearProbeModule,
        probe_trainer: L.Trainer,
        outer_trainer: L.Trainer,
    ) -> dict[float, Path]:
        checkpoint_dir = self._resolve_checkpoint_dir(outer_trainer)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        written_paths: dict[float, Path] = {}
        for idx, lr in enumerate(self.lr_values):
            checkpoint = probe_module.export_best_checkpoint(idx)
            if checkpoint is None:
                continue
            checkpoint_path = checkpoint_dir / f"best_lr={lr}.ckpt"
            torch.save(checkpoint, checkpoint_path)
            written_paths[float(lr)] = checkpoint_path
        return written_paths

    def evaluate(
        self,
        model: nn.Module,
        datamodule: L.LightningDataModule,
        trainer: L.Trainer,
    ) -> Dict[str, float]:
        device = next(model.parameters()).device

        train_features, train_labels = self.get_features(
            model, datamodule, "train", device
        )
        val_features, val_labels = self.get_features(
            model, datamodule, "val", device
        )

        probe_train_loader, probe_val_loader = self._build_probe_loaders(
            train_features, train_labels, val_features, val_labels
        )

        feature_dim = train_features.shape[1]
        num_classes = int(max(train_labels.max(), val_labels.max()) + 1)
        loss_fn = self._resolve_loss_fn(num_classes=num_classes)
        max_epochs = self._resolve_max_epochs()

        probe_module = MultiplexedLinearProbeModule(
            feature_dim=feature_dim,
            num_classes=num_classes,
            lr_values=self.lr_values,
            weight_decay=self.weight_decay,
            max_epochs=max_epochs,
            b_classifier=self.b_classifier,
            bcos_impl=self.bcos_impl,
            use_bcos_head=self.use_bcos_head,
            loss_fn=loss_fn,
        )

        callbacks = self._build_callbacks(include_checkpoint_callbacks=False)
        probe_trainer = self._build_probe_trainer(
            trainer=trainer, callbacks=callbacks, max_epochs=max_epochs
        )

        start_time = time.time()
        probe_trainer.fit(probe_module, probe_train_loader, probe_val_loader)
        duration = time.time() - start_time
        checkpoint_paths = self._write_multiplexed_best_checkpoints(
            probe_module=probe_module,
            probe_trainer=probe_trainer,
            outer_trainer=trainer,
        )

        all_results: Dict[str, float] = {}
        best_acc = 0.0
        best_lr = self.lr_values[0]

        for idx, lr in enumerate(self.lr_values):
            lr_key = str(lr).replace(".", "_")
            best_val_acc = float(probe_module.best_val_accs[idx].item())
            best_epoch = float(probe_module.best_epoch_indices[idx].item())
            all_results[f"linear_probe/best_acc_lr{lr_key}"] = best_val_acc
            all_results[f"linear_probe/best_epoch_lr{lr_key}"] = best_epoch
            all_results[f"linear_probe/time_lr{lr_key}"] = duration

            if trainer.logger:
                trainer.logger.log_metrics({
                    f"linear_probe/best_acc_lr{lr_key}": best_val_acc,
                    f"linear_probe/best_epoch_lr{lr_key}": best_epoch,
                })

            checkpoint_path = checkpoint_paths.get(float(lr))
            if checkpoint_path is not None:
                logger.info(
                    "Saved multiplexed LP checkpoint for lr={} at {} (best_epoch={}, best_acc={:.4f})",
                    lr,
                    checkpoint_path,
                    int(best_epoch),
                    best_val_acc,
                )

            if best_val_acc > best_acc:
                best_acc = best_val_acc
                best_lr = lr

        all_results["linear_probe/best_acc"] = best_acc
        all_results["linear_probe/best_lr"] = best_lr
        all_results["linear_probe/time_multiplexed"] = duration

        if trainer.logger:
            trainer.logger.log_metrics({
                "linear_probe/best_acc": best_acc,
                "linear_probe/best_lr": best_lr,
                "linear_probe/time_multiplexed": duration,
            })

        loss_name = loss_fn.__class__.__name__ if loss_fn else "Auto"
        logger.info(
            "Best Multiplexed Linear Probe: lr={}, acc={:.4f}, loss={}, total_time={:.2f}s",
            best_lr,
            best_acc,
            loss_name,
            duration,
        )

        return all_results
