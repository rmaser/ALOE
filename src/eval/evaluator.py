import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import pytorch_lightning as L
from loguru import logger
from typing import Tuple, Optional, Dict, cast
from tqdm import tqdm  # type: ignore[import-untyped]
from transformers.modeling_outputs import BaseModelOutputWithPooling

from src.utils.env import get_feature_cache_path


def _feature_cache_model_name(model: nn.Module) -> str:
    """
    Stable subdirectory name for on-disk feature caches.

    ``scripts/eval.py`` attaches the full Hydra ``cfg`` as ``model.cfg``; that has
    ``model_factory.target_model_config.name`` (Hub id), not
    ``student_model_factory_config`` (Lightning training only).  Without this,
    every eval run fell back to ``unknown_model`` and reused wrong features.
    """
    if not hasattr(model, "cfg") or model.cfg is None:
        return "unknown_model"

    cfg = model.cfg
    name: Optional[str] = None

    try:
        from omegaconf import Container, OmegaConf

        cfg_container = cast(Container, cfg)
        if OmegaConf.is_config(cfg_container):
            name = OmegaConf.select(cfg_container, "student_model_factory_config.target_model_config.name")
            if name is None:
                name = OmegaConf.select(cfg_container, "model_factory.target_model_config.name")
    except Exception:
        pass

    if name is None and hasattr(cfg, "student_model_factory_config"):
        smfc = cfg.student_model_factory_config
        if smfc is not None and hasattr(smfc, "target_model_config"):
            tmc = smfc.target_model_config
            if hasattr(tmc, "name"):
                raw_name = tmc.name
                if isinstance(raw_name, str):
                    name = raw_name

    if not name:
        return "unknown_model"
    return str(name).replace("/", "_").replace(":", "_")


class Evaluator:
    """
    Base class for evaluators that require feature extraction.
    """

    def __init__(
        self,
        feature_cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        feature_device: str = "cpu",
    ):
        cache_root = Path(feature_cache_dir) if feature_cache_dir else get_feature_cache_path()
        cache_root.mkdir(parents=True, exist_ok=True)
        self.feature_cache_dir = os.fspath(cache_root)
        self.force_recompute = force_recompute
        self.feature_device = feature_device

    def _move_cached_features(self, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        if self.feature_device == "cuda":
            return tensor.to(device, non_blocking=True)
        return tensor.cpu()

    @staticmethod
    def _feature_extraction_autocast(device: torch.device):
        if device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def get_features(
        self,
        model: nn.Module,
        datamodule: L.LightningDataModule,
        split: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts features and labels from a datamodule using a model.
        Caches features to disk to avoid re-computation.
        """
        model_name = _feature_cache_model_name(model)
        if model_name == "unknown_model":
            logger.warning(
                "Feature cache key is 'unknown_model' — could not read "
                "model_factory.target_model_config.name or student_model_factory_config "
                "from model.cfg; caches may collide across runs."
            )
        else:
            logger.debug(f"Feature cache model key: {model_name}")

        dataset_name = "unknown_dataset"
        if hasattr(datamodule, "datamodule_config") and len(datamodule.datamodule_config.datasets) > 0:
            dataset_name = datamodule.datamodule_config.datasets[0].dataset_name.replace("/", "_")

        cache_dir = os.path.join(self.feature_cache_dir, model_name, dataset_name)
        os.makedirs(cache_dir, exist_ok=True)

        features_path = os.path.join(cache_dir, f"{split}_features.pt")
        labels_path = os.path.join(cache_dir, f"{split}_labels.pt")

        if not self.force_recompute and os.path.exists(features_path) and os.path.exists(labels_path):
            logger.info(f"Loading cached {split} features from {features_path}")
            features = torch.load(features_path)
            labels = torch.load(labels_path)
            return self._move_cached_features(features, device), self._move_cached_features(labels, device)

        logger.info(f"Extracting {split} features...")

        if split == "train":
            if not hasattr(datamodule, "train_dataloader"):
                datamodule.setup("fit")
            loader = datamodule.train_dataloader()
        else:
            if not hasattr(datamodule, "val_dataloader"):
                datamodule.setup("fit")
            loader = datamodule.val_dataloader()
            if isinstance(loader, list):
                loader = loader[0]

        model.eval()
        model.to(device)

        all_features = []
        all_labels = []

        keep_features_on_cuda = self.feature_device == "cuda" and device.type == "cuda"

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Extracting {split}"):
                images = None
                labels = None

                if isinstance(batch, (list, tuple)):
                    images, labels = batch[0], batch[1]
                elif isinstance(batch, dict):
                    images = batch.get("image")
                    if images is None:
                        images = batch.get("img")

                    labels = batch.get("label")
                    if labels is None:
                        labels = batch.get("target")

                if images is None or labels is None:
                    logger.warning("Could not unpack batch. Skipping.")
                    continue

                images = images.to(device, non_blocking=True)
                with self._feature_extraction_autocast(device):
                    out = model(images)

                assert isinstance(out, BaseModelOutputWithPooling)

                features = out.pooler_output

                if features is not None:
                    if keep_features_on_cuda:
                        all_features.append(features.float())
                        all_labels.append(labels.to(device, non_blocking=True))
                    else:
                        all_features.append(features.float().cpu())
                        all_labels.append(labels.cpu())

        features = torch.cat(all_features)
        labels = torch.cat(all_labels)

        # Keep in-memory tensors on the target feature device (CUDA mode), but
        # write CPU copies to disk so cache reload remains device-agnostic.
        features_cpu = features.detach().cpu()
        labels_cpu = labels.detach().cpu()

        logger.info(f"Saving {split} features to {features_path}")
        torch.save(features_cpu, features_path)
        torch.save(labels_cpu, labels_path)

        if keep_features_on_cuda:
            return features, labels
        return self._move_cached_features(features_cpu, device), self._move_cached_features(labels_cpu, device)

    def evaluate(self, *args, **kwargs) -> Dict[str, float]:
        raise NotImplementedError
