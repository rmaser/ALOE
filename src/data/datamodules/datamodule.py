import random
from typing import Optional

import beartype
import PIL.Image
import pytorch_lightning as pl
import torch
from loguru import logger
from PIL import ImageFile
from PIL.Image import Exif
from torch.utils.data import ConcatDataset, Dataset
from torch.utils.data.dataloader import DataLoader

from src.data.datamodules.config import DataModuleConfig
from src.data.datasets.dataset import GridPGDataset, HFDatasetCreator, TransformDataset
from src.data.transforms.transforms import HFTransform
from src.eval.zero_shot_templates import OPENAI_IMAGENET_TEMPLATES


@beartype.beartype
class BcosDataModule(pl.LightningDataModule):
    def __init__(self, datamodule_config: DataModuleConfig, **kwargs):
        super().__init__()
        self.datamodule_config = datamodule_config
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        self.pg_val_set: Optional[GridPGDataset] = None

        ImageFile.LOAD_TRUNCATED_IMAGES = True  # type: ignore
        PIL.Image.Image.getexif = lambda self: Exif()

    def setup(self, stage: Optional[str] = None):
        if self.val_dataset is None and self.train_dataset is None:
            self.load_datasets()

    @property
    def class_names(self):
        return self.train_dataset.class_names

    @property
    def templates(self):
        return OPENAI_IMAGENET_TEMPLATES

    def load_datasets(self):
        train_datasets: list[Dataset] = []
        val_datasets: list[Dataset] = []
        test_datasets: list[Dataset] = []

        # Get transforms
        train_transform, val_transform = self.get_transforms().get_transform_wrappers()

        for dataset_config in self.datamodule_config.datasets:
            if dataset_config.dataset_source == "hf":
                dataset = HFDatasetCreator(dataset_config)
                datasets = dataset.load_dataset()

                # datasets["train"].hf_dataset.set_transform(train_transform)
                # datasets["validation"].hf_dataset.set_transform(val_transform)

                actual_train_transform = (
                    val_transform
                    if self.datamodule_config.force_val_transforms_in_train
                    else train_transform
                )
                if self.datamodule_config.force_val_transforms_in_train:
                    logger.info("Forcing validation transforms for training set")

                train_ds = TransformDataset(datasets["train"], actual_train_transform)
                val_ds = TransformDataset(datasets["validation"], val_transform)
                test_ds = TransformDataset(
                    datasets.get("test", datasets["validation"]), val_transform
                )

                train_datasets.append(train_ds)
                val_datasets.append(val_ds)
                test_datasets.append(test_ds)

        if len(train_datasets) == 0 or len(val_datasets) == 0:
            raise ValueError("No datasets found")

        self.train_dataset = (
            train_datasets[0]
            if len(train_datasets) == 1
            else ConcatDataset(train_datasets)
        )
        self.val_dataset = (
            val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
        )
        self.test_dataset = (
            test_datasets[0]
            if len(test_datasets) == 1
            else ConcatDataset(test_datasets)
        )

        # Select random subsets of the datasets if specified
        if self.datamodule_config.limit_train_set_size is not None:
            random.seed(self.datamodule_config.seed)
            self.train_dataset = self.train_dataset.select(
                random.sample(
                    range(len(self.train_dataset)),
                    int(self.datamodule_config.limit_train_set_size),
                )
            )
        if self.datamodule_config.limit_val_set_size is not None:
            random.seed(self.datamodule_config.seed)
            self.val_dataset = self.val_dataset.select(
                random.sample(
                    range(len(self.val_dataset)),
                    int(self.datamodule_config.limit_val_set_size),
                )
            )

        # Initialize grid pg val set if config exists
        if self.datamodule_config.grid_dataset_config is not None:
            self.pg_val_set = GridPGDataset(
                self.val_dataset, self.datamodule_config.grid_dataset_config
            )

    def get_transforms(self):
        if self.datamodule_config.transform_config.model_source == "hf":
            return HFTransform(self.datamodule_config.transform_config)
        else:
            raise ValueError(
                f"Unknown transform source: {self.datamodule_config.transform_config.model_source}"
            )

    @beartype.beartype
    def train_dataloader(self) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(self.datamodule_config.seed)
        loader_kwargs = self.datamodule_config.dataloader_config.model_dump(
            exclude={"seed"}
        )
        if loader_kwargs.get("num_workers", 0) == 0:
            loader_kwargs["prefetch_factor"] = None
            loader_kwargs["persistent_workers"] = False

        return DataLoader(self.train_dataset, **loader_kwargs)

    @beartype.beartype
    def val_dataloader(self) -> DataLoader:
        loader_kwargs = self.datamodule_config.dataloader_config.model_dump(
            exclude={"seed"}
        )
        if loader_kwargs.get("num_workers", 0) == 0:
            loader_kwargs["prefetch_factor"] = None
            loader_kwargs["persistent_workers"] = False
        loader_kwargs["shuffle"] = False
        return DataLoader(self.val_dataset, **loader_kwargs)

    @beartype.beartype
    def grid_pg_val_dataloader(self) -> DataLoader:
        loader_kwargs = self.datamodule_config.dataloader_config.model_dump(
            exclude={"seed"}
        )
        if loader_kwargs.get("num_workers", 0) == 0:
            loader_kwargs["prefetch_factor"] = None
            loader_kwargs["persistent_workers"] = False
        # loader_kwargs["shuffle"] = True

        if self.pg_val_set is None:
            raise RuntimeError(
                "pg_val_set is not initialized. Ensure setup() was called with grid_dataset_config."
            )
        return DataLoader(self.pg_val_set, **loader_kwargs)

    @beartype.beartype
    def test_dataloader(self) -> DataLoader:
        loader_kwargs = self.datamodule_config.dataloader_config.model_dump(
            exclude={"seed"}
        )
        if loader_kwargs.get("num_workers", 0) == 0:
            loader_kwargs["prefetch_factor"] = None
            loader_kwargs["persistent_workers"] = False

        if self.test_dataset is None:
            raise RuntimeError(
                "test_dataset is not initialized. Ensure setup() was called."
            )
        return DataLoader(self.test_dataset, **loader_kwargs)
