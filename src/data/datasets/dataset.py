from typing import Any
from src.data.datasets.config import GridPGDatasetConfig
from typing import Iterable
from torch.utils.data import Dataset
from .config import DatasetConfig
from loguru import logger
from datasets import load_dataset
from datasets import Dataset as HFDataset
import torch 
import numpy as np
import warnings
import pytorch_lightning as plt

from beartype import beartype

@beartype
class DatasetCreator(Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def num_classes(self):
        raise NotImplementedError("Subclasses must implement this method")

    def load_dataset(self) -> dict[str, Dataset]:
        raise NotImplementedError("Subclasses must implement this method")

class TransformDataset(Dataset):
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    @beartype
    def __getitem__(self, index: int) -> Any:
        item = self.dataset[index]
        if self.transform:
            item = self.transform(item)
        return item
    
    @property
    def class_names(self):
        if hasattr(self.dataset, "class_names"):
            return self.dataset.class_names
        return None

class HFToTorchDataset(Dataset):
    def __init__(self, hf_dataset: HFDataset):
        self.hf_dataset = hf_dataset
        
    def __len__(self):
        return len(self.hf_dataset)
        
    def __getitem__(self, index):
        return self.hf_dataset[index]
    
    @property
    def class_names(self):
        if "label" in self.hf_dataset.features:
            return self.hf_dataset.features["label"].names
        elif "fine_label" in self.hf_dataset.features:
             return self.hf_dataset.features["fine_label"].names
        elif "coarse_label" in self.hf_dataset.features:
             return self.hf_dataset.features["coarse_label"].names
        return None

class HFDatasetCreator(DatasetCreator):
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    
    def load_dataset(self) -> dict[str, Dataset]:
        # Seed everything:
        plt.seed_everything(self.config.seed)
        
        dataset_name = self.config.dataset_name

        train_set = self.load_hf_dataset(dataset_name, split="train")
        val_set = None
        for split_name in ("validation", "val", "test"):
            try:
                val_set = self.load_hf_dataset(dataset_name, split=split_name)
                break
            except Exception as e:
                logger.info(f"Split {split_name!r} not found for {dataset_name}, trying next. ({e})")
        if val_set is None:
            assert train_set is not None, "Train set not found for {dataset_name}."
            fraction = min(self.config.val_ceiling, max(self.config.val_floor, int(self.config.val_fraction * len(train_set))))
            logger.info(
                f"No validation/val/test split for {dataset_name}. Creating validation holdout from training data (n={fraction})."
            )
            train_set, val_set = self._split_train_for_validation(
                train_set=train_set,
                fraction=fraction,
            )
        
        if self.config.train_fraction != 1.0:
            np.random.seed(self.config.seed)

            # Select random number of samples from the train set
            train_set = train_set.select(np.random.choice(len(train_set), int(self.config.train_fraction * len(train_set)), replace=False))

        logger.info(f"Dataset: {self.config.dataset_name}, Train set size: {len(train_set)}, Val set size: {len(val_set)}")
        return {"train": HFToTorchDataset(train_set), "validation": HFToTorchDataset(val_set)}
    
    def load_hf_dataset(self, dataset_name, split) -> HFDataset:
        # Define transform that supports BOTH batched (lists) and single-example access

        dataset = load_dataset(dataset_name, split=split, streaming=False, cache_dir=str(self.config.dataset_cache_path))
        assert isinstance(dataset, HFDataset)
        
        if dataset_name == 'bitmind/caltech-101' and 'label' not in dataset.column_names:
            dataset = self._add_caltech101_labels(dataset)
        return (dataset)
    
    def _split_train_for_validation(self, train_set: HFDataset, fraction: int) -> tuple[HFDataset, HFDataset]:
        """Create a validation split from the training data when no validation/test split exists.
        Stratifies by label if possible.
        """
        raw_train = train_set
        if self.config.dataset_name == 'bitmind/caltech-101' and 'label' not in raw_train.column_names:
            raw_train = self._add_caltech101_labels(raw_train)
        # Simple random split (no stratification needed)
        
        split = raw_train.train_test_split(test_size=fraction, seed=self.config.seed)

        train_set = split["train"]
        val_set = split["test"]
        assert isinstance(train_set, HFDataset)
        assert isinstance(val_set, HFDataset)

        logger.info(f"Created train/val split from training data: train={len(train_set)}, val={len(val_set)} (fraction={fraction})")
        return train_set, val_set
    
    def _add_caltech101_labels(self, dataset):
        """Derive labels for Caltech101 from the first path segment of 'filename'.
        Handles BACKGROUND_Google (drops it by default to get 101 classes, matching common practice).
        Set experiment.config['caltech_include_background']=True to keep it (will then expect 102 classes).
        Performs integrity checks (contiguous labels, expected class count).
        """
        if 'filename' not in dataset.column_names:
            raise ValueError("Expected 'filename' column for Caltech101 label derivation.")
        include_background = getattr(self.config, 'caltech_include_background', False)
        filenames = dataset['filename']
        raw_classes = sorted({fn.split('/')[0] for fn in filenames})
        if not include_background and 'BACKGROUND_Google' in raw_classes:
            # Filter out background samples
            dataset = dataset.filter(lambda ex: not ex['filename'].startswith('BACKGROUND_Google/'))
            filenames = dataset['filename']
            classes = sorted({fn.split('/')[0] for fn in filenames})
            removed = 'BACKGROUND_Google'
        else:
            classes = raw_classes
            removed = None
        expected_classes = 102 if include_background else 101
        if len(classes) != expected_classes:
            sample = classes[:10]
            raise ValueError(
                f"Caltech101 label extraction produced {len(classes)} classes, expected {expected_classes}. Sample: {sample}. "
                f"include_background={include_background}, removed={removed}."
            )
        class_to_idx = {c: i for i, c in enumerate(classes)}
        logger.info(f"Caltech101: inferred {len(classes)} classes (expected {expected_classes}). Background kept={include_background}. First 5: {classes[:5]}")
        def add_label(example):
            example['label'] = class_to_idx[example['filename'].split('/')[0]]
            return example
        dataset = dataset.map(add_label)
        labels = set(dataset['label'])
        if labels != set(range(len(classes))):
            raise ValueError(f"Caltech101 labels not contiguous after mapping. Got min {min(labels)}, max {max(labels)}, count {len(labels)}.")
        max_label = max(labels)
        if max_label != expected_classes - 1:
            raise ValueError(f"Caltech101 max label {max_label} inconsistent with expected {expected_classes-1}.")
        logger.info(f"Caltech101 label integrity verified: 0..{max_label} (background kept={include_background}).")
        return dataset


class GridPGDataset(Dataset):
    def __init__(self, dataset: Iterable, gridpg_dataset_config: GridPGDatasetConfig) -> None:
        super().__init__()
        self.scale = gridpg_dataset_config.scale
        self.dataset: Dataset = dataset
        self.cache = {}
        self.num_samples = gridpg_dataset_config.num_samples
        self.rng = np.random.default_rng(seed=gridpg_dataset_config.seed)

    def update_dataset(self, new_dataset):
        """Updates the underlying dataset."""
        self.dataset = new_dataset

    def __getitem__(self, index) -> Any:
        # print(f"Length of dataset: {len(self.dataset)}")
        # print(f"Index was: {index}")
        # Convert numpy int64 to Python int for HuggingFace datasets
        index = int(index)
        num_items = self.scale ** 2

        images, labels = self.check_cache(index)
        

        if images is None:
            images = []
            labels = []
            
            first_item = self.dataset.__getitem__(index)
            # Handle both dict and tuple formats
            if isinstance(first_item, dict):
                first_image = first_item["image"]
                first_label = first_item.get("label", first_item.get("variant"))
            else:
                first_image = first_item[0]
                first_label = first_item[1]
            
            # print(f"GridPG item has shape: ", first_image.shape)
            images.append(first_image)
            labels.append(first_label)

            # Generate a shuffled array of indices excluding the current index
            available_indices = np.arange(len(self.dataset))
            available_indices = available_indices[available_indices != index]
            self.rng.shuffle(available_indices)

            # Iterate through the shuffled indices to find matching samples
            for idx in available_indices:
                if len(images) == num_items:
                    break # Exit if enough images are found

                # Convert numpy int64 to Python int for HuggingFace datasets
                idx = int(idx)
                item = self.dataset.__getitem__(idx)
                # Handle both dict and tuple formats
                if isinstance(item, dict):
                    image = item["image"]
                    label = item.get("label", first_item.get("variant"))
                else:
                    image = item[0]
                    label = item[1]
                
                if label not in labels:
                    images.append(image)
                    labels.append(label)
            
            self.store_in_cache(index, images, labels)

        # Check if enough unique samples were found
        if len(images) < num_items:
            raise Exception()
            # Return dummy tensors with correct shape
            # Get the shape of the original image
            original_image_shape = self.dataset.__getitem__(index)[0].shape
            num_channels = original_image_shape[0]
            original_height = original_image_shape[1]
            original_width = original_image_shape[2]


            # Create dummy image tensor
            dummy_grid = torch.zeros(num_channels, original_height, original_width , dtype=torch.float32)

            # Create dummy labels tensor
            dummy_labels = torch.zeros(num_items, dtype=torch.long) # Assuming labels are long

            warnings.warn("Grid PG dataset returned dummy image.")
            return dummy_grid, dummy_labels

        grid = self.grid_transform(images, scale=self.scale)        
        labels = torch.tensor(labels)
        return grid, labels

    def check_cache(self, index):
        if index in self.cache:
            return self.cache[index]
        return None, None
    
    def store_in_cache(self, index, images, labels):
        self.cache[index] = (images, labels)
    
    def __len__(self):
        return len(self.dataset)
    
    @staticmethod
    def grid_transform(image_list: list, scale):
        '''
        Arguments:
            image_list(list): list of images for creating a grid. Elements should have shape [C, H, W]
        
        Returns:
            fused grid-image
        '''
        assert (scale ** 2) == len(image_list)
        shape = image_list[0].shape
        assert shape[0] == 3 or shape[0] == 6, "Image does not have 3 channels: might not be an RGB image"

        index = 0
        stacked = []
        for i in range(scale):
            index = i*scale
            stacked.append(torch.concatenate(image_list[index:index+scale], axis = 2))
        stacked = torch.concatenate(stacked, axis = 1)
        return stacked

