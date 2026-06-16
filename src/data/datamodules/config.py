
from src.data.datasets.config import GridPGDatasetConfig
from src.data.transforms.config import TransformConfig
from src.data.datasets.config import DatasetConfig
from pydantic import BaseModel, Field
from typing import Literal, Optional
from src.utils import env
from pathlib import Path

class DataloaderConfig(BaseModel):
    # Accept powers of 2
    batch_size: Literal[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] = 64
    num_workers: int = Field(0, ge=0, le=16)
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = Field(2, ge=1, le=5)
    drop_last: bool = True
    shuffle: bool = True

class DataModuleConfig(BaseModel):
    datasets: list[DatasetConfig] = Field(min_length=1) # make sure that at least one dataset is provided
    dataset_cache_path: Path = env.get_data_path()  # Default to environment variable
    val_fraction: float = Field(0.1, ge=0.0, le=0.99)
    val_ceiling: int = Field(10000, le=100000)
    val_floor: int = Field(1000, ge=1000)
    limit_train_set_size: Optional[int] = None
    limit_val_set_size: Optional[int] = None
    seed: int = env.get_seed()  
    dataloader_config: DataloaderConfig = DataloaderConfig()
    transform_config: TransformConfig
    grid_dataset_config: Optional[GridPGDatasetConfig] = GridPGDatasetConfig()
    force_val_transforms_in_train: bool = False
  

    
