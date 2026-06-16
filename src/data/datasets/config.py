from pydantic import BaseModel, Field
from typing import Optional
from src.utils import env
from pathlib import Path

class DatasetConfig(BaseModel):
    dataset_source: str
    dataset_name: str
    dataset_cache_path: Path = env.get_data_path()  # Default to environment variable
    train_fraction: float = Field(1.0, ge=0.0, le=1.0)
    val_fraction: float = Field(0.1, ge=0.0, le=0.99)
    val_ceiling: int = Field(10000, le=100000)
    val_floor: int = Field(1000, ge=1000)
    seed: int = env.get_seed()  
    num_classes: Optional[int] = None

    
class GridPGDatasetConfig(BaseModel):
    scale: int = 2
    num_samples: int = 1024
    seed: int = env.get_seed()
    