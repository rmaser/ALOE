from pydantic import BaseModel, ConfigDict
from torch import nn
from typing import Callable, Any
import loguru

class GridProcessorConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model: nn.Module
    explain_fn: Callable
    grid_pg_scale: int
    logger: Any = loguru.logger
    confidence_threshold: float = 0.95
    map_size: int = 2
    prediction_fn: Callable = None

