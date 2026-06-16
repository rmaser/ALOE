from abc import ABC, abstractmethod
from typing import Optional
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, jaxtyped
from beartype import beartype

from dataclasses import dataclass

@jaxtyped(typechecker=beartype)
@dataclass
class ExplanationOutput:
    contribution_map: Float[Tensor, "batch 1 height width"]
    explanation: Float[Tensor, "batch height width 4"]
    prediction: Optional[Tensor] = None
    attributions: Optional[Tensor] = None
    dynamic_linear_weights: Optional[Tensor] = None
    explained_class_idx: Optional[Tensor] = None
    map_size: Optional[int] = 1
    
    def __getitem__(self, key):
        return getattr(self, key)

class BaseExplainer(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    @jaxtyped(typechecker=beartype)
    def explain(
        self, 
        model: nn.Module, 
        input_tensor: Tensor, 
        target: Optional[Tensor] = None, 
        **kwargs
    ) -> ExplanationOutput:
        """
        Generate explanation for the input tensor.
        
        Args:
            model: The model to explain.
            input_tensor: Input tensor [B, C, H, W].
            target: Target class indices [B] or None.
            
        Returns:
            Dictionary containing 'contribution_map', 'explanation', etc.
        """
        pass
