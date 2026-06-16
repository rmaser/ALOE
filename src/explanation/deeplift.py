"""
DeepLift explainer using Captum.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from captum.attr import DeepLift

from .captum_base import CaptumExplainer
from .abstract import ExplanationOutput
from .utils import ModelOutputWrapper
from beartype import beartype
from jaxtyping import jaxtyped


class DeepLiftExplainer(CaptumExplainer):
    """DeepLift: assigns contribution scores by comparing activations to reference."""
    
    @jaxtyped(typechecker=beartype)
    def explain(
        self, 
        model: nn.Module, 
        input_tensor: Tensor, 
        target: Optional[Tensor] = None,
        baselines: Optional[Tensor] = None,
        **kwargs
    ) -> ExplanationOutput:
        # Ensure input requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Get target
        target = self._get_target(model, input_tensor, target)
        
        # Default baseline: zero tensor
        if baselines is None:
            baselines = torch.zeros_like(input_tensor)
        
        # Create wrapper for model output
        forward_model = ModelOutputWrapper(model)
        
        # Compute deeplift
        dl = DeepLift(forward_model)
        attributions = dl.attribute(
            input_tensor, 
            baselines=baselines, 
            target=target
        )
        
        return self._attributions_to_output(input_tensor, attributions, target, **kwargs)
