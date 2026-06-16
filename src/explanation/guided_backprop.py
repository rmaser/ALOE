"""
Guided Backpropagation explainer using Captum.
"""
from typing import Optional
import torch.nn as nn
from torch import Tensor
from captum.attr import GuidedBackprop

from .captum_base import CaptumExplainer
from .abstract import ExplanationOutput
from .utils import ModelOutputWrapper
from beartype import beartype
from jaxtyping import jaxtyped


class GuidedBackpropExplainer(CaptumExplainer):
    """Guided Backpropagation: gradient with ReLU masking."""
    
    @jaxtyped(typechecker=beartype)
    def explain(
        self, 
        model: nn.Module, 
        input_tensor: Tensor, 
        target: Optional[Tensor] = None, 
        **kwargs
    ) -> ExplanationOutput:
        # Ensure input requires grad
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        
        # Get target
        target = self._get_target(model, input_tensor, target)
        
        # Create wrapper for model output
        forward_model = ModelOutputWrapper(model)
        
        # Compute guided backprop
        gbp = GuidedBackprop(forward_model)
        attributions = gbp.attribute(input_tensor, target=target)
        
        return self._attributions_to_output(input_tensor, attributions, target, **kwargs)
