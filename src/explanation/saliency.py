"""
Saliency explainer using Captum.
"""
from typing import Optional
import torch.nn as nn
from torch import Tensor
from captum.attr import Saliency

from .captum_base import CaptumExplainer
from .abstract import ExplanationOutput
from beartype import beartype
from jaxtyping import jaxtyped


class SaliencyExplainer(CaptumExplainer):
    """Saliency maps: gradient of output w.r.t. input."""
    
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
        def forward_fn(x):
            out = model(x)
            if hasattr(out, 'logits'):
                return out.logits
            elif hasattr(out, 'pooler_output'):
                return out.pooler_output
            elif isinstance(out, tuple):
                return out[0]
            return out
        
        # Compute saliency
        saliency = Saliency(forward_fn)
        attributions = saliency.attribute(input_tensor, target=target)
        
        return self._attributions_to_output(input_tensor, attributions, target, **kwargs)
