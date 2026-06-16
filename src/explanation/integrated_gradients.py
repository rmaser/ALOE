"""
Integrated Gradients explainer using Captum.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from captum.attr import IntegratedGradients

from .captum_base import CaptumExplainer
from .abstract import ExplanationOutput
from beartype import beartype
from jaxtyping import jaxtyped


class IntegratedGradientsExplainer(CaptumExplainer):
    """Integrated Gradients: path integral of gradients from baseline to input."""
    
    def __init__(self, n_steps: int = 50, internal_batch_size: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
    
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
        def forward_fn(x):
            out = model(x)
            if hasattr(out, 'logits'):
                return out.logits
            elif hasattr(out, 'pooler_output'):
                return out.pooler_output
            elif isinstance(out, tuple):
                return out[0]
            return out
        
        # Compute integrated gradients
        ig = IntegratedGradients(forward_fn)
        attributions = ig.attribute(
            input_tensor, 
            baselines=baselines, 
            target=target,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size
        )
        
        return self._attributions_to_output(input_tensor, attributions, target, **kwargs)
