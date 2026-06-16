"""
GradientShap explainer using Captum.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from captum.attr import GradientShap

from .captum_base import CaptumExplainer
from .abstract import ExplanationOutput
from beartype import beartype
from jaxtyping import jaxtyped


class GradientShapExplainer(CaptumExplainer):
    """GradientShap: SHAP values computed using gradient-based approximation."""
    
    def __init__(self, n_samples: int = 50, stdevs: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.stdevs = stdevs
    
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
        
        # Default baseline distribution: range from 0 to input
        if baselines is None:
            baselines = torch.cat([
                torch.zeros_like(input_tensor),
                input_tensor.clone()
            ], dim=0)
        
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
        
        # Compute gradient shap
        gs = GradientShap(forward_fn)
        attributions = gs.attribute(
            input_tensor, 
            baselines=baselines, 
            target=target,
            n_samples=self.n_samples,
            stdevs=self.stdevs
        )
        
        return self._attributions_to_output(input_tensor, attributions, target, **kwargs)
