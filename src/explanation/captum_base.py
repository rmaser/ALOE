"""
Captum-based explainer base class with shared logic.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .abstract import BaseExplainer, ExplanationOutput
from .utils import gradient_to_image


class CaptumExplainer(BaseExplainer):
    """Base class for Captum-based explainers."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs
    
    def _get_target(self, model: nn.Module, input_tensor: Tensor, target: Optional[Tensor]) -> Tensor:
        """Get target class indices, computing from model output if not provided."""
        if target is not None:
            if target.dim() == 0:
                target = target.unsqueeze(0)
            return target.to(input_tensor.device)
        
        # Forward pass to get predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits
            elif hasattr(outputs, 'pooler_output'):
                outputs = outputs.pooler_output
            elif isinstance(outputs, tuple):
                outputs = outputs[0]
            return outputs.argmax(dim=1)
    
    def _attributions_to_output(
        self, 
        input_tensor: Tensor, 
        attributions: Tensor, 
        target: Tensor,
        return_explanation: bool = True,
        **kwargs
    ) -> ExplanationOutput:
        """Convert attributions to ExplanationOutput format."""
        # Merge kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs["to_numpy"] = False
        
        # Generate visualization
        explanation = (
            gradient_to_image(
                input_tensor.detach(), 
                attributions.detach(),
                **merged_kwargs
            )
            if return_explanation
            else None
        )
        
        return ExplanationOutput(
            contribution_map=attributions.sum(1, keepdim=True).detach(),
            explanation=explanation,
            prediction=target.detach(),
            attributions=attributions.detach(),
            dynamic_linear_weights=attributions.detach(),
            explained_class_idx=target.detach()
        )
