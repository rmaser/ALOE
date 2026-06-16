from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from .abstract import BaseExplainer, ExplanationOutput
from .utils import ModelOutputWrapper, gradient_to_image
from beartype import beartype
from jaxtyping import jaxtyped

class InputXGradientExplainer(BaseExplainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    @jaxtyped(typechecker=beartype)
    def explain(
        self, 
        model: nn.Module, 
        input_tensor: Tensor, 
        target: Optional[Tensor] = None, 
        return_explanation: bool = True,
        **kwargs
    ) -> ExplanationOutput:
        
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        model.zero_grad()
        
        # Forward pass
        outputs = ModelOutputWrapper(model)(input_tensor)
        
        # Get prediction if target is None
        if target is None:
            target = outputs.argmax(dim=1)
            
        # Gather logits for target class
        if target.dim() == 0:
            target = target.unsqueeze(0)
            
        # Ensure target is on same device
        target = target.to(device=outputs.device, dtype=torch.long)
            
        to_be_explained_logit = torch.gather(outputs, 1, target.unsqueeze(1))
        
        # Backward pass
        to_be_explained_logit.sum().backward(inputs=[input_tensor])
        
        # Compute attribution: Input * Gradient
        attribution = input_tensor * input_tensor.grad
        
        # Generate visualization
        # Note: gradient_to_image expects linear_mapping (gradient) and image
        # Force to_numpy=False to ensure we get Tensors as expected by ExplanationOutput
        kwargs["to_numpy"] = False
        explanation = (
            gradient_to_image(
                input_tensor.detach(),
                input_tensor.grad.detach(),
                **self.kwargs,
                **kwargs
            )
            if return_explanation
            else None
        )
        
        grad = input_tensor.grad
        if grad is None:
            raise RuntimeError("Input tensor gradient is None after backward pass.")

        return ExplanationOutput(
            contribution_map=attribution.sum(1, keepdim=True).detach(),
            explanation=explanation,
            prediction=target.detach(),
            attributions=attribution.detach(),
            dynamic_linear_weights=grad.detach(),
            explained_class_idx=target.detach()
        )
