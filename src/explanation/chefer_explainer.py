# Implemented by Gemini 3 Pro based on LeGrad explainer
"""Chefer CAM Explainer (Gradient-weighted Attention Rollout)."""
from typing import Optional
import copy
import torch
import torch.nn as nn
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped

from .abstract import BaseExplainer, ExplanationOutput
from .utils import gradient_to_image
from .legrad.wrapper import LeWrapper

class CheferExplainer(BaseExplainer):
    """Chefer CAM (gradient-weighted attention rollout) for classifier vision models."""
    
    def __init__(self, starting_layer: int = 0, **kwargs):
        """
        Args:
            starting_layer: Which layer to use for relevance. 0 = all layers (official Chefer).
        """
        super().__init__(**kwargs)
        self.starting_layer = starting_layer
        self.kwargs = kwargs
        self._patched_model = None
        self._model_id = None
    
    def _get_patched_model(self, model: nn.Module) -> nn.Module:
        """Get or create a monkey-patched model for Chefer CAM."""
        model_id = id(model)
        
        # Return cached model if available and still valid
        if self._patched_model is not None and self._model_id == model_id:
            return self._patched_model
            
        patched = copy.deepcopy(model)
        
        # Disable parameter grads to reduce memory
        for p in patched.parameters():
            p.requires_grad = False
        
        patched = LeWrapper(patched, layer_index=self.starting_layer)        
        patched.eval()
        self._patched_model = patched
        self._model_id = model_id
        
        return patched


    def _get_target(self, model: nn.Module, input_tensor: Tensor, target: Optional[Tensor]) -> Tensor:
        """Get target class indices."""
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

    @jaxtyped(typechecker=beartype)
    def explain(
        self, 
        model: nn.Module, 
        input_tensor: Tensor, 
        target: Optional[Tensor] = None,
        normalize: bool = True,
        return_explanation: bool = True,
        **kwargs
    ) -> ExplanationOutput:
        with torch.inference_mode(False), torch.enable_grad():
            # Prepare input inside the grad-enabled block so caller inference
            # contexts cannot create non-differentiable attention tensors.
            input_tensor = input_tensor.clone().detach().requires_grad_(True)

            # Get patched model
            patched_model = self._get_patched_model(model)
            patched_model.to(input_tensor.device)

            # Get target
            target = self._get_target(model, input_tensor, target)

            # Call Wrapper
            if patched_model.model_type in ("dinov3", "aloe_hf", "hf_vit", "hf_siglip"):
                heatmap = patched_model.compute_chefer_dinov3_from_class(
                    class_idx=target,
                    image=input_tensor,
                    normalize=normalize
                )
            else:
                raise NotImplementedError(
                    "Chefer CAM is only implemented for DINOv3, HF ViT, HF SigLIP/SigLIP2, and ALOE in this codebase."
                )

            # Heatmap is [B, 1, H, W] in 0-1 range (min-max normed)
            contribution_map = heatmap.detach()

            # Generate visualization
            merged_kwargs = {**self.kwargs, **kwargs}
            merged_kwargs["to_numpy"] = False

            explanation = (
                gradient_to_image(
                    input_tensor.detach(),
                    contribution_map,
                    **merged_kwargs
                )
                if return_explanation
                else None
            )

            return ExplanationOutput(
                contribution_map=contribution_map,
                explanation=explanation,
                prediction=target.detach(),
                attributions=contribution_map,
                dynamic_linear_weights=None,
                explained_class_idx=target.detach()
            )
