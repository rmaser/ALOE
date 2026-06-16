"""
LIME explainer using Captum with ViT-aware patch masking.
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from captum.attr import Lime

from .captum_base import CaptumExplainer
from .abstract import ExplanationOutput
from beartype import beartype
from jaxtyping import jaxtyped


class LimeExplainer(CaptumExplainer):
    """LIME: local interpretable model-agnostic explanations with ViT patch masking."""
    
    def __init__(self, n_samples: int = 500, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.patch_size = patch_size
    
    def _create_vit_feature_mask(self, input_tensor: Tensor) -> Tensor:
        """Create a feature mask based on ViT patch size."""
        b, c, h, w = input_tensor.shape
        patch_size = self.patch_size
        
        assert h % patch_size == 0 and w % patch_size == 0, \
            f"Image size ({h}x{w}) must be divisible by patch_size={patch_size} for LIME feature_mask."
        
        num_patches_h = h // patch_size
        num_patches_w = w // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Create patch ids on a coarse grid [1, 1, num_patches_h, num_patches_w]
        patch_ids = torch.arange(
            num_patches, device=input_tensor.device
        ).view(1, 1, num_patches_h, num_patches_w).float()
        
        # Upsample to image resolution with nearest-neighbor to get [1, 1, H, W]
        feature_mask = F.interpolate(
            patch_ids, size=(h, w), mode="nearest"
        ).long()  # shape [1, 1, H, W], broadcastable to [B, C, H, W]
        
        return feature_mask
    
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
        
        # Create ViT-aware feature mask
        feature_mask = self._create_vit_feature_mask(input_tensor)
        
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
        
        # Compute LIME
        lime = Lime(forward_fn)
        attributions = lime.attribute(
            input_tensor, 
            target=target,
            feature_mask=feature_mask,
            n_samples=self.n_samples
        )
        
        return self._attributions_to_output(input_tensor, attributions, target, **kwargs)
