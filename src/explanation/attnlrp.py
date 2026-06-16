"""AttnLRP explainer using LXT (LRP-eXplains-Transformers)."""
from typing import Optional, Any, Literal
import copy
import torch
import torch.nn as nn
from torch import Tensor

from .abstract import BaseExplainer, ExplanationOutput
from .utils import gradient_to_image
from beartype import beartype
from jaxtyping import jaxtyped
from lxt.efficient import monkey_patch
from .attlrp.maps import aloe_map, dinov3_map, siglip2_map, google_vit_map



ModelType = Literal["dinov3", "siglip2", "google_vit", "aloe", "auto"]


def _unwrap_optimized_model(model: nn.Module) -> nn.Module:
    """Unwrap torch.compile / OptimizedModule wrappers for architecture checks."""
    if model.__class__.__name__ == "OptimizedModule" and hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def _is_native_aloe_model(model: nn.Module) -> bool:
    """Return whether *model* is one of the native ALOE Hugging Face modules."""
    model = _unwrap_optimized_model(model)
    class_name = model.__class__.__name__.lower()
    if class_name.startswith("aloe"):
        return True

    config = getattr(model, "config", None)
    model_type = str(getattr(config, "model_type", "")).lower()
    if model_type.startswith("aloe_") or hasattr(config, "aloe_backbone"):
        return True

    for _, module in model.named_modules():
        module_class = module.__class__
        module_name = module_class.__name__.lower()
        module_path = module_class.__module__.lower()
        if module_name.startswith("aloe") or ".aloe." in module_path:
            return True
    return False


def _detect_model_type(model: nn.Module) -> Optional[str]:
    """Auto-detect the model type based on module names."""
    model = _unwrap_optimized_model(model)
    if _is_native_aloe_model(model):
        return "aloe"

    model_name = model.__class__.__name__.lower()
    
    # Check class name
    if "dinov3" in model_name or "dino" in model_name:
        return "dinov3"
    if "siglip" in model_name:
        return "siglip2"
    if "vit" in model_name:
        return "google_vit"
    
    # Check module names
    # Check module names and types
    for name, module in model.named_modules():
        name_lower = name.lower()
        class_name_lower = module.__class__.__name__.lower()
        
        if "dinov3" in name_lower or "dinov3" in class_name_lower:
            return "dinov3"
        if "siglip" in name_lower or "siglip" in class_name_lower:
            return "siglip2"
        if "vit" in class_name_lower and "google" in class_name_lower: # unlikely but possible
             pass
    
    # Fallback: check if it's a generic ViT
    if "vit" in model_name:
        return "google_vit"

    return None


def _get_patch_map(model_type: str) -> Optional[dict[Any, Any]]:
    """Get the patch map for a given model type."""    
    if model_type == "dinov3":
        return dinov3_map
    elif model_type == "siglip2":
        return siglip2_map
    elif model_type == "google_vit":
        return google_vit_map
    elif model_type == "aloe":
        return aloe_map
    
    raise ValueError(f"Unknown model type: {model_type}")


def _prepare_input_tensor(input_tensor: Tensor, model_type: Optional[str]) -> Tensor:
    """Prepare a fresh input leaf tensor for the selected architecture."""
    if model_type == "aloe":
        from src.models.hf.aloe.modules.explanation import _prepare_tensor_for_explain

        return _prepare_tensor_for_explain(input_tensor)
    return input_tensor.clone().detach().requires_grad_(True)


class AttnLRPExplainer(BaseExplainer):
    """
    Attention-based Layer-wise Relevance Propagation using LXT.
    
    This explainer monkey-patches a ViT model to compute LRP attributions.
    The patched model is cached to avoid repeated deep copies.
    
    Supports automatic model type detection for DINOv3, SigLIP2, Google ViT, and native ALOE.
    """
    
    def __init__(
        self, 
        model_type: ModelType = "auto",
        patch_map: Optional[dict[Any, Any]] = None,
        **kwargs
    ):
        """
        Initialize AttnLRP explainer.
        
        Args:
            model_type: Model architecture type. Options: "dinov3", "siglip2", "google_vit", "aloe", "auto".
                       If "auto", attempts to detect the model type automatically.
            patch_map: Optional dictionary mapping layer types to LRP rules.
                      If None, uses default mapping for the detected model architecture.
        """
        super().__init__(**kwargs)
        self.model_type = model_type
        self.patch_map = patch_map
        self.kwargs = kwargs
        self._patched_model = None
        self._model_id = None  # Track which model we patched
        self._patched_model_type = None
    
    def _get_patched_model(self, model: nn.Module) -> nn.Module:
        """Get or create a monkey-patched model for LRP."""
        model_id = id(model)
        
        # Return cached model if available and still valid
        if self._patched_model is not None and self._model_id == model_id:
            return self._patched_model

        # Create new patched model
        # Clear captured_states hooks if present (from ModelFactory)
        # These can contain non-leaf tensors which fail deepcopy
        if hasattr(model, "captured_states"):
             # We create a new empty list so we don't modify the original model's list capability
             # but we avoid copying the potentially problematic tensors
            model.captured_states = []
            
        patched = copy.deepcopy(model)
        if hasattr(patched, "_orig_mod"):
            patched = patched._orig_mod
        
        # Disable parameter grads to reduce memory
        for p in patched.parameters():
            p.requires_grad = False
        
        # Use explicit model_type if provided, else detect
        if self.model_type != "auto":
             model_type = self.model_type
        else:
             model_type = _detect_model_type(model)

        if model_type is None:
            raise ValueError(f"Could not auto-detect model type for {model.__class__.__name__}. Please specify model_type manually.")
        patch_map = self.patch_map if self.patch_map is not None else _get_patch_map(model_type)
        
        # Apply monkey patching
        if patch_map is not None:
            monkey_patch(patched, patch_map=patch_map, verbose=True)
        elif model_type != "aloe":
            raise Exception("No patch map provided and auto-detection failed.")
            # Use default patching without explicit map
            monkey_patch(patched, verbose=True)
        
        patched.eval()
        self._patched_model = patched
        self._model_id = model_id
        self._patched_model_type = model_type
        
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
        return_explanation: bool = True,
        **kwargs
    ) -> ExplanationOutput:
        # Get patched model
        patched_model = self._get_patched_model(model)
        model_type = self._patched_model_type

        # Prepare input
        input_tensor = _prepare_input_tensor(input_tensor, model_type)
        patched_model.to(input_tensor.device)
        
        # Get target
        target = self._get_target(model, input_tensor, target)
        
        # Forward pass through patched model
        outputs = patched_model(input_tensor)
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        elif hasattr(outputs, 'pooler_output'):
            logits = outputs.pooler_output
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        # Gather logits for target class
        to_be_explained = torch.gather(logits, 1, target.unsqueeze(1)).sum()

        # Backward to compute LRP
        to_be_explained.backward(inputs=[input_tensor])
        
        # Get attributions (input * gradient for LRP)
        if input_tensor.grad is None:
            raise RuntimeError("Input tensor gradient is None after backward pass.")
        
        attribution = input_tensor * input_tensor.grad
        contribution_map = attribution.sum(1, keepdim=True)
        
        # Generate visualization
        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs["to_numpy"] = False
        
        explanation = (
            gradient_to_image(
                input_tensor.detach(), 
                input_tensor.grad.detach(),
                **merged_kwargs
            )
            if return_explanation
            else None
        )
        
        return ExplanationOutput(
            contribution_map=contribution_map.detach(),
            explanation=explanation,
            prediction=target.detach(),
            attributions=attribution.detach(),
            dynamic_linear_weights=input_tensor.grad.detach(),
            explained_class_idx=target.detach()
        )
