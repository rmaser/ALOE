from typing import Optional
import torch.nn as nn
from torch import Tensor
from .abstract import BaseExplainer, ExplanationOutput
from beartype import beartype
from jaxtyping import jaxtyped

class BcosExplainer(BaseExplainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    @jaxtyped(typechecker=beartype)
    def explain(
        self, 
        model: nn.Module, 
        input_tensor: Tensor, 
        target: Optional[Tensor] = None, 
        **kwargs
    ) -> ExplanationOutput:
        
        # Check if model has explain method
        # Note: We might need to access the underlying model if it's wrapped
        # Force to_numpy=False to ensure we get Tensors as expected by ExplanationOutput
        kwargs["to_numpy"] = False
        
        if hasattr(model, "explain"):
            out = model.explain(input_tensor, idx=target, **self.kwargs, **kwargs)
        elif hasattr(model, "module") and hasattr(model.module, "explain"):
             out = model.module.explain(input_tensor, idx=target, **self.kwargs, **kwargs)
        else:
            raise NotImplementedError("Model does not support B-cos explanation (missing 'explain' method).")
            
        return ExplanationOutput(**out)
