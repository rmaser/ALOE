from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker
from torch import Tensor
import torch
from typing import Union

class CosineDistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Union[Float[Tensor, "batch tokens embedding"], Float[Tensor, "batch 1 tokens embedding"]], 
        y: Union[Float[Tensor, "batch tokens embedding"], Float[Tensor, "batch 1 tokens embedding"]]
    ) -> Float[Tensor, "batch"]: # Output is now (B, T) unless you mean/sum it
        
        if x.dim() == 4:
            x = x.squeeze(dim=1)
        if y.dim() == 4:
            y = y.squeeze(dim=1)
        
        cosine_similarity = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        
        return torch.mean(1 - cosine_similarity, dim=-1)