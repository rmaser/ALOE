from jaxtyping import Float, jaxtyped
from beartype import beartype as typechecker
from torch import Tensor
import torch
from typing import Union

class MSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @jaxtyped(typechecker=typechecker)
    def forward(
        self, 
        x: Union[Float[Tensor, "batch tokens embedding"], Float[Tensor, "batch 1 tokens embedding"]], 
        y: Union[Float[Tensor, "batch tokens embedding"], Float[Tensor, "batch 1 tokens embedding"]]
    ) -> Float[Tensor, "batch"]:
        
        if x.dim() == 4:
            x = x.squeeze(dim=1)
        if y.dim() == 4:
            y = y.squeeze(dim=1)
        
        return torch.nn.functional.mse_loss(x, y, reduction='none').mean(dim=(1, 2))