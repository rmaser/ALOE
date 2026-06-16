from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch import Tensor
import torch.nn as nn

@jaxtyped(typechecker=beartype)
class CLSPooler(nn.Module):

    # @jaxtyped(typechecker=beartype)
    def forward(self, x: Float[Tensor, "batch tokens embedding"]) -> Float[Tensor, "batch embedding"]:
        return x[:, 0, :]
