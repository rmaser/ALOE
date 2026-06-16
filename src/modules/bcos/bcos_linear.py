"""
Contains a Linear layer which uses the B-cos transform.

NOTE: In case you'        skip_norm_division: bool = False,e wondering why the convolution models do not use
`BcosLinear`, it's because maintaining two versions of essentially
the same thing would be very error-prone during development and testing!
"""

from typing import Union
import math

import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bcos.modules import DetachableModule

class NormedLinear(nn.Linear):
    """
    Standard linear transform, but with unit norm weights.
    """

    def forward(self, input: Tensor) -> Tensor:
        w = self.weight / LA.vector_norm(self.weight, dim=1, keepdim=True)
        return F.linear(input, w, self.bias)

class UnnormedLinear(nn.Linear):
    """
    Standard linear transform, but with unit norm weights.
    """

    def forward(self, input: Tensor) -> Tensor:
        w = self.weight #/ LA.vector_norm(self.weight, dim=1, keepdim=True)
        # logger.info(f"DEBUG: UnnormedLinear w requires_grad={w.requires_grad}")
        return F.linear(input, w, self.bias)

class BcosLinear(DetachableModule):
    """
    BcosLinear is a linear transform with unit norm weights and a cosine similarity
    activation function. The cosine similarity is calculated between the input
    vector and the weight vector. The output is then scaled by the cosine
    similarity.

    See the paper for more details: https://arxiv.org/abs/2205.10268

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    bias : bool
        This is ignored. BcosLinear does not support bias.
    device : Optional[torch.device]
        The device of the weights.
    dtype : Optional[torch.dtype]
        The dtype of the weights.
    b : int | float
        The base of the exponential used to scale the cosine similarity.
    max_out : int
        The number of output vectors to use. If this is greater than 1, the
        output is calculated as the maximum of `max_out` vectors. This is
        equivalent to using a MaxOut activation function.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device=None,
        dtype=None,
        b: Union[int, float] = 2,
        max_out: int = 1,
        detach_output: bool = False,
    ) -> None:
        assert not bias
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = None  # Set to None to match nn.Linear interface when bias=False
        self.b = b
        self.max_out = max_out
        self.detach_output = detach_output

        if bias:
            raise ValueError("BcosLinear does not support bias. Set bias=False.")

        # Define the weight parameter directly (no nn.Linear wrapper)
        self.linear = NormedLinear(
            in_features,
            out_features * self.max_out,
            bias=False,
            device=device,
            dtype=dtype,
        )
        
        # Initialize the weights
        self.reset_parameters()
    
    @property
    def dtype(self):
        return self.linear.weight.dtype

    def reset_parameters(self) -> None:
        """Initialize the weight parameters."""
        # Use standard initialization (same as nn.Linear)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
    

    def forward(self, in_tensor: Tensor) -> Tensor:
        # This is very useful for detaching the attention weights
        out = self._forward(in_tensor)
        if self.detach_output and self.detach:
            out = out.detach()
        return out
    
    def _forward(self, in_tensor: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            in_tensor: Input tensor. Expected shape: (*, H_in)

        Returns:
            B-cos Linear output on the input tensor.
            Shape: (*, H_out)
        """
        
        # Simple linear layer
        out = self.linear(in_tensor)
        # logger.info(f"DEBUG: BcosLinear out requires_grad={out.requires_grad}")

        # MaxOut computation
        if self.max_out > 1:
            M = self.max_out
            O = self.out_features  # noqa: E741
            out = out.unflatten(dim=-1, sizes=(O, M))
            out = out.max(dim=-1, keepdim=False).values

        # if B=1, no further calculation necessary
        if self.b == 1:
            return out

        # Calculating the norm of input vectors ||x||
        norm = LA.vector_norm(in_tensor, dim=-1, keepdim=True) + 1e-12

        # Calculate the dynamic scale (|cos|^(B-1))
        # Note that cos = (x·ŵ)/||x||
        maybe_detached_out = out
        if self.detach:
            maybe_detached_out = out.detach()
            norm = norm.detach()

        if self.b == 2:
            dynamic_scaling = maybe_detached_out.abs() / norm
        else:
            abs_cos = (maybe_detached_out / norm).abs() + 1e-6
            dynamic_scaling = abs_cos.pow(self.b - 1)

        # put everything together
        out = dynamic_scaling * out  # |cos|^(B-1) (ŵ·x)

        return out

    def extra_repr(self) -> str:
        # rest in self.linear
        s = "B={b}"

        if self.max_out > 1:
            s += ", max_out={max_out}"

        if self.detach_output:
            s += ", detach_output={detach_output}"

        return s.format(**self.__dict__)
        
class BcosUnnormedLinear(BcosLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(
            self.in_features,
            self.out_features * self.max_out,
            bias=False,
            device=kwargs.get('device'),
            dtype=kwargs.get('dtype'),
        )
