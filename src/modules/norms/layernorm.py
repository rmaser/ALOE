import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from bcos.modules.common import DetachableModule

class DetachableLayerNorm(nn.LayerNorm, DetachableModule):
    """
    A non-CNN detachable Layer Norm.
    This is used for the transformers!
    """

    def __init__(self, *args, **kwargs):
        DetachableModule.__init__(self)
        kwargs.pop('b', None)
        kwargs.pop('detach_output', None)
        super().__init__(*args, **kwargs)

    def forward(self, input: "Tensor") -> "Tensor":
        # if not detaching -> just use normal pytorch forward pass
        if not self.detach:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps
            )

        # ------------ manual LN detached forward pass -------------
        d_num = len(self.normalized_shape)

        # calc stats
        var, mean = torch.var_mean(
            input, dim=tuple(range(-d_num, 0)), unbiased=False, keepdim=True
        )
        var = var.detach()
        std = (var + self.eps).sqrt_()

        # normalize
        x = (input - mean) / std

        # affine transformation
        if self.weight is not None:
            x = self.weight * x

        if self.bias is not None:
            x = x + self.bias

        return x
