import torch
import torch.nn as nn
from typing import Optional

class LogitLayer(nn.Module):
    """
    Applies optional temperature scaling and bias to logits.
    Commonly used in B-cos models to adjust the dynamic range of logits.
    """
    def __init__(self, logit_temperature: Optional[float] = None, logit_bias: Optional[float] = None):
        super().__init__()
        self.logit_temperature = logit_temperature
        self.logit_bias = logit_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.logit_temperature is not None:
            x = x * self.logit_temperature
        if self.logit_bias is not None:
            x = x + self.logit_bias
        return x
