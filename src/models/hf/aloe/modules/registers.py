from __future__ import annotations

import torch
import torch.nn as nn


class RegisterTokens(nn.Module):
    """Learned register tokens placed after CLS (when present) and before patch tokens."""

    def __init__(self, num_registers: int, hidden_size: int) -> None:
        super().__init__()
        if num_registers < 0:
            raise ValueError("num_registers must be non-negative")
        self.num_registers = num_registers
        if num_registers > 0:
            self.tokens = nn.Parameter(torch.randn(1, num_registers, hidden_size))
        else:
            self.register_parameter("tokens", None)

    def forward(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.num_registers == 0:
            raise RuntimeError("RegisterTokens.forward called with num_registers=0")
        return self.tokens.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
