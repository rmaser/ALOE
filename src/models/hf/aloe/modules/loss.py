"""HF-local loss helpers for Hub-compatible ALOE image-classification models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformOffLabelsBCEWithLogitsLoss(nn.Module):
    """BCEWithLogits with uniform off-target mass for multiclass B-cos heads.

    If ``off_label`` is not provided, off-target entries use ``1 / num_classes``.
    For example with ``num_classes=5`` and target ``3``, the dense target becomes
    ``[0.2, 0.2, 0.2, 1.0, 0.2]``.
    """

    def __init__(self, reduction: str = "mean", off_label: Optional[float] = None):
        super().__init__()
        self.reduction = reduction
        self.off_label = off_label

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert x.shape[0] == target.shape[0]

        num_classes = x.shape[-1]
        off_value = self.off_label or (1.0 / num_classes)
        if target.shape != x.shape:
            target = F.one_hot(target, num_classes=num_classes).to(dtype=x.dtype)

        target = target.clamp(min=off_value)
        return F.binary_cross_entropy_with_logits(x, target, reduction=self.reduction)

    def extra_repr(self) -> str:
        result = f"reduction={self.reduction}, "
        if self.off_label is not None:
            result += f"off_label={self.off_label}, "
        return result[:-2]
