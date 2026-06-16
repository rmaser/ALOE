# Vendored B-cos conv stem for the self-contained ALOE HF package.
# Mirrors the BcosConverter ``ConvStem`` / legacy ``DINOv3ViTEmbeddings.patch_embeddings.stem``
# printout: ``BcosConv2d_unnormed → GNLayerNormUncentered2dNoBias → ReLU`` per stage
# (``GNLayerNormUncentered2dNoBias`` ≡ ``NoBias(GNLayerNormUncentered2d)`` here).
# Checkpoint keys (e.g. ``embeddings.conv_stem.stem.N.linear.weight``) load directly.
#
# Architecture (from BcosConverter / src/modules/conv_stem.py):
#   in_channels  →  [outc_0, outc_1, …, outc_K]  conv layers
#   Each layer:  BcosUnnormedConv2d 3×3, stride=2 if outc > prev, else stride=1
#               + ``NoBias(GNLayerNormUncentered2d)`` — same as BcosConverter ``DEFAULT_NORM_LAYER``.
#               + ReLU — same as ``DEFAULT_ACT_LAYER = nn.ReLU``.
#   Final output has outc_K channels at reduced spatial resolution.
#   The patch embedding (BcosUnnormedLinear) then handles the rest.

from __future__ import annotations

from typing import Type

import torch
import torch.nn as nn

from ..configuration_aloe_vision import AloeVisionConfig
from .bcos_core import BcosUnnormedConv2d, DetachableReLU, NoBias, select_bcos_unnormed_conv2d
from .groupnorm_uncentered import GNLayerNormUncentered2d


class _ConvBlock(nn.Module):
    """Single conv+norm+act block in the conv stem."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        b: float,
        stride: int,
        *,
        conv_cls: Type[BcosUnnormedConv2d] = BcosUnnormedConv2d,
    ) -> None:
        super().__init__()
        self.conv = conv_cls(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, b=b)
        self.norm = NoBias(GNLayerNormUncentered2d)(out_ch)
        self.act = DetachableReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class AloeConvStem(nn.Module):
    """
    B-cos conv stem — mirrors the BcosConverter ``ConvStem`` architecture.

    Transforms a 6-channel input (3-channel RGB expanded via channel doubling)
    through a series of 3×3 conv layers (stride=2 when channels grow, else 1)
    down to a lower spatial resolution.  The final spatial patches are then
    handled by the regular patch-embedding ``BcosUnnormedLinear`` in
    :class:`AloeVisionEmbeddings`.

    Weight key format (for weight migration):
        ``embeddings.conv_stem.stem.{idx}.conv.linear.weight``   (conv weight)
        ``embeddings.conv_stem.stem.{idx}.norm.weight``          (affine scale; bias removed)

    Built from ``config.aloe_add_conv_stem`` (channel schedule),
    ``config.aloe_in_channels``, ``config.aloe_b_conv``, and
    ``config.aloe_bcos_impl`` (v1 vs v2 conv scaling).
    """

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        schedule = config.aloe_add_conv_stem
        if not schedule:
            raise ValueError("AloeConvStem requires config.aloe_add_conv_stem")
        channel_schedule = list(schedule)
        in_channels = config.aloe_in_channels
        b = float(config.aloe_b_conv)
        ConvCls = select_bcos_unnormed_conv2d(config)
        blocks: list[nn.Module] = []
        prev = in_channels
        for out_ch in channel_schedule:
            stride = 2 if out_ch > prev else 1
            blocks.append(_ConvBlock(prev, out_ch, b=b, stride=stride, conv_cls=ConvCls))
            prev = out_ch
        self.stem = nn.Sequential(*blocks)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``(B, out_channels, H', W')`` feature map."""
        return self.stem(x)

    @property
    def downsampling_factor(self) -> int:
        """Total spatial downsampling factor accumulated by stride-2 layers."""
        factor = 1
        for block in self.stem:
            if block.conv.linear.stride[0] == 2:
                factor *= 2
        return factor
