# Modular position-embedding building blocks + unified AloeVisionEmbeddings.
# Every backbone (SigLIP2, DINOv3, ViT, …) uses AloeVisionEmbeddings and
# selects its position embedding via config.aloe_position_embedding_type.

from __future__ import annotations

import math

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor

from .bcos_core import select_bcos_unnormed_linear
from .conv_stem import AloeConvStem
from .registers import RegisterTokens
from .rope import RoPE2D

if TYPE_CHECKING:
    from ..configuration_aloe_vision import AloeVisionConfig


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(pixel_values: Tensor, patch_size: int) -> Tensor:
    """
    Extract non-overlapping patches from a standard image.

    ``(B, C, H, W)`` → ``(B, H/P × W/P, C·P·P)``
    """
    B, C, H, W = pixel_values.shape
    P = patch_size
    x = pixel_values.reshape(B, C, H // P, P, W // P, P)
    return x.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, -1, C * P * P)


def _stem_tile_size(logical_patch_size: int, stem_downsample: int) -> int:
    """
    Stem output cells spanned by one logical patch of ``logical_patch_size`` input pixels
    when the stem downsamples by ``stem_downsample`` per spatial axis.

    BcosConverter ``ConvStem`` uses integer ``patch_size // downsampling``; when HF reports a
    smaller patch (e.g. SigLIP 14) than the backbone yaml used for the stem (16), plain
    ``//`` can yield 0.  Ceil division keeps at least one cell, matching ``patch_dim`` layout
    (typically ``out_channels × 1 × 1``) for the same checkpoint.
    """
    ds = int(stem_downsample)
    if ds < 1:
        raise ValueError(f"stem_downsample must be >= 1, got {ds}")
    p = int(logical_patch_size)
    return max(1, (p + ds - 1) // ds)


# ---------------------------------------------------------------------------
# Position-embedding modules — uniform interface
# ---------------------------------------------------------------------------

class PositionEmbeddingBase(nn.Module):
    """
    Common interface for all ALOE position embedding modules.

    ``forward`` receives the patch embeddings and an optional CLS token,
    and returns ``(cls_with_pos | None, patches_with_pos)``.
    """

    def forward(
        self,
        patch_embeds: Tensor,
        cls_token: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Optional[Tensor], Tensor]:
        raise NotImplementedError

    def reset_parameters(self) -> None:
        pass


class Absolute1DPositionEmbedding(PositionEmbeddingBase):
    """
    Learned 1-D absolute position embeddings (DINOv3 / standard ViT style).

    Position 0 is the CLS position; positions 1…N are patch positions.
    If ``has_cls=False`` the table has ``num_patches`` entries and the
    CLS token is not shifted.
    """

    def __init__(self, num_patches: int, hidden_size: int, has_cls: bool = True) -> None:
        super().__init__()
        self.has_cls = has_cls
        num_positions = num_patches + (1 if has_cls else 0)
        self.embedding = nn.Embedding(num_positions, hidden_size)

    def forward(self, patch_embeds: Tensor, cls_token: Optional[Tensor] = None, **kwargs) -> tuple[Optional[Tensor], Tensor]:
        pos = self.embedding.weight.to(dtype=patch_embeds.dtype)
        B = patch_embeds.shape[0]
        if self.has_cls and cls_token is not None:
            cls_out = (cls_token + pos[0]).expand(B, -1, -1)
            patch_out = patch_embeds + pos[1:]
            return cls_out, patch_out
        return None, patch_embeds + pos

    def reset_parameters(self) -> None:
        init.normal_(self.embedding.weight, std=self.embedding.embedding_dim ** -0.5)


class NaFlex2DPositionEmbedding(PositionEmbeddingBase):
    """
    Absolute 2-D position embeddings with NaFlex bilinear resizing (SigLIP2 style).

    The grid is stored at the base resolution (``grid_size × grid_size``) and
    bilinearly interpolated to the spatial layout given by ``spatial_shapes``
    on each forward call.  No CLS token is used with this embedding.
    """

    def __init__(self, num_patches: int, hidden_size: int) -> None:
        super().__init__()
        self.grid_size = int(num_patches ** 0.5)
        self.embedding = nn.Embedding(num_patches, hidden_size)

    @staticmethod
    def _resize(
        pos: Tensor,           # (grid_H, grid_W, D)
        spatial_shapes: Tensor,  # (B, 2) int64 – (h, w) in patches per sample
        max_length: int,
    ) -> Tensor:
        B = spatial_shapes.shape[0]
        D = pos.shape[-1]
        source_dtype = pos.dtype
        out = torch.empty((B, max_length, D), device=pos.device, dtype=source_dtype)
        grid = pos.permute(2, 0, 1).unsqueeze(0)   # (1, D, gH, gW)
        if grid.device.type == "cpu":
            grid = grid.float()
        for i in range(B):
            h, w = spatial_shapes[i].tolist()
            if h * w > max_length:
                raise ValueError("Resized positional embeddings exceed max_length.")
            resized = F.interpolate(grid, size=(h, w), mode="bilinear",
                                    align_corners=False, antialias=True)
            resized = resized.reshape(D, h * w).transpose(0, 1).to(source_dtype)
            out[i, : h * w] = resized
            out[i, h * w :] = resized[0]
        return out

    def forward(
        self,
        patch_embeds: Tensor,
        cls_token: Optional[Tensor] = None,   # ignored — SigLIP2 has no CLS
        spatial_shapes: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[None, Tensor]:
        G = self.grid_size
        pos = self.embedding.weight.reshape(G, G, -1)
        if spatial_shapes is not None:
            resized = self._resize(pos, spatial_shapes, max_length=patch_embeds.shape[1])
        else:
            # Square grid — no resizing needed.
            resized = pos.reshape(-1, pos.shape[-1]).unsqueeze(0).expand(patch_embeds.shape[0], -1, -1)
        return None, patch_embeds + resized.to(dtype=patch_embeds.dtype)

    def reset_parameters(self) -> None:
        init.normal_(self.embedding.weight, std=self.embedding.embedding_dim ** -0.5)


class RoPE2DPositionEncoding(PositionEmbeddingBase):
    """
    Null additive embedding for RoPE-based position encoding.

    ``forward`` passes tokens through unchanged — no positional bias is added
    to the token representations.  Instead, call :meth:`compute_tables` to get
    ``(cos, sin)`` tables that :class:`AloeAttention` applies inside attention.

    Token layout assumed in :meth:`compute_tables`:
    ``[CLS, register_1, …, register_R, patch_0, …, patch_{N-1}]``
    where prefix = (1 if CLS else 0) + R (matches HF ``DINOv3ViT``).
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.rope = RoPE2D(head_dim, theta)

    def forward(
        self,
        patch_embeds: Tensor,
        cls_token: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Optional[Tensor], Tensor]:
        """Return tokens unchanged — positions are handled in attention via RoPE."""
        if cls_token is not None:
            return cls_token.expand(patch_embeds.shape[0], -1, -1), patch_embeds
        return None, patch_embeds

    def compute_tables(
        self,
        grid_h: int,
        grid_w: int,
        num_prefix: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Delegate to :class:`RoPE2D` — returns ``(cos, sin)`` for attention."""
        return self.rope.compute_tables(grid_h, grid_w, num_prefix, device, dtype)


# Registry — maps config.aloe_position_embedding_type → class
_POS_EMB_REGISTRY: dict[str, type[PositionEmbeddingBase]] = {
    # Learned table over patch indices (row-major); no bilinear resize — default for SigLIP2 training.
    "absolute": Absolute1DPositionEmbedding,
    "absolute_1d": Absolute1DPositionEmbedding,
    # Optional SigLIP2-NaFlex-style 2-D grid + ``spatial_shapes`` bilinear interpolation.
    "naflex_2d": NaFlex2DPositionEmbedding,
    # "rope_2d" is handled separately in build_position_embedding (needs head_dim).
}


def build_position_embedding(config: AloeVisionConfig) -> PositionEmbeddingBase:
    """Construct the position embedding from ``config.aloe_position_embedding_type``."""
    pt = config.aloe_position_embedding_type
    if pt == "rope_2d":
        head_dim = config.hidden_size // config.num_attention_heads
        theta = getattr(config, "aloe_rope_theta", 100.0)
        return RoPE2DPositionEncoding(head_dim, theta)
    cls = _POS_EMB_REGISTRY.get(pt)
    if cls is None:
        raise ValueError(f"Unknown aloe_position_embedding_type {pt!r}. "
                         f"Supported: {sorted(_POS_EMB_REGISTRY) + ['rope_2d']}.")
    if cls is Absolute1DPositionEmbedding:
        return cls(config.num_patches, config.hidden_size, has_cls=config.aloe_cls_token)
    return cls(config.num_patches, config.hidden_size)


# ---------------------------------------------------------------------------
# Unified vision embeddings
# ---------------------------------------------------------------------------

class AloeVisionEmbeddings(nn.Module):
    """
    Unified ALOE vision embeddings — works for every backbone.

    Token layout after ``forward`` (same as HF ``DINOv3ViT`` / ``dinov2_with_registers``):
    ``[(CLS,) register_1, …, register_R, patch_1, …, patch_N]``

    CLS is included only when ``config.aloe_cls_token=True`` (index ``0``).
    :class:`AloeClsTokenPooler` reads that CLS row.

    Input to ``forward``:
    * ``pixel_values.dim() == 4`` → standard image ``(B, C, H, W)``:
      patches are extracted internally (DINOv3, ViT).
    * ``pixel_values.dim() == 3`` → pre-patchified ``(B, T, C·P·P)``:
      used directly (e.g. SigLIP2-style pipelines).

    Extra keyword arguments (e.g. ``spatial_shapes`` for SigLIP2) are
    forwarded to the position embedding module.
    """

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        from ..modeling_aloe_base import patch_size_from_aloe_config

        self.patch_size = patch_size_from_aloe_config(config)

        # Optional conv stem: transforms 6-channel input before patch extraction.
        # When present, patch_embedding receives the stem's output channels directly,
        # so flat_dim uses the stem's output channel count and the (reduced) tile size.
        self.conv_stem: AloeConvStem | None = None
        if config.aloe_add_conv_stem:
            self.conv_stem = AloeConvStem(config)
            ds = self.conv_stem.downsampling_factor
            tile_size = _stem_tile_size(self.patch_size, ds)
            flat_dim = self.conv_stem.out_channels * tile_size * tile_size
        else:
            flat_dim = config.aloe_in_channels * self.patch_size * self.patch_size

        _Lin = select_bcos_unnormed_linear(config)
        self.patch_embedding = _Lin(flat_dim, config.hidden_size, b=config.aloe_b_linear, detach_output=False)
        self.pos_embedding: PositionEmbeddingBase = build_position_embedding(config)

        self.cls_token = (
            nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            if config.aloe_cls_token
            else None
        )
        self.register_tokens = (
            RegisterTokens(config.aloe_num_registers, config.hidden_size)
            if config.aloe_num_registers > 0
            else None
        )
        # Prefix length (CLS + registers) before patch tokens — RoPE identity rows.
        self._num_prefix: int = (1 if config.aloe_cls_token else 0) + config.aloe_num_registers

    def reset_parameters(self) -> None:
        """Call from ``_init_aloe_submodules`` instead of relying on generic init."""
        if self.conv_stem is not None:
            for block in self.conv_stem.stem:
                # Legacy BcosConverter ConvStem kept the conv default init (kaiming_uniform_).
                init.kaiming_uniform_(block.conv.linear.weight, a=math.sqrt(5))
        init.xavier_uniform_(self.patch_embedding.linear.weight)
        if self.cls_token is not None:
            init.normal_(self.cls_token, std=1e-6)
        self.pos_embedding.reset_parameters()
        if self.register_tokens is not None and self.register_tokens.tokens is not None:
            init.normal_(self.register_tokens.tokens, std=1e-6)

    def get_position_embeddings(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "tuple[Tensor, Tensor] | None":
        """
        Return ``(cos, sin)`` RoPE tables when ``aloe_position_embedding_type="rope_2d"``,
        or ``None`` for all additive position embedding types.

        Tables have shape ``(1, num_prefix + grid_h*grid_w, head_dim)`` and are
        passed through the encoder stack to :class:`AloeAttention`.
        """
        if isinstance(self.pos_embedding, RoPE2DPositionEncoding):
            return self.pos_embedding.compute_tables(
                grid_h, grid_w, self._num_prefix, device, dtype
            )
        return None

    def forward(self, pixel_values: Tensor, **kwargs) -> Tensor:
        dtype = self.patch_embedding.linear.weight.dtype
        B = pixel_values.shape[0]
        device = pixel_values.device

        # Accept both standard images (B, C, H, W) and pre-patchified (B, T, flat).
        if pixel_values.dim() == 4:
            if self.conv_stem is not None:
                # Conv stem reduces spatial resolution; extract patches from its output.
                feat = self.conv_stem(pixel_values.to(dtype=dtype))   # (B, C', H', W')
                ds = self.conv_stem.downsampling_factor
                tile = _stem_tile_size(self.patch_size, ds)
                patches = extract_patches(feat, tile)
            else:
                patches = extract_patches(pixel_values, self.patch_size)
        else:
            patches = pixel_values

        patch_embeds = self.patch_embedding(patches.to(dtype=dtype))   # (B, T, D)

        cls_with_pos, patch_embeds = self.pos_embedding(
            patch_embeds, cls_token=self.cls_token, **kwargs
        )

        tokens: list[Tensor] = []
        if cls_with_pos is not None:
            tokens.append(cls_with_pos)
        if self.register_tokens is not None:
            tokens.append(self.register_tokens(B, device, dtype))
        tokens.append(patch_embeds)

        return torch.cat(tokens, dim=1)
