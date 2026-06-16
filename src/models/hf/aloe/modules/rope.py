# 2-D Rotary Position Embedding (RoPE) for ALOE vision models.
#
# Unlike additive position embeddings, RoPE is applied inside attention by
# rotating Q and K according to 2-D patch grid coordinates.  Prefix tokens
# (registers, CLS) use identity rotation (cos=1, sin=0), matching how
# Hugging Face DINOv3 applies RoPE only to patch tokens.
#
# Layout convention (matches AloeVisionEmbeddings / HF DINOv3 ViT):
#   [CLS, register_1, …, register_R, patch_0, …, patch_{N-1}]
#   ─────────────── num_prefix ──────────────  ───── N ─────
#
# Patch tables follow ``transformers`` DINOv3 ViT: normalized patch-center
# coordinates in ``[-1, 1]``, ``inv_freq = 1/θ^t`` with ``t = 0, 4/d, …``,
# then ``angles = 2π * coords * inv_freq``, flatten and tile(2).

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def rotate_half(x: Tensor) -> Tensor:
    """[x1, x2] → [-x2, x1] — standard RoPE half-rotation."""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Apply 2-D RoPE to Q and K.

    Args:
        q, k : ``(B, H, T, D)`` — query and key tensors.
        cos, sin : ``(1, T, D)`` — precomputed tables (broadcast over B and H).

    Returns:
        Rotated ``(q, k)`` with same shape.
    """
    cos = cos.unsqueeze(1)   # (1, 1, T, D) — broadcast over B and H
    sin = sin.unsqueeze(1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# Table computation
# ---------------------------------------------------------------------------

def _dinov3_patch_center_coords(
    num_patches_h: int,
    num_patches_w: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Same geometry as ``get_patches_center_coordinates`` in HF DINOv3 ViT."""
    coords_h = torch.arange(0.5, num_patches_h, dtype=dtype, device=device)
    coords_w = torch.arange(0.5, num_patches_w, dtype=dtype, device=device)
    coords_h = coords_h / num_patches_h
    coords_w = coords_w / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)
    coords = coords.flatten(0, 1)
    return 2.0 * coords - 1.0


def compute_rope_2d_tables(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    num_prefix: int,
    device: torch.device,
    dtype: torch.dtype,
    theta: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """
    Build ``(cos, sin)`` of shape ``(1, num_prefix + grid_h*grid_w, head_dim)``.

    Patch rows match Hugging Face ``DINOv3ViTRopePositionEmbedding`` (normalized
    patch centers, ``inv_freq`` spacing ``4/head_dim``).  Prefix rows are
    ``cos=1``, ``sin=0`` so :func:`apply_rotary_pos_emb` matches "RoPE on
    patches only" when prefix length matches HF's cls + registers.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2-D RoPE."

    inv_freq = 1.0 / (
        theta ** torch.arange(0, 1, 4.0 / head_dim, device=device, dtype=torch.float32)
    )

    patch_coords = _dinov3_patch_center_coords(
        grid_h, grid_w, device=device, dtype=torch.float32
    )
    angles = 2 * math.pi * patch_coords[:, :, None] * inv_freq[None, None, :]
    angles = angles.flatten(1, 2)
    angles = angles.tile(2)

    patch_cos = torch.cos(angles)
    patch_sin = torch.sin(angles)

    if num_prefix > 0:
        pref_cos = torch.ones(num_prefix, head_dim, device=device, dtype=torch.float32)
        pref_sin = torch.zeros(num_prefix, head_dim, device=device, dtype=torch.float32)
        cos = torch.cat([pref_cos, patch_cos], dim=0).unsqueeze(0).to(dtype)
        sin = torch.cat([pref_sin, patch_sin], dim=0).unsqueeze(0).to(dtype)
    else:
        cos = patch_cos.unsqueeze(0).to(dtype)
        sin = patch_sin.unsqueeze(0).to(dtype)

    return cos, sin


# ---------------------------------------------------------------------------
# Module wrapper (stateless — parameters are fixed frequencies)
# ---------------------------------------------------------------------------

class RoPE2D(torch.nn.Module):
    """
    Stateless 2-D RoPE module.

    No learnable parameters — frequencies are determined by ``theta`` and
    ``head_dim``.  Call :meth:`compute_tables` to get ``(cos, sin)`` tables
    for a specific spatial layout.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2-D RoPE."
        self.head_dim = head_dim
        self.theta = theta

    def compute_tables(
        self,
        grid_h: int,
        grid_w: int,
        num_prefix: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(cos, sin)`` of shape ``(1, num_prefix + grid_h*grid_w, head_dim)``."""
        return compute_rope_2d_tables(
            self.head_dim, grid_h, grid_w, num_prefix, device, dtype, self.theta
        )
