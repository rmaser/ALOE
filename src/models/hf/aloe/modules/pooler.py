from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask

from .bcos_core import AloeMultiHeadAttentionPooler, NoBiasDetachableLayerNorm, select_bcos_unnormed_linear
from .layers import AloeMLP

if TYPE_CHECKING:
    from ..configuration_aloe_vision import AloeVisionConfig


def build_aloe_pooler(config: AloeVisionConfig) -> nn.Module | None:
    """
    Build the pooler head from ``config.aloe_pooler_type``.

    Returns ``None`` for ``"none"`` (backbone-only use, no pooling needed).
    """
    pt = config.aloe_pooler_type
    if pt == "none":
        return None
    if pt == "cls_token":
        return AloeClsTokenPooler(config)
    if pt == "multihead_attention":
        return AloeSiglip2MultiheadAttentionPoolingHead(config)
    if pt == "mean":
        return AloeMeanPooler(config)
    raise ValueError(f"Unknown aloe_pooler_type: {pt!r}")


class AloeClsTokenPooler(nn.Module):
    """
    Extract the CLS token as the pooled representation.

    Layout is ``[CLS, register_*, patches]`` (HF DINOv3 order), so CLS is always
    sequence index ``0`` when ``aloe_cls_token`` is enabled.
    """

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        if not config.aloe_cls_token:
            raise ValueError("aloe_pooler_type='cls_token' requires config.aloe_cls_token=True")
        self.cls_index = 0

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Return the CLS row as the pooled embedding."""
        del output_attentions
        return hidden_state[:, self.cls_index]


class AloeMeanPooler(nn.Module):
    """Mean-pool over patch tokens, skipping CLS and register prefix."""

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        self._skip_prefix = (1 if config.aloe_cls_token else 0) + config.aloe_num_registers

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Mean over patch tokens only (strip ``[CLS, register_*]`` prefix)."""
        del output_attentions
        return hidden_state[:, self._skip_prefix :, :].mean(dim=1)


class AloeSiglip2MultiheadAttentionPoolingHead(nn.Module):
    """
    Learned-probe attention pooler using the self-contained
    :class:`AloeMultiHeadAttentionPooler` (B-cos out-projection).
    """

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        self.probe = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        _Lin = select_bcos_unnormed_linear(config)
        self.attention = AloeMultiHeadAttentionPooler(
            embedding_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            out_b=config.aloe_b_linear,
            out_proj_cls=_Lin,
            attn_implementation=getattr(config, "_attn_implementation", "sdpa"),
            attention_dropout=config.attention_dropout,
        )
        self.layernorm = NoBiasDetachableLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = AloeMLP(config)
        self.num_heads = config.num_attention_heads
        self._use_flash_attention_2 = getattr(config, "_attn_implementation", None) == "flash_attention_2"
        # Skip CLS token and register tokens — probe attends over patch tokens only.
        self._skip_prefix = (1 if config.aloe_cls_token else 0) + config.aloe_num_registers

    def forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """Attention from a learned probe to patch tokens only (CLS+register prefix skipped), LN+MLP residual, first probe slot out."""
        batch_size = hidden_state.shape[0]
        probe = self.probe.expand(batch_size, -1, -1)

        # Drop CLS and register tokens; probe only attends over patch tokens.
        patch_state = hidden_state[:, self._skip_prefix:, :]

        attn_mask = None
        if attention_mask is not None:
            patch_attention_mask = attention_mask[:, self._skip_prefix:]
            if self._use_flash_attention_2 and not output_attentions:
                attn_mask = patch_attention_mask
            else:
                attn_mask = _prepare_4d_attention_mask(
                    patch_attention_mask,
                    patch_state.dtype,
                    tgt_len=probe.shape[1],
                )

        hidden_state = self.attention(
            probe, patch_state, patch_state, attn_mask=attn_mask, output_attentions=output_attentions
        )[0]
        residual = hidden_state
        hidden_state = self.layernorm(hidden_state)
        hidden_state = residual + self.mlp(hidden_state)

        return hidden_state[:, 0]
