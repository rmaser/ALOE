from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .attention_utils import eager_attention_forward

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:

    class GradientCheckpointingLayer(nn.Module):  # type: ignore[no-redef]
        pass

from .bcos_core import (
    DetachableLayerNorm,
    DetachableModule,
    NoBias,
    build_detachable_activation,
    select_bcos_unnormed_linear,
)

if TYPE_CHECKING:
    from ..configuration_aloe_vision import AloeVisionConfig


class AloeAttention(DetachableModule):
    """
    B-cos multi-head self-attention shared across all ALOE backbones.

    Q/K/V are plain linear projections (no B-cos); the module itself extends
    :class:`DetachableModule` so that q and k are detached in explanation mode,
    making the attention pattern a frozen dynamic weight.  Only the output
    projection is B-cos so contribution maps flow back linearly.
    """

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim}, num_heads={self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.is_causal = False

        # Fused QKV: one (D, 3D) GEMM instead of three (D, D) GEMMs.
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)
        _Lin = select_bcos_unnormed_linear(config)
        self.out_proj = _Lin(self.embed_dim, self.embed_dim, b=config.aloe_b_linear)

        # Cache at init — avoids a dict lookup on every forward call.
        if config._attn_implementation == "eager":
            self._attention_fn: Callable[..., Any] = eager_attention_forward
        else:
            self._attention_fn = ALL_ATTENTION_FUNCTIONS[config._attn_implementation]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Fused QKV → heads → SDPA/eager → merge → B-cos output projection."""
        batch_size, seq_length, embed_dim = hidden_states.shape

        # Single fused projection, then split into views (no copy).
        q, k, v = self.qkv_proj(hidden_states).split(self.embed_dim, dim=-1)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        if self.detach:
            q = q.detach()
            k = k.detach()

        if position_embeddings is not None:
            from .rope import apply_rotary_pos_emb
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        attn_output, attn_weights = self._attention_fn(
            self, q, k, v, attention_mask,
            is_causal=self.is_causal,
            scaling=self.scale,
            dropout=0.0 if not self.training else self.dropout,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, (attn_weights if output_attentions else None)


class AloeMLP(nn.Module):
    """B-cos two-layer MLP (fc1 → activation → fc2) shared across all ALOE backbones."""

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        act_name = getattr(config, "hidden_act", "gelu")
        self.activation_fn = build_detachable_activation(act_name)
        _Lin = select_bcos_unnormed_linear(config)
        self.fc1 = _Lin(config.hidden_size, config.intermediate_size, b=config.aloe_b_linear)
        self.fc2 = _Lin(config.intermediate_size, config.hidden_size, b=config.aloe_b_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand with B-cos fc1, activation, project back with B-cos fc2."""
        return self.fc2(self.activation_fn(self.fc1(x)))


class LayerScale(nn.Module):
    """
    Per-channel learnable scalar multiplier applied to a residual branch output
    before it is added back to the main stream.

    Matches DINOv2/DINOv3 ``Dinov2LayerScale``: stores the scalar vector as
    ``self.lambda1`` so that checkpoint keys are compatible::

        encoder.layers.{N}.layer_scale1.lambda1
        encoder.layers.{N}.layer_scale2.lambda1
    """

    def __init__(self, dim: int, init_value: float = 1.0) -> None:
        super().__init__()
        self.lambda1 = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.lambda1


class AloeEncoderLayer(GradientCheckpointingLayer):
    """Pre-norm B-cos transformer block shared across all ALOE backbones.

    When ``config.aloe_use_layer_scale`` is ``True`` (default for DINOv3),
    per-channel :class:`LayerScale` multipliers are applied to the attention
    and MLP residual branches, exactly mirroring ``Dinov2LayerScale``.
    """

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = NoBias(DetachableLayerNorm)(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = AloeAttention(config)
        self.layer_norm2 = NoBias(DetachableLayerNorm)(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = AloeMLP(config)

        use_ls = getattr(config, "aloe_use_layer_scale", False)
        ls_init = getattr(config, "aloe_layer_scale_init", 1.0)
        if use_ls:
            self.layer_scale1 = LayerScale(self.embed_dim, ls_init)
            self.layer_scale2 = LayerScale(self.embed_dim, ls_init)
        else:
            self.layer_scale1 = None
            self.layer_scale2 = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: bool = False,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
    ) -> tuple:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
        )
        if self.layer_scale1 is not None:
            hidden_states = self.layer_scale1(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if self.layer_scale2 is not None:
            hidden_states = self.layer_scale2(hidden_states)
        hidden_states = residual + hidden_states

        out: list[Any] = [hidden_states]
        if output_attentions:
            out.append(attn_weights)
        return tuple(out)


class AloeEncoder(nn.Module):
    """Stacked :class:`AloeEncoderLayer` blocks shared across all ALOE backbones."""

    def __init__(self, config: AloeVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([AloeEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        position_embeddings: Optional[tuple[Tensor, Tensor]] = None,
    ) -> Any:
        from transformers.modeling_outputs import BaseModelOutput

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Use lists during accumulation — tuple + tuple is O(N²) copies over all layers.
        encoder_states: list | None = [] if output_hidden_states else None
        all_attentions: list | None = [] if output_attentions else None
        hidden_states = inputs_embeds

        for layer in self.layers:
            if output_hidden_states:
                encoder_states.append(hidden_states)
            layer_out = layer(
                hidden_states, attention_mask,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_out[0]
            if output_attentions:
                all_attentions.append(layer_out[1])

        if output_hidden_states:
            encoder_states.append(hidden_states)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(encoder_states) if encoder_states is not None else None,
            attentions=tuple(all_attentions) if all_attentions is not None else None,
        )
