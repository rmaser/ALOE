# Shared ALOE vision model base (B-cos, HF PreTrainedModel).
# All B-cos / explanation dependencies come from the self-contained aloe/modules/.

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .modules.bcos_core import NoBiasDetachableLayerNorm
from .modules.explanation import BcosUtilMixin


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def patch_size_from_aloe_config(config: Any) -> int:
    """Effective patch size in pixels: ``aloe_patch_size`` if set, else first element of ``patch_size``."""
    ps = getattr(config, "aloe_patch_size", None)
    if ps is not None:
        return int(ps)
    p = config.patch_size
    if isinstance(p, (list, tuple)):
        return int(p[0])
    return int(p)


def extend_attention_mask_for_registers(
    attention_mask: torch.Tensor | None,
    num_register_tokens: int,
    device: torch.device,
    *,
    num_leading_cls_tokens: int = 0,
) -> torch.Tensor | None:
    """Prepend ones for ``[CLS][registers]`` prefix when the mask covers patch tokens only."""
    num_prefix = int(num_leading_cls_tokens) + int(num_register_tokens)
    if attention_mask is None or num_prefix == 0:
        return attention_mask
    if attention_mask.dim() != 2:
        return attention_mask
    prefix = torch.ones(attention_mask.shape[0], num_prefix, device=device, dtype=attention_mask.dtype)
    return torch.cat([prefix, attention_mask], dim=1)


# ---------------------------------------------------------------------------
# Shared transformer spine  (encode → post-norm → pool)
# ---------------------------------------------------------------------------

class AloeVisionTransformerBase(nn.Module):
    """
    Shared transformer spine: vision-prefix mask extension → encoder loop →
    post-LayerNorm → pooler.

    Subclasses provide:
    * ``self.embeddings`` — module called with backbone-specific inputs,
      returning ``(B, T, D)`` hidden states already including registers.
    * ``self.encoder``  — :class:`AloeSiglip2Encoder` or equivalent.
    * Call ``_build_standard_components(config)`` from ``__init__`` to
      construct ``post_layernorm`` and ``head`` from config.

    The backbone-specific ``forward`` just calls
    ``self.embed(...)`` then ``self.forward_from_embeddings(...)``.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        """Store config and whether the encoder runs FlashAttention-2 (skips 4D mask prep)."""
        super().__init__()
        self.config = config
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def _build_standard_components(self, config: PretrainedConfig) -> None:
        """Build post_layernorm and head from config. Call at end of subclass __init__."""
        from .modules.pooler import build_aloe_pooler

        self.post_layernorm = NoBiasDetachableLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head: nn.Module | None = build_aloe_pooler(config)

    def forward_from_embeddings(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        position_embeddings: Optional[tuple] = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        """
        Core forward: extend mask for ``[CLS][registers]`` prefix → 4D mask → encoder →
        post-norm → pool.  Called by every backbone's ``forward``.

        ``position_embeddings`` is ``(cos, sin)`` for RoPE backbones (DINOv3)
        or ``None`` for additive position embedding backbones (SigLIP2, ViT).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        n_cls = 1 if getattr(self.config, "aloe_cls_token", False) else 0
        extended_mask = extend_attention_mask_for_registers(
            attention_mask,
            self.config.aloe_num_registers,
            hidden_states.device,
            num_leading_cls_tokens=n_cls,
        )

        if extended_mask is not None and not self._use_flash_attention_2:
            encoder_attn_mask = _prepare_4d_attention_mask(extended_mask, hidden_states.dtype)
        else:
            encoder_attn_mask = extended_mask

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=encoder_attn_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        last_hidden_state = self.post_layernorm(encoder_outputs.last_hidden_state)
        pooler_output = (
            self.head(last_hidden_state, extended_mask, output_attentions=output_attentions)
            if self.head is not None
            else None
        )

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Concrete shared transformer (used by all ALOE backbones)
# ---------------------------------------------------------------------------

class AloeVisionTransformer(AloeVisionTransformerBase):
    """
    Concrete ViT-style transformer shared by all ALOE backbones.

    Embedding behaviour (CLS/registers/position type) is driven entirely by
    ``config``; no backbone-specific subclass is needed.

    ``forward`` separates *embedding* kwargs (e.g. ``spatial_shapes`` when
    ``aloe_position_embedding_type="naflex_2d"``) from encoder kwargs so they never bleed into the encoder.
    """

    def __init__(self, config: PretrainedConfig) -> None:
        """Wire embeddings, encoder stack, post-norm, and pooler from ``config``."""
        super().__init__(config)
        from .modules.embeddings import AloeVisionEmbeddings
        from .modules.layers import AloeEncoder

        self.embeddings = AloeVisionEmbeddings(config)
        self.encoder = AloeEncoder(config)
        self._build_standard_components(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **embed_kwargs: Any,          # e.g. spatial_shapes — consumed by embeddings only
    ) -> "BaseModelOutputWithPooling":
        """Patchify + position encoding, optional RoPE tables, then encoder + pooler."""
        hidden_states = self.embeddings(pixel_values, **embed_kwargs)

        # Compute 2-D RoPE tables when the backbone uses rope_2d position encoding.
        # Only possible for standard-image inputs (dim==4); pre-patchified inputs
        # dim==3 (pre-patchified) inputs never use RoPE so position_embeddings stays None.
        position_embeddings = None
        if pixel_values.dim() == 4:
            _, _, H, W = pixel_values.shape
            ps = patch_size_from_aloe_config(self.config)
            position_embeddings = self.embeddings.get_position_embeddings(
                H // ps, W // ps, pixel_values.device, hidden_states.dtype
            )

        return self.forward_from_embeddings(
            hidden_states, attention_mask, output_attentions, output_hidden_states,
            position_embeddings=position_embeddings,
        )


# ---------------------------------------------------------------------------
# HF PreTrainedModel base  (BcosUtilMixin + PreTrainedModel)
# ---------------------------------------------------------------------------

class AloePreTrainedVisionModel(BcosUtilMixin, PreTrainedModel):
    """
    HF-compatible base for ALOE vision models.

    Subclasses set ``config_class``, build ``self.vision_model``
    (a :class:`AloeVisionTransformerBase` subclass), and override
    ``_init_aloe_submodules`` for backbone-specific weight init.
    """

    is_bcos_model: bool = True
    supports_gradient_checkpointing: bool = True
    main_input_name = "pixel_values"
    input_modalities = ("image",)
    base_model_prefix = "aloe_vision"

    def __init__(self, config: PretrainedConfig) -> None:
        """HF ``PreTrainedModel`` init; marks the module as B-cos for tooling."""
        super().__init__(config)
        self.is_bcos_model = True

    def _init_aloe_submodules(self, module: nn.Module) -> bool:
        """
        Backbone-specific weight init hook. Return ``True`` if *module* was
        fully initialised here so the generic fallback in ``_init_weights`` is
        skipped.  Subclasses override this to handle pooler heads etc.
        The shared ALOE attention, MLP, and embedding modules are handled here.
        """
        from .modules.embeddings import AloeVisionEmbeddings
        from .modules.layers import AloeAttention, AloeMLP

        if isinstance(module, AloeVisionEmbeddings):
            module.reset_parameters()
            return True
        if isinstance(module, AloeAttention):
            # Init each QKV block independently (same variance as 3 separate xavier inits).
            w = module.qkv_proj.weight.view(3, module.embed_dim, module.embed_dim)
            for i in range(3):
                init.xavier_uniform_(w[i])
            init.xavier_uniform_(module.out_proj.linear.weight)
            return True
        if isinstance(module, AloeMLP):
            init.xavier_uniform_(module.fc1.linear.weight)
            init.xavier_uniform_(module.fc2.linear.weight)
            return True
        return False

    @torch.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        """HF hook: ALOE-specific init first, then generic Linear/LayerNorm/Embedding."""
        if self._init_aloe_submodules(module):
            return
        if isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=module.embedding_dim**-0.5)
        elif isinstance(module, nn.Linear):
            _lecun = getattr(init, "lecun_normal_", None)
            if _lecun is not None:
                _lecun(module.weight)
            else:
                init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                init.zeros_(module.bias)
            if module.weight is not None:
                init.ones_(module.weight)
