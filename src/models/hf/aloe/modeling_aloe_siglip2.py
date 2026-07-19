# ALOE SigLIP2 — only the SigLIP2-specific outer forward signature and
# pooler-head init live here.  Embeddings, encoder, and transformer spine
# are all shared via AloeVisionTransformer + AloeVisionEmbeddings.

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_aloe_vision import AloeSiglip2VisionConfig
from .modeling_aloe_base import AloePreTrainedVisionModel, AloeVisionTransformer
from .modules.pooler import AloeSiglip2MultiheadAttentionPoolingHead


def _init_siglip2_attention_pooling_head(module: AloeSiglip2MultiheadAttentionPoolingHead) -> None:
    """Xavier-init the SigLIP2 pooler without replacing loaded view-backed weights."""
    if not getattr(module.probe, "_is_hf_initialized", False):
        init.xavier_uniform_(module.probe)
    init.xavier_uniform_(module.attention.q_proj.weight)
    if not getattr(module.attention.kv_proj.weight, "_is_hf_initialized", False):
        kv_w = module.attention.kv_proj.weight.view(
            2,
            module.attention.embedding_dim,
            module.attention.embedding_dim,
        )
        for i in range(2):
            init.xavier_uniform_(kv_w[i])
    init.xavier_uniform_(module.attention.out_proj.linear.weight)


class AloeSiglip2VisionModel(AloePreTrainedVisionModel):
    """
    ALOE SigLIP2 vision model (native B-cos modules, no BcosConverter).

    Architecture: learned absolute position embeddings (``aloe_position_embedding_type``
    ``"absolute"`` = standard 1-D patch table; use ``"naflex_2d"`` only for NaFlex
    resizing), no CLS token, attention-pooling head.
    """

    config_class = AloeSiglip2VisionConfig
    base_model_prefix = "siglip2"
    _no_split_modules = ["AloeVisionEmbeddings", "AloeEncoderLayer",
                         "AloeSiglip2MultiheadAttentionPoolingHead"]

    def __init__(self, config: AloeSiglip2VisionConfig) -> None:
        """Attach shared :class:`AloeVisionTransformer` (SigLIP2 pooler type comes from config)."""
        super().__init__(config)
        self.vision_model = AloeVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Patch stem / projection used as HF ``get_input_embeddings``."""
        return self.vision_model.embeddings.patch_embedding

    def _init_aloe_submodules(self, module: nn.Module) -> bool:
        """Handle SigLIP2 MHSA pooler; delegate other ALOE blocks to the base."""
        if isinstance(module, AloeSiglip2MultiheadAttentionPoolingHead):
            _init_siglip2_attention_pooling_head(module)
            return True
        return super()._init_aloe_submodules(module)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_attention_mask: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        """Delegate to ``vision_model`` (embed → encode → pool)."""
        return self.vision_model(
            pixel_values,
            attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

