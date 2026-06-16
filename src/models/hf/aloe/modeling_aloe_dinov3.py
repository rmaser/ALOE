# ALOE DINOv3 — config binding only; architecture lives in the shared base.

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_aloe_vision import AloeDinoV3VisionConfig
from .modeling_aloe_base import AloePreTrainedVisionModel, AloeVisionTransformer


class AloeDinoV3VisionModel(AloePreTrainedVisionModel):
    """
    ALOE DINOv3 vision model (native B-cos modules, no BcosConverter at load time).

    **Legacy reference** (ModelFactory + BcosConverter + HF ``DINOv3ViTModel``), for weight /
    parity debugging — native uses different *class names* and a fused QKV block, but the
    stem norm/act and block math are aligned with this tree:

    * ``embeddings.patch_embeddings`` → :class:`~.modules.conv_stem.AloeConvStem` (``stem``:
      ``BcosConv2d_unnormed`` → ``GNLayerNormUncentered2dNoBias`` → ``ReLU`` per stage) plus
      ``Rearrange`` + ``BcosUnnormedLinear`` patch projection → ``AloeVisionEmbeddings``.
    * ``rope_embeddings`` → :class:`~.modules.embeddings.RoPE2DPositionEncoding` (2-D RoPE in
      attention; ``aloe_rope_theta`` vs HF ``rope_theta`` must match).
    * ``layer.*`` → ``AloeEncoder`` / ``AloeEncoderLayer``: pre-norm,
      ``DetachableLayerNormNoBias``-style norms, ``DINOv3ViTLayerScale`` → ``LayerScale``;
      MLP ``up_proj`` / ``down_proj`` / ``DetachableGELU`` → ``AloeMLP`` (``fc1`` / ``fc2``).
    * Attention: legacy **separate** ``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj`` → native
      fused ``qkv_proj`` + ``out_proj`` (see ``weight_mapper._fuse_qkv`` — load order Q, K, V).
    * Final ``norm`` → ``AloeVisionTransformer.post_layernorm`` (``NoBiasDetachableLayerNorm``).

    Pooling: CLS at index 0 → :class:`~.modules.pooler.AloeClsTokenPooler`.
    """

    config_class = AloeDinoV3VisionConfig
    base_model_prefix = "dinov3"
    _no_split_modules = ["AloeVisionEmbeddings", "AloeEncoderLayer"]

    def __init__(self, config: AloeDinoV3VisionConfig) -> None:
        """Attach shared :class:`AloeVisionTransformer` (RoPE + CLS pooler from config)."""
        super().__init__(config)
        self.vision_model = AloeVisionTransformer(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Patch stem / projection used as HF ``get_input_embeddings``."""
        return self.vision_model.embeddings.patch_embedding

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPooling:
        """Delegate to ``vision_model`` (embed → encode → CLS pool)."""
        return self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )


