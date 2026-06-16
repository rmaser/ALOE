# ALOE standard ViT — config binding only; architecture lives in the shared base.

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .configuration_aloe_vision import AloeViTVisionConfig
from .modeling_aloe_base import AloePreTrainedVisionModel, AloeVisionTransformer


class AloeViTVisionModel(AloePreTrainedVisionModel):
    """
    ALOE standard ViT vision model (native B-cos modules, no BcosConverter).

    Architecture: CLS-token ViT, 1-D absolute position embeddings,
    16×16 patches, 224 px images.
    """

    config_class = AloeViTVisionConfig
    base_model_prefix = "vit"
    _no_split_modules = ["AloeVisionEmbeddings", "AloeEncoderLayer"]

    def __init__(self, config: AloeViTVisionConfig) -> None:
        """Attach shared :class:`AloeVisionTransformer` (1-D absolute pos + CLS pool)."""
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
        """Delegate to ``vision_model`` (embed → encode → pool)."""
        return self.vision_model(
            pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
