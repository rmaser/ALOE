from functools import partial
from torchvision.models import vision_transformer
import transformers
from lxt.efficient.patches import patch_method, non_linear_forward, layer_norm_forward, cp_multi_head_attention_forward
from src.models.hf.aloe.modules.bcos_core import (
    AloeMultiHeadAttentionPooler,
    DetachableGELU,
    DetachableGELUApprox,
    DetachableLayerNorm,
)
from src.models.hf.aloe.modules.layers import AloeAttention
from .aloe import aloe_attn_forward, aloe_pooler_attn_forward
from .dinov3 import dinov3_attn_forward
from .siglip2 import siglip2_attn_forward
from .google_vit import google_vit_attn_forward
import torch

 # AttnLRP outside the attention mechanism & CP-LRP inside the attention mechnism is easier to tune for gamma
dinov3_map = {
    vision_transformer.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    vision_transformer.nn.LayerNorm: partial(patch_method, layer_norm_forward, keep_original=True),
    transformers.models.dinov3_vit.modeling_dinov3_vit.DINOv3ViTAttention: partial(patch_method, dinov3_attn_forward, keep_original=True),
}

siglip2_map = {
    vision_transformer.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    vision_transformer.nn.LayerNorm: partial(patch_method, layer_norm_forward, keep_original=True),
    vision_transformer.nn.MultiheadAttention: partial(patch_method, cp_multi_head_attention_forward, keep_original=True),
    transformers.models.siglip2.modeling_siglip2.Siglip2Attention: partial(patch_method, siglip2_attn_forward, keep_original=True),
    transformers.models.siglip.modeling_siglip.SiglipAttention: partial(patch_method, siglip2_attn_forward, keep_original=True),
    torch.nn.MultiheadAttention: partial(patch_method, cp_multi_head_attention_forward, keep_original=True),
}

google_vit_map = {
    vision_transformer.nn.GELU: partial(patch_method, non_linear_forward, keep_original=True),
    vision_transformer.nn.LayerNorm: partial(patch_method, layer_norm_forward, keep_original=True),
    transformers.models.vit.modeling_vit.ViTSelfAttention: partial(patch_method, google_vit_attn_forward, keep_original=True),
}

aloe_map = {
    DetachableGELU: partial(patch_method, non_linear_forward, keep_original=True),
    DetachableGELUApprox: partial(patch_method, non_linear_forward, keep_original=True),
    DetachableLayerNorm: partial(patch_method, layer_norm_forward, keep_original=True),
    AloeAttention: partial(patch_method, aloe_attn_forward, keep_original=True),
    AloeMultiHeadAttentionPooler: partial(patch_method, aloe_pooler_attn_forward, keep_original=True),
}
