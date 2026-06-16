from typing import Any, Literal, Optional

from pydantic import BaseModel

class ModelConfig(BaseModel):
    name: str
    is_bcos: bool = False
    interpolate_teacher_image_size: Optional[int] = None
    add_hidden_layer_forward_hooks: bool = False
    load_pretrained_weights: Optional[bool] = None
    hf_vision_init_checkpoint: Optional[str] = None
    load_weights_from_local_checkpoints: Optional[list[dict[str, Any]]] = None
    backbone_checkpoint: Optional[str] = None
    linear_probe_checkpoint: Optional[str] = None
    use_bcos_head: bool = True

    
class HFModelConfig(ModelConfig):
    model_part: str = "vision"
    # Set True when ``name`` is a native ALOE repo (``trust_remote_code`` on ``AutoModel``).
    trust_remote_code: bool = False

    
class VitConfig(BaseModel):
    feature_dim: int = 768
    patch_size: int = 16
    register_tokens: int = 0
    cls_token: bool = False
    layers: int = 12
    num_labels: Optional[int] = None
    aloe_use_logit_layer: bool = True
    attention_target_modules: Optional[list[str]] = None
    # Square resize used in training / HF image processor (may exceed base checkpoint ``image_size``).
    override_resolution: Optional[int] = None
    # Native ALOE Hub only: override ``config.hidden_act`` (e.g. ``"identity"`` = no GELU in MLP; nonlinearity from B-cos linear only).
    hidden_act: Optional[str] = None
    # Native ALOE only: ``v2`` = ``BcosUnnormedLinear_v2`` / ``BcosUnnormedConv2d_v2`` forward; omit to use Hub ``config.json``.
    aloe_bcos_impl: Optional[Literal["v1", "v2"]] = None

class HFDinov3Config(VitConfig, HFModelConfig):
    type: Literal["dinov3"] = "dinov3"
    feature_dim: int
    patch_size: int
    register_tokens: int = 4
    cls_token: bool = True
    # DINOv3ViTModel.forward never populates hidden_states; forward is monkey-patched instead.
    add_hidden_layer_forward_hooks: bool = True
    attention_target_modules: Optional[list[str]] = ["values", "queries", "keys"]

class Siglip2Config(VitConfig, HFModelConfig):
    type: Literal["siglip2"] = "siglip2"
    # SigLIP2 vision returns hidden_states when asked; hooks off unless you need synthesis.
    attention_target_modules: Optional[list[str]] = ["values", "queries", "keys"]

class GoogleVitConfig(VitConfig, HFModelConfig):
    type: Literal["google_vit"] = "google_vit"
    add_hidden_layer_forward_hooks: bool = True
    attention_target_modules: Optional[list[str]] = ["query", "value", "key"]
