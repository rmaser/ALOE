from typing import Any, Literal, Optional

from pydantic import BaseModel, field_validator

from src.models.config import ModelConfig


class CheckpointLoadConfig(BaseModel):
    """Configuration for loading weights from a checkpoint."""

    path: str
    patterns: list[str] = [".*"]
    ignore_patterns: list[str] = []
    map_hf_vision_to_native_aloe: bool = False
    native_aloe_backbone: Optional[Literal["siglip2", "dinov3", "vit"]] = None
    map_hf_trust_remote_code: bool = False


class ModelFactoryConfig(BaseModel):
    """HF-native ALOE model construction and checkpoint-loading settings."""

    target_model_config: ModelConfig
    is_bcos: bool = False
    model_repo: str = "default"
    load_pretrained_weights: bool = True

    keep_pretrained_weights_trainable: list[str] = []
    freeze_pretrained_weights: bool = False
    load_weights_from_local_checkpoints: list[CheckpointLoadConfig] = []
    load_weights_from_hf_checkpoints: list[CheckpointLoadConfig] = []

    @field_validator(
        "load_weights_from_local_checkpoints",
        "load_weights_from_hf_checkpoints",
        mode="before",
    )
    @classmethod
    def _none_checkpoint_lists_to_empty(cls, v: Any) -> list[Any]:
        return [] if v is None else v

    hf_vision_init_checkpoint: Optional[str] = None
    hf_vision_init_trust_remote_code: bool = False
    hf_vision_init_backbone: Optional[Literal["siglip2", "dinov3", "vit"]] = None

    compile: bool = False
    compile_mode: str = "default"

    attn_implementation: Optional[str] = None
    use_gradient_checkpointing: bool = False

    strip_classifier: bool = False
    strip_pooler: bool = False
    add_hidden_layer_forward_hooks: bool = False
