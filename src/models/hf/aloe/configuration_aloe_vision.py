from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from transformers.configuration_utils import PretrainedConfig
from transformers.models.dinov2 import Dinov2Config
from transformers.models.siglip2 import Siglip2VisionConfig
from transformers.models.vit import ViTConfig

# "absolute" / "absolute_1d" — learned embedding per patch index (standard; SigLIP2 default)
# "naflex_2d" — 2-D grid + bilinear resize via ``spatial_shapes`` (optional NaFlex)
# "absolute_1d"            — 1-D learned, position 0 = CLS (standard ViT)
# "rope_2d"                — 2-D Rotary Position Embedding applied inside attention (DINOv3)
PositionEmbeddingType = Literal["absolute", "naflex_2d", "absolute_1d", "rope_2d"]
# "cls_token"           – extract CLS at sequence index 0 (layout [CLS, reg*, patches])
# "multihead_attention" – learned-probe attention pooler (SigLIP2-style)
# "mean"                – mean over patch tokens, skipping CLS + registers
# "none"                – no pooler (model used as a backbone; returns last hidden state only)
PoolerType = Literal["cls_token", "multihead_attention", "mean", "none"]
BackboneName = Literal["siglip2", "dinov3", "vit"]
AloeBcosImpl = Literal["v1", "v2"]


def _validate_aloe_bcos_impl(value: Any) -> None:
    """Reject invalid implementations before cooperative HF config dispatch."""
    if value not in ("v1", "v2"):
        raise ValueError(f"aloe_bcos_impl must be 'v1' or 'v2', got {value!r}")


class AloeVisionConfig(PretrainedConfig):
    """
    Shared ALOE vision settings (B-cos hyperparameters, registers, pooler, position mode).

    Used cooperatively in the MRO with backbone configs (e.g. ``Siglip2VisionConfig``):
    the backbone config passes unknown keyword arguments through so this class can pop
    ``aloe_*`` fields before ``PretrainedConfig`` runs.
    """

    model_type = "aloe_vision"

    def __init__(self, **kwargs: Any) -> None:
        """Parse all ``aloe_*`` fields, then delegate to ``PretrainedConfig``."""
        self.aloe_patch_size: int = int(kwargs.pop("aloe_patch_size", 16))
        self.aloe_in_channels: int = int(kwargs.pop("aloe_in_channels", 6))
        self.aloe_feature_dim: Optional[int] = kwargs.pop("aloe_feature_dim", None)
        self.aloe_cls_token: bool = bool(kwargs.pop("aloe_cls_token", False))
        self.aloe_num_registers: int = int(kwargs.pop("aloe_num_registers", 0))
        self.aloe_b_conv: float = float(kwargs.pop("aloe_b_conv", 2.0))
        self.aloe_b_linear: float = float(kwargs.pop("aloe_b_linear", 2.0))
        self.aloe_add_conv_stem: Optional[list[int]] = kwargs.pop("aloe_add_conv_stem", None)
        self.aloe_to_patch_embedding_key: Optional[str] = kwargs.pop("aloe_to_patch_embedding_key", None)
        self.aloe_position_embedding_type: PositionEmbeddingType = kwargs.pop(
            "aloe_position_embedding_type", "absolute"
        )
        self.aloe_pooler_type: PoolerType = kwargs.pop("aloe_pooler_type", "multihead_attention")
        self.aloe_backbone: BackboneName = kwargs.pop("aloe_backbone", "siglip2")
        # RoPE base frequency; only used when aloe_position_embedding_type="rope_2d".
        # Prefer explicit aloe_rope_theta; else ``rope_theta`` from the HF backbone
        # (DINOv3 ViT uses 100, not 10k).  Do not pop rope_theta — parent configs need it.
        _rope_backbone = kwargs.get("rope_theta")
        _aloe_rope = kwargs.pop("aloe_rope_theta", None)
        if _aloe_rope is not None:
            _theta = float(_aloe_rope)
        elif _rope_backbone is not None:
            _theta = float(_rope_backbone)
        else:
            _theta = 10000.0
        # Legacy exports: aloe_rope_theta stuck at default 10k while merged HF config had rope_theta=100.
        if (
            _theta == 10000.0
            and _rope_backbone is not None
            and float(_rope_backbone) != 10000.0
            and self.aloe_position_embedding_type == "rope_2d"
        ):
            _theta = float(_rope_backbone)
        self.aloe_rope_theta: float = _theta
        # LayerScale: per-channel learned multipliers on attn and MLP residuals.
        # Enabled by default for DINOv3; off for SigLIP2/ViT.
        self.aloe_use_layer_scale: bool = bool(kwargs.pop("aloe_use_layer_scale", False))
        self.aloe_layer_scale_init: float = float(kwargs.pop("aloe_layer_scale_init", 1.0))
        # Original HF model name used as base (e.g. "google/siglip2-base-patch16-224").
        # Stored so eval tooling can auto-load the correct image processor without
        # a separate backbone config entry.
        self.aloe_base_model_name: Optional[str] = kwargs.pop("aloe_base_model_name", None)
        self.aloe_use_logit_layer: bool = bool(kwargs.pop("aloe_use_logit_layer", True))
        _bcos_impl = kwargs.pop("aloe_bcos_impl", "v1")
        if _bcos_impl not in ("v1", "v2"):
            raise ValueError(f"aloe_bcos_impl must be 'v1' or 'v2', got {_bcos_impl!r}")
        self.aloe_bcos_impl: AloeBcosImpl = _bcos_impl  # type: ignore[assignment]

        super().__init__(**kwargs)

        self.num_channels = self.aloe_in_channels


# ---------------------------------------------------------------------------
# Helpers shared by all _aloe_kw builders below
# ---------------------------------------------------------------------------

def _aloe_kw(
    aloe_patch_size, aloe_in_channels, aloe_feature_dim, aloe_cls_token,
    aloe_num_registers, aloe_b_conv, aloe_b_linear, aloe_add_conv_stem,
    aloe_to_patch_embedding_key, aloe_position_embedding_type,
    aloe_pooler_type, aloe_backbone, aloe_rope_theta,
    aloe_use_layer_scale=False, aloe_layer_scale_init=1.0,
    aloe_base_model_name=None,
    aloe_bcos_impl: AloeBcosImpl = "v1",
) -> dict:
    """Collect ``aloe_*`` constructor args into one dict for backbone ``super().__init__`` merges."""
    return dict(
        aloe_patch_size=aloe_patch_size,
        aloe_in_channels=aloe_in_channels,
        aloe_feature_dim=aloe_feature_dim,
        aloe_cls_token=aloe_cls_token,
        aloe_num_registers=aloe_num_registers,
        aloe_b_conv=aloe_b_conv,
        aloe_b_linear=aloe_b_linear,
        aloe_add_conv_stem=aloe_add_conv_stem,
        aloe_to_patch_embedding_key=aloe_to_patch_embedding_key,
        aloe_position_embedding_type=aloe_position_embedding_type,
        aloe_pooler_type=aloe_pooler_type,
        aloe_backbone=aloe_backbone,
        aloe_rope_theta=aloe_rope_theta,
        aloe_use_layer_scale=aloe_use_layer_scale,
        aloe_layer_scale_init=aloe_layer_scale_init,
        aloe_base_model_name=aloe_base_model_name,
        aloe_bcos_impl=aloe_bcos_impl,
    )


# ---------------------------------------------------------------------------
# from_pretrained / from_backbone_pretrained helpers (DRY across vision + IC configs)
# ---------------------------------------------------------------------------


def _split_trust_remote_code(kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Copy kwargs, pop ``trust_remote_code`` (default True), return ``(kw, trust)``."""
    kw = dict(kwargs)
    trust = bool(kw.pop("trust_remote_code", True))
    return kw, trust


def _vision_config_from_pretrained_with_num_labels(
    cls: type,
    pretrained_model_name_or_path: str,
    *,
    num_labels: int,
    **kwargs: Any,
) -> Any:
    """Load a vision config from Hub/local and set ``num_labels`` for classifier-ready configs."""
    kw, trust = _split_trust_remote_code(kwargs)
    cfg = cls.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust, **kw)
    cfg.num_labels = int(num_labels)
    return cfg



def _ic_config_from_vision_backbone(
    ic_cls: type,
    vision_cls: type,
    pretrained_model_name_or_path: str,
    *,
    num_labels: int,
    **kwargs: Any,
) -> Any:
    """Build an image-classification *config* from a vision backbone repo.

    This function is *config-only*: it loads the backbone vision config and
    constructs the supervised config class (changes ``model_type`` + sets
    ``num_labels``). It does *not* load any model weights.
    """
    kw, trust = _split_trust_remote_code(kwargs)
    base = vision_cls.from_backbone_pretrained(
        pretrained_model_name_or_path, num_labels=num_labels, trust_remote_code=trust, **kw
    )
    d = base.to_dict()
    d.pop("model_type", None)
    return ic_cls(num_labels=num_labels, **d)


# ---------------------------------------------------------------------------
# SigLIP2
# ---------------------------------------------------------------------------

class AloeSiglip2VisionConfig(Siglip2VisionConfig, AloeVisionConfig):
    """SigLIP2 vision config plus all :class:`AloeVisionConfig` fields."""

    model_type = "aloe_siglip2_vision"
    # ``Siglip2VisionConfig`` sets ``base_config_key = "vision_config"`` for use inside
    # ``Siglip2Config``.  Our checkpoints are *standalone* vision repos: the real
    # parameters (``aloe_*``, ``auto_map``, ``model_type``) live at the top level of
    # ``config.json``.  If we kept the parent ``base_config_key``, ``from_pretrained``
    # would replace the dict with the nested ``vision_config`` only — dropping
    # ``auto_map`` and breaking ``AutoModel.from_pretrained(..., trust_remote_code=True)``.
    base_config_key = ""

    def __init__(
        self,
        aloe_patch_size: int = 16,
        aloe_in_channels: int = 6,
        aloe_feature_dim: Optional[int] = None,
        aloe_cls_token: bool = False,
        aloe_num_registers: int = 0,
        aloe_b_conv: float = 2.0,
        aloe_b_linear: float = 2.0,
        aloe_add_conv_stem: Optional[list[int]] = None,
        aloe_to_patch_embedding_key: Optional[str] = None,
        aloe_position_embedding_type: PositionEmbeddingType = "absolute",
        aloe_pooler_type: PoolerType = "multihead_attention",
        aloe_backbone: BackboneName = "siglip2",
        aloe_rope_theta: float = 10000.0,
        aloe_use_layer_scale: bool = False,
        aloe_layer_scale_init: float = 1.0,
        aloe_base_model_name: Optional[str] = None,
        aloe_bcos_impl: AloeBcosImpl = "v2",
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        num_patches: int = 256,
        patch_size: int = 16,
        hidden_act: str = "gelu_pytorch_tanh",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _validate_aloe_bcos_impl(aloe_bcos_impl)
        merged = {**_aloe_kw(
            aloe_patch_size, aloe_in_channels, aloe_feature_dim, aloe_cls_token,
            aloe_num_registers, aloe_b_conv, aloe_b_linear, aloe_add_conv_stem,
            aloe_to_patch_embedding_key, aloe_position_embedding_type,
            aloe_pooler_type, aloe_backbone, aloe_rope_theta,
            aloe_use_layer_scale=aloe_use_layer_scale,
            aloe_layer_scale_init=aloe_layer_scale_init,
            aloe_base_model_name=aloe_base_model_name,
            aloe_bcos_impl=aloe_bcos_impl,
        ), **kwargs}
        if "num_labels" not in merged:
            merged["num_labels"] = 1000
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=num_channels,
            num_patches=num_patches,
            patch_size=patch_size,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **merged,
        )
        if self.aloe_feature_dim is None:
            self.aloe_feature_dim = self.hidden_size
        if self.aloe_position_embedding_type not in ("absolute", "absolute_1d", "naflex_2d"):
            raise ValueError(
                f"SigLIP2 requires aloe_position_embedding_type 'absolute', 'absolute_1d', "
                f"or 'naflex_2d' (got {self.aloe_position_embedding_type!r})."
            )
        if self.aloe_backbone != "siglip2":
            raise ValueError(
                f"AloeSiglip2VisionConfig requires aloe_backbone='siglip2' (got {self.aloe_backbone!r})."
            )
        self.num_channels = self.aloe_in_channels

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeSiglip2VisionConfig":
        """Create a supervised-ready vision config from a backbone repo.

        Config-only: weights are loaded later by the model class'
        ``from_pretrained`` / checkpoint-loading path.
        """
        return _vision_config_from_pretrained_with_num_labels(
            cls, pretrained_model_name_or_path, num_labels=num_labels, **kwargs
        )


# ---------------------------------------------------------------------------
# DINOv3
# ---------------------------------------------------------------------------

class AloeDinoV3VisionConfig(Dinov2Config, AloeVisionConfig):
    """
    DINOv2/v3 vision config plus all :class:`AloeVisionConfig` fields.

    Defaults: 14×14 patches, 518 px images, CLS-token pooling,
    optional registers (via ``aloe_num_registers``).
    """

    model_type = "aloe_dinov3_vision"

    def __init__(
        self,
        aloe_patch_size: int = 16,
        aloe_in_channels: int = 6,
        aloe_feature_dim: Optional[int] = None,
        aloe_cls_token: bool = True,
        aloe_num_registers: int = 4,
        aloe_b_conv: float = 2.0,
        aloe_b_linear: float = 2.0,
        aloe_add_conv_stem: Optional[list[int]] = None,
        aloe_to_patch_embedding_key: Optional[str] = None,
        aloe_position_embedding_type: PositionEmbeddingType = "rope_2d",
        aloe_pooler_type: PoolerType = "cls_token",
        aloe_backbone: BackboneName = "dinov3",
        aloe_rope_theta: float = 100.0,
        aloe_use_layer_scale: bool = True,
        aloe_layer_scale_init: float = 1.0,
        aloe_base_model_name: Optional[str] = None, 
        aloe_bcos_impl: AloeBcosImpl = "v2",
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _validate_aloe_bcos_impl(aloe_bcos_impl)
        merged = {**_aloe_kw(
            aloe_patch_size, aloe_in_channels, aloe_feature_dim, aloe_cls_token,
            aloe_num_registers, aloe_b_conv, aloe_b_linear, aloe_add_conv_stem,
            aloe_to_patch_embedding_key, aloe_position_embedding_type,
            aloe_pooler_type, aloe_backbone, aloe_rope_theta,
            aloe_use_layer_scale=aloe_use_layer_scale,
            aloe_layer_scale_init=aloe_layer_scale_init,
            aloe_base_model_name=aloe_base_model_name,
            aloe_bcos_impl=aloe_bcos_impl,
        ), **kwargs}
        if "num_labels" not in merged:
            merged["num_labels"] = 1000
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout,
            **merged,
        )
        if self.aloe_feature_dim is None:
            self.aloe_feature_dim = self.hidden_size
        if self.aloe_backbone != "dinov3":
            raise ValueError(
                f"AloeDinoV3VisionConfig requires aloe_backbone='dinov3' (got {self.aloe_backbone!r})."
            )
        self.num_channels = self.aloe_in_channels
        self.num_patches = (self.image_size // self.aloe_patch_size) ** 2

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeDinoV3VisionConfig":
        """Create a supervised-ready vision config from a DINOv3 backbone repo.

        Config-only: weights are loaded later by the model class'
        ``from_pretrained`` / checkpoint-loading path.
        """
        return _vision_config_from_pretrained_with_num_labels(
            cls, pretrained_model_name_or_path, num_labels=num_labels, **kwargs
        )


# ---------------------------------------------------------------------------
# Standard ViT (Google ViT / ViT-B/16 family)
# ---------------------------------------------------------------------------

class AloeViTVisionConfig(ViTConfig, AloeVisionConfig):
    """
    Standard ViT config plus all :class:`AloeVisionConfig` fields.

    Defaults: 16×16 patches, 224 px images, CLS-token pooling.
    ``attention_dropout`` is mapped from ViT's ``attention_probs_dropout_prob``.
    """

    model_type = "aloe_vit_vision"

    def __init__(
        self,
        aloe_patch_size: int = 16,
        aloe_in_channels: int = 6,
        aloe_feature_dim: Optional[int] = None,
        aloe_cls_token: bool = True,
        aloe_num_registers: int = 0,
        aloe_b_conv: float = 2.0,
        aloe_b_linear: float = 2.0,
        aloe_add_conv_stem: Optional[list[int]] = None,
        aloe_to_patch_embedding_key: Optional[str] = None,
        aloe_position_embedding_type: PositionEmbeddingType = "absolute_1d",
        aloe_pooler_type: PoolerType = "cls_token",
        aloe_backbone: BackboneName = "vit",
        aloe_rope_theta: float = 10000.0,
        aloe_use_layer_scale: bool = False,
        aloe_layer_scale_init: float = 1.0,
        aloe_base_model_name: Optional[str] = None,
        aloe_bcos_impl: AloeBcosImpl = "v1",
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        attention_probs_dropout_prob: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _validate_aloe_bcos_impl(aloe_bcos_impl)
        merged = {**_aloe_kw(
            aloe_patch_size, aloe_in_channels, aloe_feature_dim, aloe_cls_token,
            aloe_num_registers, aloe_b_conv, aloe_b_linear, aloe_add_conv_stem,
            aloe_to_patch_embedding_key, aloe_position_embedding_type,
            aloe_pooler_type, aloe_backbone, aloe_rope_theta,
            aloe_use_layer_scale=aloe_use_layer_scale,
            aloe_layer_scale_init=aloe_layer_scale_init,
            aloe_base_model_name=aloe_base_model_name,
            aloe_bcos_impl=aloe_bcos_impl,
        ), **kwargs}
        if "num_labels" not in merged:
            merged["num_labels"] = 1000
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            **merged,
        )
        if self.aloe_feature_dim is None:
            self.aloe_feature_dim = self.hidden_size
        if self.aloe_backbone != "vit":
            raise ValueError(
                f"AloeViTVisionConfig requires aloe_backbone='vit' (got {self.aloe_backbone!r})."
            )
        self.num_channels = self.aloe_in_channels
        self.num_patches = (self.image_size // self.aloe_patch_size) ** 2
        # Bridge ViT naming → ALOE canonical name used by AloeAttention
        self.attention_dropout: float = self.attention_probs_dropout_prob

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeViTVisionConfig":
        """Create a supervised-ready vision config from a ViT backbone repo.

        Config-only: weights are loaded later by the model class'
        ``from_pretrained`` / checkpoint-loading path.
        """
        return _vision_config_from_pretrained_with_num_labels(
            cls, pretrained_model_name_or_path, num_labels=num_labels, **kwargs
        )


# ---------------------------------------------------------------------------
# Image classification (supervised head; ``explain`` uses logits)
# ---------------------------------------------------------------------------


def _pop_image_classification_logit_kwargs(kwargs: dict[str, Any]) -> tuple[bool, Optional[float], Optional[float]]:
    """Strip head kwargs before ``AloeVisionConfig`` / backbone ``__init__``.

    ``aloe_use_logit_layer`` enables :class:`~src.modules.logit_layer.LogitLayer` inside the
    cross-entropy **loss** and in ``Aloe*ForImageClassification.explain``; ``forward`` returns
    raw classifier logits in its ``logits`` field.
    """
    use = bool(kwargs.pop("aloe_use_logit_layer", True))
    bias = kwargs.pop("aloe_logit_bias", None)
    temperature = kwargs.pop("aloe_logit_temperature", None)
    if bias is not None:
        bias = float(bias)
    if temperature is not None:
        temperature = float(temperature)
    return use, bias, temperature


class AloeSiglip2ForImageClassificationConfig(AloeSiglip2VisionConfig):
    """Same as :class:`AloeSiglip2VisionConfig` plus ``num_labels`` for the classifier head."""

    model_type = "aloe_siglip2_image_classification"

    def __init__(self, num_labels: int = 2, **kwargs: Any) -> None:
        use_logit, logit_bias, logit_temperature = _pop_image_classification_logit_kwargs(kwargs)
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        self.aloe_use_logit_layer: bool = use_logit and self.num_labels > 1
        self.aloe_logit_bias: Optional[float] = logit_bias
        self.aloe_logit_temperature: Optional[float] = logit_temperature

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeSiglip2ForImageClassificationConfig":
        """Build a supervised (IC) *config* from a backbone repo.

        Config-only. The paired model helper
        ``AloeForImageClassificationBase.from_backbone_pretrained`` is where the
        backbone weights are actually loaded.
        """
        return _ic_config_from_vision_backbone(
            cls,
            AloeSiglip2VisionConfig,
            pretrained_model_name_or_path,
            num_labels=num_labels,
            **kwargs,
        )


class AloeDinoV3ForImageClassificationConfig(AloeDinoV3VisionConfig):
    """Same as :class:`AloeDinoV3VisionConfig` plus ``num_labels``."""

    model_type = "aloe_dinov3_image_classification"

    def __init__(self, num_labels: int = 2, **kwargs: Any) -> None:
        use_logit, logit_bias, logit_temperature = _pop_image_classification_logit_kwargs(kwargs)
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        self.aloe_use_logit_layer: bool = use_logit and self.num_labels > 1
        self.aloe_logit_bias: Optional[float] = logit_bias
        self.aloe_logit_temperature: Optional[float] = logit_temperature

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeDinoV3ForImageClassificationConfig":
        """Build a supervised (IC) *config* from a DINOv3 backbone repo.

        Config-only. The paired model helper
        ``AloeForImageClassificationBase.from_backbone_pretrained`` is where the
        backbone weights are actually loaded.
        """
        return _ic_config_from_vision_backbone(
            cls,
            AloeDinoV3VisionConfig,
            pretrained_model_name_or_path,
            num_labels=num_labels,
            **kwargs,
        )


class AloeViTForImageClassificationConfig(AloeViTVisionConfig):
    """Vision config with ``model_type`` ``aloe_vit_image_classification`` (``AutoConfig`` only)."""

    model_type = "aloe_vit_image_classification"

    def __init__(self, num_labels: int = 2, **kwargs: Any) -> None:
        use_logit, logit_bias, logit_temperature = _pop_image_classification_logit_kwargs(kwargs)
        super().__init__(**kwargs)
        self.num_labels = int(num_labels)
        self.aloe_use_logit_layer: bool = use_logit and self.num_labels > 1
        self.aloe_logit_bias: Optional[float] = logit_bias
        self.aloe_logit_temperature: Optional[float] = logit_temperature

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeViTForImageClassificationConfig":
        """Build a supervised (IC) *config* from a ViT backbone repo.

        Config-only. The paired model helper
        ``AloeForImageClassificationBase.from_backbone_pretrained`` is where the
        backbone weights are actually loaded.
        """
        return _ic_config_from_vision_backbone(
            cls,
            AloeViTVisionConfig,
            pretrained_model_name_or_path,
            num_labels=num_labels,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Hugging Face Auto* registration (runs on import — Hub ``trust_remote_code``)
# ---------------------------------------------------------------------------

_ALOE_HF_AUTOS_REGISTERED: bool = False


def _try_hf_register(register_fn: Callable[..., None], *args: Any, **kwargs: Any) -> None:
    """Invoke ``Auto*.register``; swallow duplicate-registration ``ValueError``."""
    try:
        register_fn(*args, **kwargs)
    except ValueError:
        pass


def register_aloe_with_huggingface_autos() -> None:
    """
    Register all ALOE vision configs, backbone models, image processors, and
    ``*ForImageClassification`` heads with Transformers ``AutoConfig`` /
    ``AutoModel`` / ``AutoImageProcessor`` / ``AutoModelForImageClassification``.

    Called automatically when this module is loaded so users do not need to call
    ``register_all_aloe_models()`` for standard Hub or local ``trust_remote_code``
    workflows.  Duplicate registrations are ignored.
    """
    global _ALOE_HF_AUTOS_REGISTERED
    if _ALOE_HF_AUTOS_REGISTERED:
        return
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForImageClassification
    except ImportError:
        return
    try:
        from .image_processing_aloe import AloeImageProcessor
        from .modeling_aloe_dinov3 import AloeDinoV3VisionModel
        from .modeling_aloe_for_image_classification import (
            AloeDinoV3ForImageClassification,
            AloeSiglip2ForImageClassification,
            AloeViTForImageClassification,
        )
        from .modeling_aloe_siglip2 import AloeSiglip2VisionModel
        from .modeling_aloe_vit import AloeViTVisionModel
    except ImportError:
        return

    _ALOE_HF_AUTOS_REGISTERED = True

    backbone: list[tuple[type, type]] = [
        (AloeSiglip2VisionConfig, AloeSiglip2VisionModel),
        (AloeDinoV3VisionConfig, AloeDinoV3VisionModel),
        (AloeViTVisionConfig, AloeViTVisionModel),
    ]
    for cfg_cls, model_cls in backbone:
        _try_hf_register(AutoConfig.register, cfg_cls.model_type, cfg_cls)
        _try_hf_register(AutoModel.register, cfg_cls, model_cls)
        _try_hf_register(AutoImageProcessor.register, cfg_cls, AloeImageProcessor)

    # IC-only ``model_type``s (AutoConfig). ForIC Auto mapping uses vision configs only (Transformers register rule).
    ic_config_types = (
        AloeSiglip2ForImageClassificationConfig,
        AloeDinoV3ForImageClassificationConfig,
        AloeViTForImageClassificationConfig,
    )
    for cfg_cls in ic_config_types:
        _try_hf_register(AutoConfig.register, cfg_cls.model_type, cfg_cls)

    for cfg_cls, model_cls in (
        (AloeSiglip2VisionConfig, AloeSiglip2ForImageClassification),
        (AloeDinoV3VisionConfig, AloeDinoV3ForImageClassification),
        (AloeViTVisionConfig, AloeViTForImageClassification),
    ):
        _try_hf_register(AutoModelForImageClassification.register, cfg_cls, model_cls)


register_aloe_with_huggingface_autos()
