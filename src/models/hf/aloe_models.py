"""
Unified ALOE model registry and native-format loading utilities.

Registers all three ALOE backbone configs/models with HF ``AutoConfig`` /
``AutoModel``, and image-classification models (including backbone configs) with
``AutoModelForImageClassification`` (see :mod:`src.models.hf.aloe.configuration_aloe_vision`).
Importing :mod:`src.models.hf.aloe.configuration_aloe_vision` (e.g. via Hub ``trust_remote_code``)
registers Auto classes once; :func:`register_all_aloe_models` repeats registration safely if you want it explicit.

Usage
-----
Load a model from the HF Hub or a local ``save_pretrained`` directory
without going through ModelFactory::

    from src.models.hf.aloe_models import load_aloe_native_model

    model, processor = load_aloe_native_model(
        "your-org/aloe-siglip2-base-distilled",
        backbone="siglip2",
    )

Load a **Lightning** student (ModelFactory layout) into ``Aloe*ForImageClassification``::

    from src.models.hf.aloe_models import load_aloe_for_image_classification_from_factory_checkpoint

    model, processor = load_aloe_for_image_classification_from_factory_checkpoint(
        "path/to/last.ckpt",
        backbone="siglip2",
        num_labels=1000,
        hub_model_name="google/siglip2-base-patch16-224",
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from transformers import AutoModel


# ---------------------------------------------------------------------------
# Registration helpers (all delegate to configuration_aloe_vision; idempotent)
# ---------------------------------------------------------------------------

def register_all_aloe_models() -> None:
    """Register all ALOE configs, backbone models, processors, and classifiers with HF Auto classes."""
    from src.models.hf.aloe.configuration_aloe_vision import register_aloe_with_huggingface_autos

    register_aloe_with_huggingface_autos()
    logger.debug("ALOE Hugging Face Auto registration ensured.")


def register_aloe_siglip2_with_auto() -> None:
    register_all_aloe_models()


def register_aloe_dinov3_with_auto() -> None:
    register_all_aloe_models()


def register_aloe_vit_with_auto() -> None:
    register_all_aloe_models()


def register_aloe_image_classification_with_auto() -> None:
    register_all_aloe_models()


# ---------------------------------------------------------------------------
# Native model loader (HF Hub or local save_pretrained directory)
# ---------------------------------------------------------------------------

def load_aloe_native_model(
    model_path: str | Path,
    *,
    backbone: str | None = None,
    device: str = "cpu",
    torch_dtype: Any = None,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    """
    Load a native ALOE model from the HF Hub or a local ``save_pretrained`` dir.

    This does NOT go through ModelFactory / BcosConverter — it loads via
    ``AutoModel.from_pretrained`` (with ``trust_remote_code=True`` when using
    Hub ``auto_map``). The checkpoint must have been saved with
    ``model.save_pretrained()``.

    Parameters
    ----------
    model_path :
        HF Hub repo id (e.g. ``"your-org/aloe-siglip2-base"``) or local path.
    backbone :
        Reserved for compatibility; loading uses the checkpoint ``model_type`` /
        Hugging Face ``auto_map``.
    device :
        Target device (``"cpu"``, ``"cuda"``, ``"cuda:0"`` …).
    torch_dtype :
        Optional dtype (e.g. ``torch.float16``).  ``None`` = keep saved dtype.
    trust_remote_code :
        Passed to ``from_pretrained``; needed when code is not vendored.

    Returns
    -------
    model : AloePreTrainedVisionModel
    image_processor : HF image processor
    """
    from transformers import AutoImageProcessor

    _ = backbone  # API compatibility only

    # When loading from a local exported directory the code files are embedded
    # there; trust_remote_code must be True for Auto to pick them up.
    local_path = Path(model_path)
    if local_path.exists() and (local_path / "modeling_aloe_siglip2.py").exists():
        trust_remote_code = True

    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModel.from_pretrained(str(model_path), **kwargs)
    model.eval()
    model.to(device)

    try:
        processor = AutoImageProcessor.from_pretrained(
            str(model_path), trust_remote_code=trust_remote_code
        )
    except Exception:
        processor = None
        logger.warning(f"Could not load image processor from {model_path}.")

    return model, processor


# ---------------------------------------------------------------------------
# Weight migration helpers
# ---------------------------------------------------------------------------

def _patch_size_from_pretrained_dict(cfg_dict: dict[str, Any]) -> int | None:
    raw = cfg_dict.get("patch_size")
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return int(raw[0])
    return int(raw)


def build_aloe_config_from_factory(
    model_factory_config: Any,
    *,
    backbone: str,
) -> Any:
    """
    Build a native ALOE config from a ``ModelFactoryConfig`` object.

    Parameters
    ----------
    model_factory_config :
        A ``ModelFactoryConfig`` (Pydantic model) returned by ``build_model_factory_config``.
    backbone :
        ``"siglip2"``, ``"dinov3"``, or ``"vit"``.
    """
    from transformers import AutoConfig

    from src.models.hf.aloe.configuration_aloe_vision import (
        AloeDinoV3VisionConfig,
        AloeSiglip2VisionConfig,
        AloeViTVisionConfig,
    )

    target_cfg = model_factory_config.target_model_config
    base_hf_cfg = AutoConfig.from_pretrained(target_cfg.name)
    # ``Siglip2Config`` nests the ViT in ``vision_config``.  Flat ``to_dict()`` keeps
    # ``hidden_size`` / ``num_hidden_layers`` only under that nested key, so merging
    # the parent dict into ``AloeSiglip2VisionConfig`` silently falls back to base
    # defaults (768 hidden, 12 layers) and breaks publishing for large/SO models.
    if backbone == "siglip2":
        vision_cfg = getattr(base_hf_cfg, "vision_config", None)
        if vision_cfg is not None:
            base_dict = vision_cfg.to_dict()
        else:
            base_dict = base_hf_cfg.to_dict()
        base_dict.pop("model_type", None)
    else:
        base_dict = base_hf_cfg.to_dict()

    # ModelFactory loads ``AutoModel.from_pretrained(target_cfg.name)`` and never
    # rebuilds patch geometry from ``VitConfig.patch_size`` — the running model
    # always matches HF.  Native export sets ``aloe_patch_size``, which
    # :func:`~src.models.hf.aloe.modeling_aloe_base.patch_size_from_aloe_config`
    # prefers over ``config.patch_size``, so a stale yaml value can desync export
    # from the checkpoint (e.g. DINOv3 *vits16* with yaml ``patch_size: 14``).
    yaml_patch = int(target_cfg.patch_size)
    hf_patch = _patch_size_from_pretrained_dict(base_dict)
    if hf_patch is not None and hf_patch != yaml_patch:
        logger.warning(
            f"target_model_config.patch_size={yaml_patch} differs from Hugging Face "
            f"{target_cfg.name!r} (patch_size={hf_patch}). Using HF for aloe_patch_size "
            "so native export matches ModelFactory weights."
        )
        effective_patch = hf_patch
    else:
        effective_patch = yaml_patch

    aloe_overrides: dict[str, Any] = {
        "aloe_patch_size": effective_patch,
        "aloe_in_channels": getattr(target_cfg, "in_channels", 6),
        "aloe_feature_dim": target_cfg.feature_dim,
        "aloe_cls_token": target_cfg.cls_token,
        "aloe_num_registers": target_cfg.register_tokens,
        "aloe_base_model_name": target_cfg.name,
    }
    # Training resize from backbone yaml (``VitConfig.override_resolution``).  Without this on
    # the Pydantic model, Hydra drops the field and publish loads a base-resolution processor
    # while weights expect e.g. 432 px (SO400M).
    orv = getattr(target_cfg, "override_resolution", None)
    if orv is not None:
        side = int(orv)
        if backbone == "siglip2":
            yaml_ps = int(target_cfg.patch_size)
            g: int | None = None
            if side % yaml_ps == 0:
                g = side // yaml_ps
            elif side % effective_patch == 0:
                g = side // effective_patch
                if yaml_ps != effective_patch:
                    logger.info(
                        f"override_resolution={side}: num_patches grid uses HF patch_size={effective_patch} "
                        f"(yaml has {yaml_ps}); if this mismatches training, align backbone yaml with HF."
                    )
            if g is not None:
                aloe_overrides["num_patches"] = g * g
            else:
                logger.warning(
                    f"override_resolution={side} not divisible by yaml patch_size={yaml_ps} "
                    f"nor HF patch_size={effective_patch}; leaving num_patches from Hugging Face vision config."
                )
        elif backbone in ("dinov3", "vit"):
            aloe_overrides["image_size"] = side
    elif backbone == "siglip2":
        # ``Siglip2VisionConfig`` stores the patch grid as ``num_patches`` rather
        # than deriving it from ``image_size``.  Our ALOE config default is 256
        # patches, but most local checkpoints are 224px (14x14 = 196 patches).
        # Preserve explicit ``override_resolution`` above for the SO400M 432px
        # ALOE variant; otherwise mirror the HF base config geometry.
        image_size = base_dict.get("image_size")
        if image_size is not None:
            side = int(image_size)
            if side % effective_patch == 0:
                aloe_overrides["num_patches"] = (side // effective_patch) ** 2
            else:
                logger.warning(
                    f"SigLIP2 image_size={side} from {base_model_name!r} is not divisible by "
                    f"patch_size={effective_patch}; leaving num_patches from Hugging Face vision config."
                )

    # DINOv3 (HF ``DINOv3ViTConfig``) uses ``rope_theta`` (e.g. 100).  Our RoPE tables
    # read ``aloe_rope_theta`` only — without this, exports default to 10000 and
    # break parity with ModelFactory / ``facebook/dinov3-*`` checkpoints.
    rt = base_dict.get("rope_theta")
    if rt is not None:
        aloe_overrides["aloe_rope_theta"] = float(rt)

    # B-cos v1 vs v2 forward (``BcosUnnormedLinear`` / ``BcosUnnormedConv2d`` vs *_v2).
    # Hub ``config.json`` may still say v1; Hydra ``VitConfig.aloe_bcos_impl`` must win.
    impl = getattr(target_cfg, "aloe_bcos_impl", None)
    if impl is not None:
        aloe_overrides["aloe_bcos_impl"] = impl

    cfg_cls = {
        "siglip2": AloeSiglip2VisionConfig,
        "dinov3": AloeDinoV3VisionConfig,
        "vit": AloeViTVisionConfig,
    }[backbone]

    return cfg_cls(**{**base_dict, **aloe_overrides})


def _ic_config_and_model_classes(
    backbone: str,
) -> tuple[type, type]:
    from src.models.hf.aloe.configuration_aloe_vision import (
        AloeDinoV3ForImageClassificationConfig,
        AloeSiglip2ForImageClassificationConfig,
        AloeViTForImageClassificationConfig,
    )
    from src.models.hf.aloe.modeling_aloe_for_image_classification import (
        AloeDinoV3ForImageClassification,
        AloeSiglip2ForImageClassification,
        AloeViTForImageClassification,
    )

    m = {
        "siglip2": (
            AloeSiglip2ForImageClassificationConfig,
            AloeSiglip2ForImageClassification,
        ),
        "dinov3": (
            AloeDinoV3ForImageClassificationConfig,
            AloeDinoV3ForImageClassification,
        ),
        "vit": (
            AloeViTForImageClassificationConfig,
            AloeViTForImageClassification,
        ),
    }
    if backbone not in m:
        raise ValueError(f"backbone must be 'siglip2', 'dinov3', or 'vit', got {backbone!r}")
    return m[backbone]


def _build_ic_config(
    *,
    backbone: str,
    num_labels: int,
    hub_model_name: str | None,
    model_factory_config: Any | None,
) -> Any:
    cfg_cls, _ = _ic_config_and_model_classes(backbone)
    if hub_model_name is not None:
        return cfg_cls.from_backbone_pretrained(
            hub_model_name,
            num_labels=num_labels,
            trust_remote_code=True,
        )
    if model_factory_config is not None:
        vision_cfg = build_aloe_config_from_factory(
            model_factory_config,
            backbone=backbone,
        )
        d = vision_cfg.to_dict()
        d.pop("model_type", None)
        d.pop("num_labels", None)
        return cfg_cls(num_labels=num_labels, **d)
    raise ValueError("Provide exactly one of hub_model_name or model_factory_config.")


def _state_dict_looks_like_native_ic(sd: dict[str, Any]) -> bool:
    """Weights already in ``Aloe*ForImageClassification`` layout (e.g. ``save_pretrained``)."""
    if not sd:
        return False
    has_vm = any(k.startswith("vision_model.") for k in sd)
    has_root_classifier = any(
        k.startswith("classifier.") and not k.startswith("vision_model.") for k in sd
    )
    has_factory_trunk = any(
        (k.startswith("embeddings.") or k.startswith("encoder.")) and not k.startswith("vision_model.")
        for k in sd
    )
    return has_vm and has_root_classifier and not has_factory_trunk


def load_aloe_for_image_classification_from_factory_checkpoint(
    checkpoint_path: str | Path,
    *,
    backbone: str,
    num_labels: int,
    hub_model_name: str | None = None,
    model_factory_config: Any | None = None,
    device: str = "cpu",
    torch_dtype: Any = None,
    load_image_processor: bool = True,
) -> tuple[Any, Any | None]:
    """
    Build ``Aloe*ForImageClassification`` and load a Lightning / flat student checkpoint.

    Supports:

    * **ModelFactory layout** — keys like ``embeddings.*``, ``encoder.layers.*``,
      ``classifier.linear.weight`` (as returned by
      :func:`src.models.model_factory.utils.load_state_dict_from_checkpoint`).
      Weights are mapped with :func:`src.models.hf.aloe.weight_mapper.map_factory_weights`
      and ``for_image_classification=True`` so heads stay top-level.

    * **Native layout** — keys already use ``vision_model.*`` plus top-level ``classifier.*``
      (no flat ``embeddings.*`` trunk). Loaded with ``strict=False`` (no mapper).

    Parameters
    ----------
    checkpoint_path :
        Path to a ``.ckpt`` with ``state_dict`` (e.g. ``DistillHFModel``).
    backbone :
        ``"siglip2"``, ``"dinov3"``, or ``"vit"``.
    num_labels :
        Number of classifier outputs (e.g. 1000 for ImageNet).
    hub_model_name :
        HF id for :meth:`Aloe*ForImageClassificationConfig.from_backbone_pretrained`.
    model_factory_config :
        Alternative to ``hub_model_name``: a :class:`ModelFactoryConfig` instance
        (same as :func:`build_aloe_config_from_factory`).
    device :
        Where to move the model after loading.
    torch_dtype :
        Optional dtype for model parameters.
    load_image_processor :
        If ``True`` and ``hub_model_name`` or factory target name is available,
        load :class:`transformers.AutoImageProcessor`.

    Returns
    -------
    model, processor
        ``processor`` may be ``None`` if image processor loading fails or is disabled.
    """
    import torch
    from transformers import AutoImageProcessor

    from src.models.model_factory.utils import (
        load_state_dict_from_checkpoint,
        log_unexpected_state_dict_keys,
    )
    from src.models.hf.aloe.weight_mapper import map_factory_weights

    register_all_aloe_models()

    cfg = _build_ic_config(
        backbone=backbone,
        num_labels=num_labels,
        hub_model_name=hub_model_name,
        model_factory_config=model_factory_config,
    )
    _, model_cls = _ic_config_and_model_classes(backbone)
    model = model_cls(cfg)

    sd = load_state_dict_from_checkpoint(str(checkpoint_path), verbose=False)

    if _state_dict_looks_like_native_ic(sd):
        logger.info("Checkpoint looks like native ForImageClassification layout; loading without mapper.")
        with torch.no_grad():
            missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            log_unexpected_state_dict_keys(
                list(unexpected),
                context="Native-layout ForImageClassification checkpoint",
            )
        if missing:
            logger.info(f"Missing keys (often RoPE / buffers): {len(missing)}")
    else:
        mapped_sd, report = map_factory_weights(
            sd,
            model,
            backbone=backbone,
            for_image_classification=True,
        )
        with torch.no_grad():
            missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
        report.raise_if_critical(missing, unexpected)
        if missing:
            logger.info(
                f"{len(missing)} target keys missing after factory→IC map "
                f"(expected for some buffers): {missing[:5]}"
            )

    if torch_dtype is not None:
        model.to(dtype=torch_dtype)
    model.eval()
    model.to(device)

    processor = None
    if load_image_processor:
        proc_name = hub_model_name
        if proc_name is None and model_factory_config is not None:
            proc_name = model_factory_config.target_model_config.name
        if proc_name is not None:
            try:
                processor = AutoImageProcessor.from_pretrained(
                    proc_name,
                    use_fast=True,
                    trust_remote_code=True,
                )
            except Exception as e:
                logger.warning(f"Could not load image processor from {proc_name!r}: {e}")

    return model, processor


__all__ = [
    "build_aloe_config_from_factory",
    "load_aloe_for_image_classification_from_factory_checkpoint",
    "load_aloe_native_model",
    "register_aloe_dinov3_with_auto",
    "register_aloe_image_classification_with_auto",
    "register_aloe_siglip2_with_auto",
    "register_aloe_vit_with_auto",
    "register_all_aloe_models",
]
