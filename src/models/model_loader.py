"""
Canonical ALOE model loading utilities.

This module is the single source of truth for:
- composing Hydra configs for native ALOE backbones/experiments
- converting composed configs into ModelFactoryConfig
- loading (model, image_processor, config) bundles from the HF-native ModelFactory
- loading native ALOE models directly from HF Hub or local `save_pretrained` dirs

The LLaVA-MORE bridge (``aloe_model_loader.py``) should delegate here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


_MODEL_TYPE_TO_EXPERIMENT: dict[str, str] = {
    "distilled": "eval/embeddings/native_distilled",
    "baseline": "eval/embeddings/native_hf",
    "native_hf": "eval/embeddings/native_hf",
    "native_distilled": "eval/embeddings/native_distilled",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _configs_dir() -> str:
    return str(_repo_root() / "configs")


def _resolve_model_experiment(model_type: str) -> str:
    return _MODEL_TYPE_TO_EXPERIMENT.get(model_type, f"eval/embeddings/{model_type}")


def _compose_hydra_config(config_name: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from src.utils.omegaconf_resolvers import register_aloe_omegaconf_resolvers

    register_aloe_omegaconf_resolvers()
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=_configs_dir(), version_base=None):
            return compose(config_name=config_name, overrides=overrides)
    finally:
        GlobalHydra.instance().clear()


def _target_model_config_node(cfg):
    return cfg.model_factory.target_model_config


@dataclass
class AloeModelBundle:
    model: Any
    image_processor: Any
    model_factory_config: Any


def compose_aloe_model_config(backbone_name: str, model_type: str = "native_distilled"):
    return _compose_hydra_config(
        config_name="eval",
        overrides=[
            f"experiment={_resolve_model_experiment(model_type)}",
            f"model/backbone@backbone.student={backbone_name}",
        ],
    )


def load_backbone_image_processor(name: str, override_resolution: Optional[int] = None):
    from transformers import AutoImageProcessor

    if override_resolution:
        size = {"height": override_resolution, "width": override_resolution}
        return AutoImageProcessor.from_pretrained(
            name,
            use_fast=True,
            do_resize=True,
            size=size,
            crop_size=size,
            do_center_crop=True,
        )
    return AutoImageProcessor.from_pretrained(name, use_fast=True, do_resize=True)


def build_model_factory_config(cfg):
    import hydra
    from omegaconf import OmegaConf
    from src.models.model_factory.config import ModelFactoryConfig

    model_factory_cfg = cfg.model_factory
    target_model_config = hydra.utils.instantiate(_target_model_config_node(cfg))
    mf_dict: dict[str, Any] = {"target_model_config": target_model_config}

    for field_name in ModelFactoryConfig.model_fields:
        if field_name == "target_model_config" or field_name not in model_factory_cfg:
            continue
        value = model_factory_cfg[field_name]
        if OmegaConf.is_config(value):
            mf_dict[field_name] = OmegaConf.to_container(value, resolve=True)
        else:
            mf_dict[field_name] = value

    return ModelFactoryConfig(**mf_dict)


def get_aloe_model_bundle(
    backbone_name: str,
    model_type: str = "native_distilled",
    device: str = "cuda",
    compile_model: bool = False,
    use_gradient_checkpointing: bool = False,
) -> AloeModelBundle:
    from src.models.model_factory.model_factory import ModelFactory

    cfg = compose_aloe_model_config(backbone_name, model_type)
    target_cfg_node = _target_model_config_node(cfg)
    override_res = target_cfg_node.get("override_resolution", None)

    model_factory_config = build_model_factory_config(cfg)
    model_factory_config.compile = compile_model
    model_factory_config.use_gradient_checkpointing = use_gradient_checkpointing

    target_cfg = model_factory_config.target_model_config
    image_processor = load_backbone_image_processor(target_cfg.name, override_res)

    factory = ModelFactory(model_factory_config)
    model = factory.create_model()
    model.eval()
    model.to(device)

    return AloeModelBundle(model=model, image_processor=image_processor, model_factory_config=model_factory_config)


def load_aloe_native_model(
    model_path: str | Path,
    *,
    backbone: str | None = None,
    device: str = "cpu",
    torch_dtype: Any = None,
    trust_remote_code: bool = False,
) -> AloeModelBundle:
    from src.models.hf.aloe_models import load_aloe_native_model as _load

    model, processor = _load(
        model_path,
        backbone=backbone,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    return AloeModelBundle(model=model, image_processor=processor, model_factory_config=None)


__all__ = [
    "AloeModelBundle",
    "build_model_factory_config",
    "compose_aloe_model_config",
    "get_aloe_model_bundle",
    "load_aloe_native_model",
    "load_backbone_image_processor",
]
