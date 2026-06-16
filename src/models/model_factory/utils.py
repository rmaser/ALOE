import torch
import torch.nn as nn
from beartype import beartype
from typing import Dict
from loguru import logger
from .config import ModelFactoryConfig

def get_check_if_kept_trainable_fn(keep_pretrained_weights_trainable_keys: list[str]):
    def check_if_kept_trainable(key):
        if keep_pretrained_weights_trainable_keys is None:
            return False
        for keep_key in keep_pretrained_weights_trainable_keys:
            if key.startswith(keep_key):
                return True
        return False
    return check_if_kept_trainable

def initialize_vit_weights(module):
    """Initialize model weights using ViT-appropriate initialization scheme"""    
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        # Truncated normal with std=0.02 (BERT/ViT standard)
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    # Special embeddings
    elif hasattr(module, 'cls_token'):
        nn.init.trunc_normal_(module.cls_token, std=0.02)
    elif hasattr(module, 'position_embeddings'):
        nn.init.trunc_normal_(module.position_embeddings, std=0.02)
@beartype
def drop_shape_mismatched_against_model(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Drop checkpoint tensors that share a key with ``model`` but have a **different shape**.

    PyTorch raises on shape mismatch even when ``strict=False``; this is needed when a vanilla HF
    patch embedding (e.g. ``[D, 16·16·3]``) is mapped onto a native student that uses a conv stem
    (patch linear ``[D, C_stem]`` with ``C_stem ≠ 768``).
    """
    target = model.state_dict()
    out: Dict[str, torch.Tensor] = {}
    mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for k, v in state_dict.items():
        if k not in target:
            out[k] = v
            continue
        t = target[k]
        if t.shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(t.shape)))
            continue
        out[k] = v
    if mismatched:
        for k, ckpt_shape, model_shape in mismatched[:8]:
            logger.warning(
                "Omitting {}: checkpoint shape {} ≠ model {} (e.g. HF patch vs conv-stem layout)",
                k,
                ckpt_shape,
                model_shape,
            )
        if len(mismatched) > 8:
            logger.warning("… {} more shape mismatches omitted from load", len(mismatched) - 8)
    return out


def log_unexpected_state_dict_keys(unexpected_keys: list[str], *, context: str | None = None) -> None:
    """
    Log checkpoint keys that ``load_state_dict(..., strict=False)`` did not assign to the model.

    A non-empty list often means a stale checkpoint, wrong ``model_part`` / wrapper layout, or keys
    left over after filtering — worth verifying before trusting the run.
    """
    if not unexpected_keys:
        return
    sorted_keys = sorted(unexpected_keys)
    if context:
        logger.warning(
            "{} — {} unexpected checkpoint key(s) were not consumed by the model.",
            context,
            len(unexpected_keys),
        )
    else:
        logger.warning(
            "Loaded weights: {} unexpected checkpoint key(s) were not consumed by the model "
            "(checkpoint vs model layout or mapping mismatch).",
            len(unexpected_keys),
        )
    logger.warning("Unexpected keys (full list): {}", sorted_keys)


def maybe_normalize_head_state_dict_keys(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Align classifier / logit-layer keys with the target model layout."""
    target_keys = set(model.state_dict().keys())
    out = dict(state_dict)

    has_wrapped_classifier = any(k.startswith("classifier.linear.") for k in target_keys)
    has_plain_classifier = any(
        k.startswith("classifier.weight") or k.startswith("classifier.bias") for k in target_keys
    )

    def _rename(src: str, dst: str) -> None:
        if src in out and dst not in out:
            out[dst] = out.pop(src)
            logger.info("Renamed {} to {}", src, dst)

    if has_wrapped_classifier and not has_plain_classifier:
        _rename("classifier.weight", "classifier.linear.weight")
        _rename("classifier.bias", "classifier.linear.bias")
    elif has_plain_classifier and not has_wrapped_classifier:
        _rename("classifier.linear.weight", "classifier.weight")
        _rename("classifier.linear.bias", "classifier.bias")

    has_logit_layer = any(k.startswith("logit_layer.") for k in target_keys)
    if not has_logit_layer:
        dropped = [k for k in list(out) if k.startswith("logit_layer.")]
        for key in dropped:
            out.pop(key)
        if dropped:
            logger.info("Dropped {} logit-layer key(s) for plain-head target model", len(dropped))

    return out


def maybe_prefix_state_dict_with_vision_model(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Native ALOE vision classes (e.g. ``AloeSiglip2VisionModel``) store weights under ``vision_model.*``.

    Lightning checkpoints from runs where the factory returned only the inner tower have keys like
    ``embeddings.*`` / ``encoder.*`` (bare spine). Remap those to ``vision_model.*`` so the same
    ckpt loads into the full wrapper returned by :meth:`HfModelFactory.get_pretrained_HF_model`.
    """
    if not hasattr(model, "vision_model"):
        return state_dict
    model_keys = set(model.state_dict().keys())
    ck_keys = list(state_dict.keys())
    if not ck_keys:
        return state_dict
    n_hit = sum(1 for k in ck_keys if k in model_keys)
    if n_hit == len(ck_keys):
        return state_dict
    if n_hit > 0:
        return state_dict
    prefixed = [f"vision_model.{k}" for k in ck_keys]
    if all(pk in model_keys for pk in prefixed):
        logger.info(
            "Checkpoint uses bare vision-spine keys; prefixing with vision_model. "
            "to match native ALOE vision wrapper layout."
        )
        return {f"vision_model.{k}": state_dict[k] for k in ck_keys}
    return state_dict


@beartype
def load_state_dict_into_model(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    raise_on_unexpected_keys: bool = True,
) -> None:
    """Load the given state dict into the model and log loading status."""

    state_dict = dict(state_dict)
    state_dict = maybe_prefix_state_dict_with_vision_model(model, state_dict)
    state_dict = maybe_normalize_head_state_dict_keys(model, state_dict)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if logger:
        logger.info(
            f"Loaded checkpoint weights after wrapping - Missing: {len(missing_keys)}, "
            f"Unexpected: {len(unexpected_keys)}"
        )
        if missing_keys:
            logger.info(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            log_unexpected_state_dict_keys(unexpected_keys)
            if raise_on_unexpected_keys:
                raise Exception(
                    "Unexpected keys found when loading checkpoint weights after wrapping. "
                    f"Please check the model architecture and checkpoint compatibility: {unexpected_keys}"
                )
        
@beartype
def load_state_dict_from_checkpoint(checkpoint_path: str, verbose: bool = False) -> Dict[str, torch.Tensor]:
    """Load only the model state dict from a Lightning checkpoint"""
    try:
        if verbose and logger:
            logger.info(f"Loading model state dict from checkpoint: {checkpoint_path}")
        
        # Load checkpoint as dict without instantiating Lightning module
        # Use weights_only=False for Lightning checkpoints as they contain experiment configs
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        if 'state_dict' not in checkpoint:
            raise ValueError(f"Checkpoint does not contain 'state_dict' key. Available keys: {list(checkpoint.keys())}")
        
        state_dict = checkpoint['state_dict']

        def _collect_prefixed_keys(prefixes: dict[str, bool]) -> Dict[str, torch.Tensor]:
            collected: Dict[str, torch.Tensor] = {}
            for prefix, strip_prefix in prefixes.items():
                for key, value in state_dict.items():
                    if not key.startswith(prefix):
                        continue
                    new_key = key[len(prefix):] if strip_prefix else key
                    if new_key.startswith('_orig_mod.'):
                        new_key = new_key[10:]
                    collected[new_key] = value
            return collected

        # Prefer backbone/trunk checkpoints when present.
        model_state_dict = _collect_prefixed_keys({
            'model.': True,
            'student_model.': True,
        })
        if model_state_dict and verbose and logger:
            logger.info(f"Found {len(model_state_dict)} backbone weights in checkpoint")

        # LP / trained-classifier checkpoints only contain the head; keep classifier and
        # optional calibration/logit-layer weights together instead of returning the first prefix only.
        if not model_state_dict:
            model_state_dict = _collect_prefixed_keys({
                'classifier.': False,
                'logit_layer.': False,
            })
            if model_state_dict and verbose and logger:
                logger.info(f"Found {len(model_state_dict)} classifier/logit-layer weights in checkpoint")

        if not model_state_dict:
            raise ValueError(
                "No model weights found in checkpoint state dict. Checked prefixes: "
                "model., student_model., classifier., logit_layer."
            )
        
        if verbose and logger:
            logger.info(f"Extracted {len(model_state_dict)} model parameters from checkpoint")
        
        return model_state_dict
        
    except Exception as e:
        raise Exception(f"Failed to load model state dict from checkpoint {checkpoint_path}: {e}") from e


@beartype
def load_state_dict_from_hf_model(hf_model_name: str, verbose: bool = False) -> Dict[str, torch.Tensor]:
    """Load only the classifier state dict from a HuggingFace checkpoint"""
    try:
        if verbose and logger:
            logger.info(f"Loading classifier state dict from HuggingFace model: {hf_model_name}")
        
        from transformers import ViTForImageClassification
        
        hf_model = ViTForImageClassification.from_pretrained(hf_model_name)
        hf_state_dict = hf_model.state_dict()
        
        
        return hf_state_dict
        
    except Exception as e:
        raise Exception(f"Failed to load classifier state dict from HuggingFace model {hf_model_name}: {e}") from e


@beartype
def load_hf_automodel_state_dict(
    name_or_path: str,
    *,
    trust_remote_code: bool = False,
    verbose: bool = False,
) -> Dict[str, torch.Tensor]:
    """Load ``state_dict`` from ``AutoModel.from_pretrained`` (multimodal / vision HF checkpoints)."""
    if verbose and logger:
        logger.info(f"Loading AutoModel state dict from: {name_or_path}")
    try:
        from transformers import AutoModel

        kw = {}
        if trust_remote_code:
            kw["trust_remote_code"] = True
        m = AutoModel.from_pretrained(name_or_path, **kw)
        return m.state_dict()
    except Exception as e:
        raise Exception(f"Failed to load AutoModel state dict from {name_or_path!r}: {e}") from e

@beartype
def get_model_factory_config_from_checkpoint(checkpoint_path: str, verbose: bool = False) -> ModelFactoryConfig:
    """Extracts the student ModelFactoryConfig from a checkpoint."""
    try:
        if verbose and logger:
            logger.info(f"Loading ModelFactoryConfig from checkpoint: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if "hyper_parameters" not in checkpoint:
            raise ValueError("Checkpoint does not contain 'hyper_parameters'.")
            
        hparams = checkpoint["hyper_parameters"]
        
        # Check if it was saved via LightningModuleConfig (which dumps to dict)
        if "student_model_factory_config" in hparams:
            config_dict = hparams["student_model_factory_config"]
        # Fallback: maybe it was saved as 'cfg' which contains the config
        elif "cfg" in hparams and "student_model_factory_config" in hparams["cfg"]:
            config_dict = hparams["cfg"]["student_model_factory_config"]
        else:
            raise ValueError("Could not find 'student_model_factory_config' in checkpoint hyper_parameters.")
            
        return ModelFactoryConfig(**config_dict)
        
    except Exception as e:
        raise Exception(f"Failed to extract ModelFactoryConfig from {checkpoint_path}: {e}") from e
