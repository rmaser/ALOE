"""
Load **standard Hugging Face** vision checkpoints into a native ALOE vision model.

``map_factory_weights`` expects BcosConverter-style keys (``.linear.weight`` on linears).
HF checkpoints use plain ``nn.Linear`` / ``nn.Conv2d`` parameter names; this module
normalizes those keys, then reuses ``map_factory_weights`` (DRY).

**Init checkpoint** should be a public HF model (e.g. ``google/siglip2-base-patch16-224``),
not an ALOE Hub repo (those weights are already native — use ``load_pretrained_weights=True``).

**SigLIP2:** HF multi-head-attention pooler uses fused ``in_proj``; ALOE uses split Q/KV B-cos
linears. Factory init keeps and converts ``head.*`` so ``pooler_output`` matches the legacy
converter path and distillation starts from the real attention pooler instead of random head weights.

**Google ViT** (``ViTModel``, ``google/vit-*``): encoder keys are promoted to ``*.linear.*`` for
``backbone=\"vit\"`` (same ``map_factory_weights`` path as DINOv2-style BERT layout).
"""

from __future__ import annotations

import re
from typing import Any, Literal

import torch
from loguru import logger

from src.models.hf.aloe.weight_mapper import (
    _reshape_conv_to_linear,
    map_factory_weights,
)

NativeAloeBackbone = Literal["siglip2", "dinov3", "vit"]

_COMPILE_PREFIX = "_orig_mod."
# DDP / Lightning wrappers that appear before ``vision_model.`` / ``_orig_mod.`` in ``state_dict`` keys.
_TRAINING_STATE_DICT_PREFIXES: tuple[str, ...] = ("module.", "model.", "teacher_model.")


def _strip_checkpoint_wrappers_from_key(key: str) -> str:
    """
    Canonicalize ``state_dict`` keys for matching (compile, DDP, Lightning).

    Repeatedly strips ``torch.compile`` ``_orig_mod.``, then one of ``module.`` (DDP),
    ``model.`` / ``teacher_model.`` (common Lightning attribute names), until stable.
    """
    k = key
    while True:
        before = k
        while k.startswith(_COMPILE_PREFIX):
            k = k[len(_COMPILE_PREFIX) :]
        for p in _TRAINING_STATE_DICT_PREFIXES:
            if k.startswith(p):
                k = k[len(p) :]
                break
        if k == before:
            break
    return k


def _siglip2_teacher_has_native_style_head_for_reg(sub: dict[str, torch.Tensor]) -> bool:
    """
    Whether ``head.*`` in a SigLIP2 subtree matches native ALOE MHSA pooler keys.

    Public HF ``Siglip2MultiheadAttentionPoolingHead`` uses ``nn.MultiheadAttention`` (``in_proj_*``);
    that layout is converted to native ``q_proj`` / ``kv_proj`` in
    :func:`_convert_hf_siglip2_attention_pooling_head_to_native_keys` for teacher reg only.

    Native ALOE / Hub repos expose ``head.probe``, ``q_proj`` / ``kv_proj``, ``layernorm``, or split k/v.
    """
    if not any(k.startswith("head.") for k in sub):
        return False
    if "head.probe" in sub:
        return True
    if "head.attention.in_proj_weight" in sub:
        return True
    if "head.attention.kv_proj.weight" in sub:
        return True
    if "head.attention.q_proj.weight" in sub:
        return True
    if "head.layernorm.weight" in sub:
        return True
    return "head.attention.k_proj.weight" in sub and "head.attention.v_proj.weight" in sub


def _unwrap_compiled(module: torch.nn.Module) -> torch.nn.Module:
    """Strip ``torch.compile`` wrappers so we see bare parameter names and submodule attrs."""
    while hasattr(module, "_orig_mod"):
        module = module._orig_mod
    return module


def _strip_compile_prefix(key: str) -> str:
    """Strip compile / DDP / Lightning prefixes; see :func:`_strip_checkpoint_wrappers_from_key`."""
    return _strip_checkpoint_wrappers_from_key(key)


def align_mapped_state_dict_to_student_layout(
    mapped_sd: dict[str, torch.Tensor],
    student: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    """
    ``map_factory_weights`` always emits keys under ``vision_model.*`` (native Hub / full class).

    ModelFactory with ``model_part=\"vision\"`` returns the **bare** trunk
    (:class:`~.modeling_aloe_base.AloeVisionTransformer`), whose parameters live at
    ``embeddings.*`` / ``encoder.*`` with **no** prefix — same asymmetry broke
    earlier mapped-teacher utilities, not ``torch.compile``.

    When ``student`` has no ``vision_model`` child, strip the prefix so keys match
    ``student.state_dict()``.
    """
    m = _unwrap_compiled(student)
    vm = getattr(m, "vision_model", None)
    if isinstance(vm, torch.nn.Module):
        return mapped_sd
    prefix = "vision_model."
    out: dict[str, torch.Tensor] = {}
    for k, v in mapped_sd.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


def infer_native_aloe_backbone(target_model_config: Any) -> NativeAloeBackbone | None:
    """Map ``ModelConfig.type`` to ``map_factory_weights`` backbone name."""
    t = getattr(target_model_config, "type", None)
    if t == "siglip2":
        return "siglip2"
    if t == "dinov3":
        return "dinov3"
    if t in ("vit", "google_vit"):
        return "vit"
    return None


def extract_siglip2_vision_subtree(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Keep only ``vision_model.*`` tensors and strip the prefix (Siglip2Model checkpoints)."""
    prefix = "vision_model."
    if not any(k.startswith(prefix) for k in state_dict):
        return dict(state_dict)
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def normalize_hf_siglip2_vision_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    drop_head: bool = True,
) -> dict[str, torch.Tensor]:
    """
    HF SigLIP2 vision (``Siglip2VisionTransformer``) → keys understood by ``_map_siglip2``.

    - Strips ``vision_model.`` when present.
    - By default, drops ``head.*`` so callers can opt into encoder-only normalization when desired.
      Factory init and mapped-teacher utilities use ``drop_head=False`` and convert the HF pooler into
      native key names so ``pooler_output`` stays aligned with the student head.
    - Promotes plain Linear parameters to ``*.linear.{weight,bias}`` as expected by the mapper.
    """
    sd = extract_siglip2_vision_subtree(state_dict)
    if drop_head:
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}

    def promote_linear_block(old_base: str, *, has_bias: bool = True) -> None:
        wkey = f"{old_base}.weight"
        if wkey in sd:
            sd[f"{old_base}.linear.weight"] = sd.pop(wkey)
        if has_bias and f"{old_base}.bias" in sd:
            sd[f"{old_base}.linear.bias"] = sd.pop(f"{old_base}.bias")

    promote_linear_block("embeddings.patch_embedding")

    for k in list(sd.keys()):
        m = re.match(
            r"^(encoder\.layers\.\d+\.self_attn\.(?:q|k|v|out)_proj)\.(weight|bias)$",
            k,
        )
        if m:
            base, param = m.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)
        m2 = re.match(r"^(encoder\.layers\.\d+\.mlp\.fc[12])\.(weight|bias)$", k)
        if m2:
            base, param = m2.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)

    return sd


def _convert_hf_siglip2_attention_pooling_head_to_native_keys(sd: dict[str, torch.Tensor]) -> None:
    """
    Map HuggingFace ``Siglip2MultiheadAttentionPoolingHead`` weights into native ALOE key names.

    HF uses ``nn.MultiheadAttention`` with fused ``in_proj_weight`` ``[3D, D]`` (Q, K, V blocks);
    native :class:`~.modules.bcos_core.AloeMultiHeadAttentionPooler` uses ``q_proj`` and ``kv_proj``.
    Also promotes plain ``out_proj`` / head MLP linears to ``*.linear.*`` and drops biases native
    modules do not use.

    Mutates ``sd`` in place. Safe to call when ``drop_head=False`` for mapped-teacher use; no-op
    if ``in_proj_weight`` is absent (already-native checkpoint).
    """
    in_w_key = "head.attention.in_proj_weight"
    if in_w_key in sd:
        if "head.attention.q_proj.weight" in sd:
            logger.warning(
                "SigLIP2 teacher-reg: both {} and head.attention.q_proj.weight present; dropping in_proj.",
                in_w_key,
            )
            sd.pop(in_w_key, None)
            sd.pop("head.attention.in_proj_bias", None)
        else:
            W = sd.pop(in_w_key)
            if W.dim() != 2 or W.shape[0] % 3 != 0 or W.shape[0] // 3 != W.shape[1]:
                logger.warning(
                    "SigLIP2 teacher-reg: cannot split {} (shape {}); restoring key.",
                    in_w_key,
                    tuple(W.shape),
                )
                sd[in_w_key] = W
            else:
                d_model = W.shape[1]
                sd["head.attention.q_proj.weight"] = W[:d_model].clone()
                sd["head.attention.kv_proj.weight"] = torch.cat(
                    [W[d_model : 2 * d_model], W[2 * d_model : 3 * d_model]], dim=0
                ).clone()
    sd.pop("head.attention.in_proj_bias", None)

    ow, olw = "head.attention.out_proj.weight", "head.attention.out_proj.linear.weight"
    if ow in sd and olw not in sd:
        sd[olw] = sd.pop(ow)
    sd.pop("head.attention.out_proj.bias", None)

    for base in ("head.mlp.fc1", "head.mlp.fc2"):
        wk, lw = f"{base}.weight", f"{base}.linear.weight"
        if wk in sd and lw not in sd:
            sd[lw] = sd.pop(wk)
        sd.pop(f"{base}.bias", None)

    sd.pop("head.layernorm.bias", None)


def normalize_hf_dinov3_vit_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    HF ``DINOv3ViTModel`` (top-level ``embeddings`` / ``layer.*``) → mapper-native layout.

    - Promotes ``layer.*.attention|mlp`` plain Linear weights to ``*.linear.weight``.
    - Maps Conv2d ``embeddings.patch_embeddings.weight`` → ``embeddings.patch_embedding.linear.weight``
      (flattened like the existing mapper).
    """
    sd = dict(state_dict)

    pw = "embeddings.patch_embeddings.weight"
    if pw in sd:
        w = sd.pop(pw)
        sd["embeddings.patch_embedding.linear.weight"] = (
            _reshape_conv_to_linear(w) if w.dim() == 4 else w
        )
    pb = "embeddings.patch_embeddings.bias"
    if pb in sd:
        sd.pop(pb, None)

    for k in list(sd.keys()):
        m = re.match(
            r"^(layer\.\d+\.attention\.(?:q|k|v|o)_proj)\.(weight|bias)$",
            k,
        )
        if m:
            base, param = m.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)
        m2 = re.match(r"^(layer\.\d+\.mlp\.(?:up|down)_proj)\.(weight|bias)$", k)
        if m2:
            base, param = m2.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)

    return sd


def normalize_hf_google_vit_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    HF ``ViTModel`` (``google/vit-*``) → keys understood by ``_map_dinov3_or_vit(..., backbone=\"vit\")``.

    Promotes plain ``Linear`` / patch ``Conv2d`` parameters to ``*.linear.{weight,bias}``.
    Drops ``pooler.*`` (vision transfer only uses the encoder).
    """
    sd = dict(state_dict)

    if sd and all(k.startswith("vit.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items()}

    for k in list(sd.keys()):
        if k.startswith("pooler."):
            sd.pop(k)

    pw = "embeddings.patch_embeddings.projection.weight"
    if pw in sd:
        w = sd.pop(pw)
        sd["embeddings.patch_embeddings.projection.linear.weight"] = w
    pb = "embeddings.patch_embeddings.projection.bias"
    if pb in sd:
        sd.pop(pb)

    for k in list(sd.keys()):
        m = re.match(
            r"^(encoder\.layer\.\d+\.attention\.attention\.(?:query|key|value))\.(weight|bias)$",
            k,
        )
        if m:
            base, param = m.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)

    for k in list(sd.keys()):
        m = re.match(
            r"^(encoder\.layer\.\d+\.attention\.output\.dense)\.(weight|bias)$",
            k,
        )
        if m:
            base, param = m.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)

    for k in list(sd.keys()):
        m = re.match(
            r"^(encoder\.layer\.\d+\.intermediate\.dense)\.(weight|bias)$",
            k,
        )
        if m:
            base, param = m.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)

    for k in list(sd.keys()):
        m = re.match(
            r"^(encoder\.layer\.\d+\.output\.dense)\.(weight|bias)$",
            k,
        )
        if m:
            base, param = m.groups()
            sd[f"{base}.linear.{param}"] = sd.pop(k)

    return sd


def normalize_hf_vision_state_dict_for_factory(
    state_dict: dict[str, torch.Tensor],
    backbone: NativeAloeBackbone,
) -> dict[str, torch.Tensor]:
    if backbone == "siglip2":
        normalized = normalize_hf_siglip2_vision_state_dict(state_dict, drop_head=False)
        _convert_hf_siglip2_attention_pooling_head_to_native_keys(normalized)
        return normalized
    if backbone == "dinov3":
        return normalize_hf_dinov3_vit_state_dict(state_dict)
    if backbone == "vit":
        return normalize_hf_google_vit_state_dict(state_dict)
    raise NotImplementedError(
        f"HF→native bridge for backbone {backbone!r} is not implemented "
        "(use load_pretrained_weights=True or a mapped checkpoint)."
    )


def map_hf_vision_state_dict_to_native_model(
    model: torch.nn.Module,
    raw_state_dict: dict[str, torch.Tensor],
    *,
    backbone: NativeAloeBackbone,
) -> None:
    """
    Normalize HF keys, run ``map_factory_weights``, align layout, then ``load_state_dict`` (strict=False).

    Tensor keys that match the model by name but with a **different shape** (e.g. vanilla HF patch
    ``[D, 768]`` vs native conv-stem patch linear ``[D, 512]``) are **omitted** so PyTorch does not
    raise: conv-stem and patch-proj stay at init in that case.
    """
    normalized = normalize_hf_vision_state_dict_for_factory(raw_state_dict, backbone)
    mapped_sd, report = map_factory_weights(normalized, model, backbone=backbone)
    mapped_sd = align_mapped_state_dict_to_student_layout(mapped_sd, model)
    from src.models.model_factory.utils import (
        drop_shape_mismatched_against_model,
        load_state_dict_into_model,
    )

    mapped_sd = drop_shape_mismatched_against_model(model, mapped_sd)
    load_state_dict_into_model(model, mapped_sd, raise_on_unexpected_keys=False)
    report.log()
    logger.info("HF vision → native ALOE weight transfer applied (backbone={})", backbone)


def infer_native_aloe_backbone_from_student_module(module: torch.nn.Module) -> NativeAloeBackbone | None:
    """
    Infer ``map_factory_weights`` backbone from a **native ALOE** vision module's HF config.

    Returns ``None`` for vanilla HF students or unknown ``model_type``.
    Handles ``torch.compile``-wrapped modules transparently.
    """
    module = _unwrap_compiled(module)
    cfg = getattr(module, "config", None)
    if cfg is None:
        return None
    mt = str(getattr(cfg, "model_type", "") or "")
    if mt in (
        "aloe_siglip2_vision",
        "aloe_siglip2_image_classification",
    ):
        return "siglip2"
    if mt in (
        "aloe_dinov3_vision",
        "aloe_dinov3_image_classification",
    ):
        return "dinov3"
    if "aloe_vit" in mt:
        return "vit"
    return None


def teacher_tensors_in_student_key_space(
    teacher: torch.nn.Module,
    student: torch.nn.Module,
    *,
    backbone: NativeAloeBackbone,
) -> dict[str, torch.Tensor]:
    """
    Same HF-normalize + ``map_factory_weights`` path as init transfer, but return a dict of
    **teacher** tensors keyed like the student ``state_dict``.

    Only keys present in both the mapped dict and ``student.state_dict()`` are kept.
    ``backbone`` must be ``\"siglip2\"``, ``\"dinov3\"``, or ``\"vit\"``.

    **SigLIP2 head:** HF checkpoint init still drops ``head.*`` (see
    :func:`normalize_hf_siglip2_vision_state_dict`). For mapped-teacher use, ``head.*`` is kept
    when the subtree looks like a SigLIP2 pooler (``head.probe``, HF ``in_proj_weight``, native
    ``q_proj`` / ``kv_proj``, etc.); HF ``in_proj`` is split into ``q_proj`` + ``kv_proj`` before
    :func:`~src.models.hf.aloe.weight_mapper.map_factory_weights` so keys match the native student.

    Handles ``torch.compile``-wrapped modules and common training prefixes (``module.``,
    ``model.``, ``teacher_model.``) on ``state_dict`` keys so teacher tensors align with the student.
    """
    if backbone not in ("siglip2", "dinov3", "vit"):
        raise ValueError(
            f"teacher_tensors_in_student_key_space expected backbone siglip2|dinov3|vit; got {backbone!r}"
        )
    _unwrap_compiled(teacher)
    student_inner = _unwrap_compiled(student)

    raw: dict[str, torch.Tensor] = {}
    for sk, tensor in teacher.state_dict().items():
        ck = _strip_checkpoint_wrappers_from_key(sk)
        if ck in raw:
            logger.warning(
                "teacher_tensors_in_student_key_space: duplicate key after canonicalizing "
                "{!r} → {!r}; keeping the last entry.",
                sk,
                ck,
            )
        raw[ck] = tensor
    if backbone == "siglip2":
        sub = extract_siglip2_vision_subtree(raw)
        drop_head = not _siglip2_teacher_has_native_style_head_for_reg(sub)
        normalized = normalize_hf_siglip2_vision_state_dict(raw, drop_head=drop_head)
        if not drop_head:
            _convert_hf_siglip2_attention_pooling_head_to_native_keys(normalized)
    else:
        normalized = normalize_hf_vision_state_dict_for_factory(raw, backbone)
    mapped_sd, _report = map_factory_weights(normalized, student_inner, backbone=backbone)
    mapped_sd = align_mapped_state_dict_to_student_layout(mapped_sd, student_inner)
    student_keys = {_strip_checkpoint_wrappers_from_key(k) for k in student.state_dict().keys()}
    return {k: v.detach() for k, v in mapped_sd.items() if k in student_keys}
