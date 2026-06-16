"""
Weight migration utilities: BcosConverter (ModelFactory) → native ALOE HF format.

The BcosConverter produces a model whose state dict has NO ``vision_model.``
prefix and uses *separate* Q, K, V projection weights.  Mapper output uses a
``vision_model.`` prefix and *fused* ``qkv_proj`` / ``kv_proj`` weights to match
full ``Aloe*VisionModel`` checkpoints.  When ModelFactory returns the bare trunk
(``model_part=\"vision\"``), :func:`~src.models.hf.aloe.hf_vision_checkpoint_bridge.align_mapped_state_dict_to_student_layout`
strips that prefix before ``load_state_dict`` / mapped-teacher use.

Usage example
-------------
::

    factory = ModelFactory(cfg)
    src_model = factory.create_model()
    src_sd = src_model.state_dict()

    cfg_native = AloeSiglip2VisionConfig(...)
    tgt_model = AloeSiglip2VisionModel(cfg_native)

    mapped_sd, report = map_factory_weights(src_sd, tgt_model, backbone="siglip2")
    missing, unexpected = tgt_model.load_state_dict(mapped_sd, strict=False)
    report.raise_if_critical(missing, unexpected)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

if TYPE_CHECKING:
    import torch.nn as nn


# ---------------------------------------------------------------------------
# Migration report
# ---------------------------------------------------------------------------

@dataclass
class MigrationReport:
    """Summary of what the weight mapper did and what might need attention."""

    mapped: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    fused: list[str] = field(default_factory=list)
    renamed: list[str] = field(default_factory=list)

    def log(self) -> None:
        logger.info(f"Weight mapper: {len(self.mapped)} direct copies, "
                    f"{len(self.fused)} fusions, {len(self.renamed)} renames, "
                    f"{len(self.dropped)} dropped.")
        if self.dropped:
            logger.debug("Dropped source keys: " + ", ".join(self.dropped[:10]) +
                         ("…" if len(self.dropped) > 10 else ""))

    def raise_if_critical(
        self,
        missing_keys: list[str],
        unexpected_keys: list[str],
        *,
        allow_missing_rope: bool = True,
    ) -> None:
        """Raise on any missing non-optional target key."""
        # Conv-stem weights in target are always expected to be present.
        critical_missing = [
            k for k in missing_keys
            # position_embedding may be absent for RoPE models (intentionally dropped)
            if not (allow_missing_rope and "pos_embedding" in k)
        ]
        if critical_missing:
            raise ValueError(
                f"Weight migration produced {len(critical_missing)} missing keys in "
                f"the target model.  First few: {critical_missing[:5]}"
            )
        if unexpected_keys:
            from src.models.model_factory.utils import log_unexpected_state_dict_keys

            log_unexpected_state_dict_keys(
                unexpected_keys,
                context="Weight migration (factory→native)",
            )


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _unwrap_module_for_structure(m: "nn.Module") -> "nn.Module":
    """Unwrap ``torch.compile`` (``_orig_mod``) for encoder depth / attribute probing."""
    while hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def _encoder_layer_count_from_target(target_model: "nn.Module") -> int | None:
    """
    Number of encoder blocks on the **vision trunk**.

    Supports both ``Aloe*VisionModel`` (``vision_model.encoder.layers``) and the bare
    trunk returned by ModelFactory after ``model_part=\"vision\"`` (``encoder.layers`` only).
    """
    m = _unwrap_module_for_structure(target_model)
    enc = None
    if hasattr(m, "vision_model") and hasattr(m.vision_model, "encoder"):
        enc = m.vision_model.encoder
    elif hasattr(m, "encoder"):
        enc = m.encoder
    if enc is not None:
        layers = getattr(enc, "layers", None)
        if layers is not None:
            return len(layers)
    return None


def _infer_num_layers_from_source_keys(source_state_dict: dict[str, Any]) -> int:
    """Fallback when target has no ``encoder.layers``: count unique layer indices (not weight+bias pairs)."""
    pat = re.compile(
        r"^(?:encoder\.)?layer[s]?\.(\d+)\.(?:norm1|norm2|layernorm_before|layer_norm1)(?:\.|$)"
    )
    indices: set[int] = set()
    for k in source_state_dict:
        mm = pat.match(k)
        if mm:
            indices.add(int(mm.group(1)))
    return len(indices)


def _fuse_qkv(sd: dict, q_key: str, k_key: str, v_key: str) -> torch.Tensor:
    """Concatenate Q, K, V weight tensors along dim-0 → fused QKV weight."""
    q = sd.pop(q_key)
    k = sd.pop(k_key)
    v = sd.pop(v_key)
    return torch.cat([q, k, v], dim=0)


def _fuse_kv(sd: dict, k_key: str, v_key: str) -> torch.Tensor:
    """Concatenate K, V weight tensors along dim-0 → fused KV weight."""
    k = sd.pop(k_key)
    v = sd.pop(v_key)
    return torch.cat([k, v], dim=0)


def _try_pop(sd: dict, *keys: str) -> tuple[str | None, torch.Tensor | None]:
    """Return the first key that exists in sd, or (None, None)."""
    for k in keys:
        if k in sd:
            return k, sd.pop(k)
    return None, None


def _reshape_conv_to_linear(w: torch.Tensor) -> torch.Tensor:
    """Reshape Conv2d weight ``(D, C, P, P)`` → Linear weight ``(D, C·P·P)``."""
    D = w.shape[0]
    return w.reshape(D, -1)


def _map_conv_stem(
    sd: dict,
    *,
    stem_prefix: str,
    out: dict,
    rpt: MigrationReport,
) -> None:
    """
    Map source ConvStem keys (flat sequential) to the target AloeConvStem (grouped _ConvBlock).

    The BcosConverter ``ConvStem`` replaces the original patch-embedding module entirely.
    Its ``stem`` is a *flat* ``nn.Sequential`` with 3 entries per block::

        {stem_prefix}.stem.{i*3}.linear.weight    # BcosConv2d_unnormed
        {stem_prefix}.stem.{i*3+1}.weight         # NoBias(GNLayerNorm)
        {stem_prefix}.stem.{i*3+2}                # activation – no parameters

    The native ALOE target groups them as ``_ConvBlock``::

        embeddings.conv_stem.stem.{i}.conv.linear.weight
        embeddings.conv_stem.stem.{i}.norm.weight
        embeddings.conv_stem.stem.{i}.norm.bias   ← absent in source; initialized to 0

    The ``patch_projection`` inside the source ConvStem corresponds to the
    target ``embeddings.patch_embedding.linear.weight``.
    """
    stem_key_prefix = f"{stem_prefix}.stem."
    if not any(k.startswith(stem_key_prefix) and k.endswith(".linear.weight") for k in sd):
        return  # no ConvStem at this prefix

    # Flat indices where conv layers sit (they carry .linear.weight).
    flat_conv_idxs = sorted(set(
        int(k[len(stem_key_prefix):].split(".")[0])
        for k in sd
        if k.startswith(stem_key_prefix) and k.endswith(".linear.weight")
    ))

    for block_idx, flat_ci in enumerate(flat_conv_idxs):
        conv_k = f"{stem_prefix}.stem.{flat_ci}.linear.weight"
        norm_k = f"{stem_prefix}.stem.{flat_ci + 1}.weight"
        tgt_conv = f"embeddings.conv_stem.stem.{block_idx}.conv.linear.weight"
        tgt_norm = f"embeddings.conv_stem.stem.{block_idx}.norm.weight"

        if conv_k in sd:
            out[f"vision_model.{tgt_conv}"] = sd.pop(conv_k)
            rpt.renamed.append(f"{conv_k} → {tgt_conv}")
        if norm_k in sd:
            out[f"vision_model.{tgt_norm}"] = sd.pop(norm_k)
            rpt.renamed.append(f"{norm_k} → {tgt_norm}")

    # The patch_projection linear inside the ConvStem maps to the target patch embedding.
    for proj_k in (
        f"{stem_prefix}.patch_projection.linear.weight",
        f"{stem_prefix}.patch_projection.weight",
    ):
        if proj_k in sd:
            tgt_k = "embeddings.patch_embedding.linear.weight"
            out[f"vision_model.{tgt_k}"] = sd.pop(proj_k)
            rpt.renamed.append(f"{proj_k} → {tgt_k}")
            break


def _find_conv_stem_prefix(sd: dict) -> str | None:
    """Return the path prefix of the ConvStem in the source state dict, or None."""
    for k in sd:
        # Flat conv keys look like: <prefix>.stem.<digit>.linear.weight
        m = re.match(r"^(.*?)\.stem\.\d+\.linear\.weight$", k)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Per-backbone mappers
# ---------------------------------------------------------------------------

def _map_siglip2(src: dict, num_layers: int) -> tuple[dict, MigrationReport]:
    """
    Map a BcosConverter SigLIP2 state dict (``vision_model`` already extracted)
    to native ``AloeSiglip2VisionModel`` format.

    Source key patterns (after BcosConverter, ``vision_model`` prefix removed):
      - ``embeddings.patch_embedding.linear.weight``
      - ``embeddings.position_embedding.weight``   (nn.Embedding)
      - ``encoder.layers.{N}.self_attn.{q,k,v}_proj.linear.weight``
      - ``encoder.layers.{N}.self_attn.out_proj.linear.weight``
      - ``encoder.layers.{N}.layer_norm{1,2}.weight``  (+ ``.bias`` — dropped)
      - ``encoder.layers.{N}.mlp.fc{1,2}.linear.weight``
      - ``post_layernorm.weight``, ``post_layernorm.bias``
      - ``head.*``  (pooler: probe, attention, mlp, layernorm)
    """
    sd = dict(src)   # shallow copy — we pop as we go
    out: dict[str, torch.Tensor] = {}
    rpt = MigrationReport()

    def _add(tgt: str, val: torch.Tensor, src_key: str = "", fused: bool = False,
             renamed: bool = False) -> None:
        out[f"vision_model.{tgt}"] = val
        if fused:
            rpt.fused.append(f"→ {tgt}")
        elif renamed:
            rpt.renamed.append(f"{src_key} → {tgt}")
        else:
            rpt.mapped.append(tgt)

    # ---- conv stem (when BcosConverter replaced patch_embedding with ConvStem) ----
    # Must run BEFORE the generic patch_embedding passthrough.
    _map_conv_stem(sd, stem_prefix="embeddings.patch_embedding", out=out, rpt=rpt)

    # ---- patch embedding (unchanged name) ----
    for k in list(sd):
        if k.startswith("embeddings.patch_embedding"):
            _add(k, sd.pop(k))

    # ---- position embedding: rename ----
    pos_key, pos_val = _try_pop(
        sd,
        "embeddings.position_embedding.weight",   # HF SigLIP2 Embedding name
        "embeddings.pos_embedding.embedding.weight",  # already new-format (re-export)
    )
    if pos_val is not None:
        _add("embeddings.pos_embedding.embedding.weight", pos_val,
             src_key=pos_key, renamed=True)

    # ---- conv stem (just add prefix) ----
    for k in list(sd):
        if k.startswith("embeddings.conv_stem"):
            _add(k, sd.pop(k))

    # ---- CLS / register tokens ----
    # Source (HF SigLIP2 + BcosConverter): register_tokens is a plain Parameter.
    # Native ALOE: wrapped in RegisterTokens module → key is register_tokens.tokens.
    for k in list(sd):
        if k.startswith("embeddings.cls_token"):
            _add(k, sd.pop(k))
        elif k == "embeddings.register_tokens":
            _add("embeddings.register_tokens.tokens", sd.pop(k),
                 src_key=k, renamed=True)
        elif k.startswith("embeddings.register_tokens"):
            _add(k, sd.pop(k))

    # ---- encoder layers ----
    for i in range(num_layers):
        prefix = f"encoder.layers.{i}"

        # layer norms — drop bias (target uses NoBias)
        for ln in ("layer_norm1", "layer_norm2"):
            w_key = f"{prefix}.{ln}.weight"
            b_key = f"{prefix}.{ln}.bias"
            if w_key in sd:
                _add(w_key, sd.pop(w_key))
            if b_key in sd:
                sd.pop(b_key)
                rpt.dropped.append(b_key)

        # fused QKV
        q_k = f"{prefix}.self_attn.q_proj.linear.weight"
        k_k = f"{prefix}.self_attn.k_proj.linear.weight"
        v_k = f"{prefix}.self_attn.v_proj.linear.weight"
        if all(x in sd for x in (q_k, k_k, v_k)):
            fused = _fuse_qkv(sd, q_k, k_k, v_k)
            _add(f"{prefix}.self_attn.qkv_proj.weight", fused, fused=True)
        elif f"{prefix}.self_attn.qkv_proj.weight" in sd:
            # already fused (re-export from native model)
            _add(f"{prefix}.self_attn.qkv_proj.weight",
                 sd.pop(f"{prefix}.self_attn.qkv_proj.weight"))

        # out_proj and MLP (unchanged names)
        for sub in (
            "self_attn.out_proj.linear.weight",
            "mlp.fc1.linear.weight",
            "mlp.fc2.linear.weight",
        ):
            full_k = f"{prefix}.{sub}"
            if full_k in sd:
                _add(full_k, sd.pop(full_k))

    # ---- post layernorm ----
    for k in ("post_layernorm.weight", "post_layernorm.bias"):
        if k in sd:
            _add(k, sd.pop(k))

    # ---- pooler head ----
    # probe
    if "head.probe" in sd:
        _add("head.probe", sd.pop("head.probe"))

    # q_proj (plain Linear in native model — strip .linear. wrapper if present)
    q_key, q_val = _try_pop(
        sd,
        "head.attention.q_proj.linear.weight",   # BcosConverter wrapped
        "head.attention.q_proj.weight",          # already plain
    )
    if q_val is not None:
        _add("head.attention.q_proj.weight", q_val, src_key=q_key, renamed=True)

    # fused KV for pooler
    # BcosMultiHeadAttention uses plain nn.Linear for k/v (no .linear. wrapper);
    # also handle the .linear. variant for any older checkpoints.
    hk_key, hk_val = _try_pop(
        sd,
        "head.attention.k_proj.linear.weight",   # if wrapped in BcosUnnormedLinear
        "head.attention.k_proj.weight",           # plain nn.Linear (current BcosMultiHeadAttention)
    )
    hv_key, hv_val = _try_pop(
        sd,
        "head.attention.v_proj.linear.weight",
        "head.attention.v_proj.weight",
    )
    if hk_val is not None and hv_val is not None:
        _add("head.attention.kv_proj.weight", torch.cat([hk_val, hv_val], dim=0), fused=True)
    elif "head.attention.kv_proj.weight" in sd:
        _add("head.attention.kv_proj.weight", sd.pop("head.attention.kv_proj.weight"))

    # out_proj and head MLP/norm (unchanged names)
    for k in list(sd):
        if k.startswith("head."):
            _add(k, sd.pop(k))

    # anything remaining goes over with plain prefix
    for k in list(sd):
        _add(k, sd.pop(k))
        logger.debug(f"Weight mapper: passthrough key {k}")

    return out, rpt


def _map_dinov3_or_vit(
    src: dict,
    num_layers: int,
    *,
    backbone: str,   # "dinov3" | "vit"
    drop_position_embeddings: bool = True,  # True when using rope_2d
) -> tuple[dict, MigrationReport]:
    """
    Map a BcosConverter DINOv2/DINOv3 or ViT state dict to native ALOE format.

    Two source formats are supported (auto-detected by key prefix):

    **DINOv3-native** (``facebook/dinov3-*`` custom architecture, keys start with ``layer.``):
      - ``embeddings.patch_embeddings.stem.{N}.linear.weight``   (flat ConvStem)
      - ``embeddings.patch_embeddings.patch_projection.linear.weight``
      - ``layer.{N}.norm{1,2}.weight``
      - ``layer.{N}.attention.{q,k,v}_proj.linear.weight``       (separate, needs fusing)
      - ``layer.{N}.attention.o_proj.linear.weight``
      - ``layer.{N}.mlp.{up,down}_proj.linear.weight``           (→ fc1, fc2)
      - ``layer.{N}.layer_scale{1,2}.lambda1``
      - ``norm.weight``

    **DINOv2 BERT-style** (HF ``facebook/dinov2-*``, keys start with ``encoder.layer.``):
      - ``embeddings.patch_embeddings.projection.stem.*``         (ConvStem at projection)
      - ``embeddings.patch_embeddings.projection.linear.weight``  (plain Conv2d)
      - ``encoder.layer.{N}.attention.attention.{query,key,value}.linear.weight``
      - ``encoder.layer.{N}.attention.output.dense.linear.weight``
      - ``encoder.layer.{N}.mlp.fc{1,2}.linear.weight``
      - ``encoder.layer.{N}.norm{1,2}.weight``
      - ``encoder.layer.{N}.layer_scale{1,2}.lambda1``
      - ``layernorm.weight``

    **HF ViT** (similar to DINOv2 BERT-style but with different norm/MLP names).
    """
    sd = dict(src)
    out: dict[str, torch.Tensor] = {}
    rpt = MigrationReport()

    def _add(tgt: str, val: torch.Tensor, src_key: str = "",
             fused: bool = False, renamed: bool = False) -> None:
        out[f"vision_model.{tgt}"] = val
        if fused:
            rpt.fused.append(f"→ {tgt}")
        elif renamed:
            rpt.renamed.append(f"{src_key} → {tgt}")
        else:
            rpt.mapped.append(tgt)

    # Auto-detect format: DINOv3-native has top-level "layer.{i}.*" keys.
    is_dinov3_native = any(k.startswith("layer.") for k in sd)

    if is_dinov3_native:
        # ------------------------------------------------------------------ #
        #  DINOv3-native format  (facebook/dinov3-* custom architecture)     #
        # ------------------------------------------------------------------ #

        # Drop mask_token — not used in native ALOE.
        if "embeddings.mask_token" in sd:
            sd.pop("embeddings.mask_token")
            rpt.dropped.append("embeddings.mask_token")

        # CLS / register tokens
        # Source (Facebook DINOv3): register_tokens is a plain Parameter [1, R, D].
        # Native ALOE: wrapped in RegisterTokens module → key is register_tokens.tokens.
        # Concat order matches HF ``[CLS, reg*, patches]``.
        for k in list(sd):
            if k == "embeddings.cls_token":
                _add(k, sd.pop(k))
            elif k == "embeddings.register_tokens":
                _add("embeddings.register_tokens.tokens", sd.pop(k),
                     src_key=k, renamed=True)
            elif k.startswith("embeddings.register_tokens"):
                # Already in native ALOE format (re-export).
                _add(k, sd.pop(k))

        # Conv stem: auto-detect prefix (typically "embeddings.patch_embeddings")
        stem_pfx = _find_conv_stem_prefix(sd)
        if stem_pfx:
            _map_conv_stem(sd, stem_prefix=stem_pfx, out=out, rpt=rpt)
        # If no ConvStem (plain Conv2d patch projection):
        pe_key, pe_val = _try_pop(
            sd,
            "embeddings.patch_embedding.linear.weight",     # re-export
        )
        if pe_val is not None:
            if pe_val.dim() == 4:
                pe_val = _reshape_conv_to_linear(pe_val)
            _add("embeddings.patch_embedding.linear.weight", pe_val,
                 src_key=pe_key, renamed=True)

        # Encoder layers
        for i in range(num_layers):
            src_pfx = f"layer.{i}"
            tgt_pfx = f"encoder.layers.{i}"

            # Layer norms  (norm1/2 → layer_norm1/2)
            for norm_sfx in ("norm1", "norm2"):
                tgt_norm = f"layer_{norm_sfx}"      # norm1 → layer_norm1
                w = sd.pop(f"{src_pfx}.{norm_sfx}.weight", None)
                b = sd.pop(f"{src_pfx}.{norm_sfx}.bias", None)
                if w is not None:
                    _add(f"{tgt_pfx}.{tgt_norm}.weight", w,
                         src_key=f"{src_pfx}.{norm_sfx}.weight", renamed=True)
                if b is not None:
                    rpt.dropped.append(f"{src_pfx}.{norm_sfx}.bias")

            # Layer scale
            for ls_name in ("layer_scale1", "layer_scale2"):
                ls_src = f"{src_pfx}.{ls_name}.lambda1"
                ls_tgt = f"{tgt_pfx}.{ls_name}.lambda1"
                if ls_src in sd:
                    _add(ls_tgt, sd.pop(ls_src), src_key=ls_src, renamed=True)

            # Fused QKV (q_proj, k_proj, v_proj → qkv_proj)
            q_k = f"{src_pfx}.attention.q_proj.linear.weight"
            k_k = f"{src_pfx}.attention.k_proj.linear.weight"
            v_k = f"{src_pfx}.attention.v_proj.linear.weight"
            if all(x in sd for x in (q_k, k_k, v_k)):
                _add(f"{tgt_pfx}.self_attn.qkv_proj.weight",
                     _fuse_qkv(sd, q_k, k_k, v_k), fused=True)
            elif f"{tgt_pfx}.self_attn.qkv_proj.weight" in sd:
                _add(f"{tgt_pfx}.self_attn.qkv_proj.weight",
                     sd.pop(f"{tgt_pfx}.self_attn.qkv_proj.weight"))

            # Out projection (o_proj → out_proj)
            o_k = f"{src_pfx}.attention.o_proj.linear.weight"
            if o_k in sd:
                _add(f"{tgt_pfx}.self_attn.out_proj.linear.weight", sd.pop(o_k),
                     src_key=o_k, renamed=True)

            # MLP  (up_proj → fc1, down_proj → fc2)
            for src_mlp, tgt_mlp in (
                (f"{src_pfx}.mlp.up_proj.linear.weight",   f"{tgt_pfx}.mlp.fc1.linear.weight"),
                (f"{src_pfx}.mlp.down_proj.linear.weight", f"{tgt_pfx}.mlp.fc2.linear.weight"),
            ):
                if src_mlp in sd:
                    _add(tgt_mlp, sd.pop(src_mlp), src_key=src_mlp, renamed=True)

        # Post layernorm  (norm → post_layernorm)
        for src_k, tgt_k in (
            ("norm.weight", "post_layernorm.weight"),
            ("norm.bias",   "post_layernorm.bias"),
        ):
            if src_k in sd:
                _add(tgt_k, sd.pop(src_k), src_key=src_k, renamed=True)

    else:
        # ------------------------------------------------------------------ #
        #  DINOv2 BERT-style / HF ViT format  (encoder.layer.* prefix)      #
        # ------------------------------------------------------------------ #

        # CLS token
        if "embeddings.cls_token" in sd:
            _add("embeddings.cls_token", sd.pop("embeddings.cls_token"))

        # Register tokens
        for k in list(sd):
            if "register_tokens" in k:
                _add(k, sd.pop(k))

        # Conv stem (BcosConverter placed it at embeddings.patch_embeddings.projection)
        _map_conv_stem(sd, stem_prefix="embeddings.patch_embeddings.projection",
                       out=out, rpt=rpt)

        # Plain patch embedding (Conv2d → reshape to Linear)
        pe_key, pe_val = _try_pop(
            sd,
            "embeddings.patch_embeddings.projection.linear.weight",  # BcosConverter
            "embeddings.patch_embedding.linear.weight",               # already migrated
        )
        if pe_val is not None:
            if pe_val.dim() == 4:
                pe_val = _reshape_conv_to_linear(pe_val)
            _add("embeddings.patch_embedding.linear.weight", pe_val,
                 src_key=pe_key, renamed=True)

        # Bias for patch projection (drop)
        pe_bias_key, pe_bias_val = _try_pop(
            sd, "embeddings.patch_embeddings.projection.bias",
            "embeddings.patch_embeddings.projection.linear.bias",
        )
        if pe_bias_val is not None:
            rpt.dropped.append(pe_bias_key)

        # Position embeddings
        pos_key, pos_val = _try_pop(
            sd,
            "embeddings.position_embeddings",           # DINOv2 — shape (1, N+1, D)
            "embeddings.position_embeddings.weight",
        )
        if pos_val is not None:
            if drop_position_embeddings:
                rpt.dropped.append(pos_key)
            else:
                if pos_val.dim() == 3:
                    pos_val = pos_val.squeeze(0)
                _add("embeddings.pos_embedding.embedding.weight", pos_val,
                     src_key=pos_key, renamed=True)

        # Already-migrated conv_stem keys (re-export path)
        for k in list(sd):
            if k.startswith("embeddings.conv_stem"):
                _add(k, sd.pop(k))

        # Encoder layers
        for i in range(num_layers):
            src_pfx = f"encoder.layer.{i}"
            tgt_pfx = f"encoder.layers.{i}"

            # Layer norms (DINOv2: norm1/2 | ViT: layernorm_before/after)
            for src_ln, tgt_ln in (
                (f"{src_pfx}.norm1",           f"{tgt_pfx}.layer_norm1"),
                (f"{src_pfx}.norm2",           f"{tgt_pfx}.layer_norm2"),
                (f"{src_pfx}.layernorm_before", f"{tgt_pfx}.layer_norm1"),
                (f"{src_pfx}.layernorm_after",  f"{tgt_pfx}.layer_norm2"),
            ):
                w = sd.pop(f"{src_ln}.weight", None)
                b = sd.pop(f"{src_ln}.bias", None)
                if w is not None:
                    _add(f"{tgt_ln}.weight", w, src_key=f"{src_ln}.weight", renamed=True)
                if b is not None:
                    rpt.dropped.append(f"{src_ln}.bias")

            # Layer scale (DINOv2/DINOv3)
            for ls_name in ("layer_scale1", "layer_scale2"):
                ls_src = f"{src_pfx}.{ls_name}.lambda1"
                ls_tgt = f"{tgt_pfx}.{ls_name}.lambda1"
                if ls_src in sd:
                    _add(ls_tgt, sd.pop(ls_src), src_key=ls_src, renamed=True)

            # Fused QKV (BERT-style)
            q_k = f"{src_pfx}.attention.attention.query.linear.weight"
            k_k = f"{src_pfx}.attention.attention.key.linear.weight"
            v_k = f"{src_pfx}.attention.attention.value.linear.weight"
            if all(x in sd for x in (q_k, k_k, v_k)):
                _add(f"{tgt_pfx}.self_attn.qkv_proj.weight",
                     _fuse_qkv(sd, q_k, k_k, v_k), fused=True)
            elif f"{tgt_pfx}.self_attn.qkv_proj.weight" in sd:
                _add(f"{tgt_pfx}.self_attn.qkv_proj.weight",
                     sd.pop(f"{tgt_pfx}.self_attn.qkv_proj.weight"))

            # Drop q/k/v biases
            for bias_k in (
                f"{src_pfx}.attention.attention.query.linear.bias",
                f"{src_pfx}.attention.attention.key.linear.bias",
                f"{src_pfx}.attention.attention.value.linear.bias",
            ):
                if bias_k in sd:
                    rpt.dropped.append(bias_k)
                    sd.pop(bias_k)

            # Attention out_proj
            for src_k, tgt_k in (
                (f"{src_pfx}.attention.output.dense.linear.weight",
                 f"{tgt_pfx}.self_attn.out_proj.linear.weight"),
                (f"{src_pfx}.attention.output.dense.linear.bias", None),
            ):
                if src_k in sd:
                    val = sd.pop(src_k)
                    if tgt_k is not None:
                        _add(tgt_k, val, src_key=src_k, renamed=True)
                    else:
                        rpt.dropped.append(src_k)

            # MLP  (DINOv2: fc1/fc2 | ViT: intermediate.dense/output.dense)
            for src_k, tgt_k in (
                (f"{src_pfx}.mlp.fc1.linear.weight",
                 f"{tgt_pfx}.mlp.fc1.linear.weight"),
                (f"{src_pfx}.mlp.fc2.linear.weight",
                 f"{tgt_pfx}.mlp.fc2.linear.weight"),
                (f"{src_pfx}.intermediate.dense.linear.weight",
                 f"{tgt_pfx}.mlp.fc1.linear.weight"),
                (f"{src_pfx}.output.dense.linear.weight",
                 f"{tgt_pfx}.mlp.fc2.linear.weight"),
            ):
                if src_k in sd:
                    _add(tgt_k, sd.pop(src_k), src_key=src_k, renamed=(src_k != tgt_k))
            for src_k in (
                f"{src_pfx}.mlp.fc1.linear.bias", f"{src_pfx}.mlp.fc2.linear.bias",
                f"{src_pfx}.intermediate.dense.linear.bias",
                f"{src_pfx}.output.dense.linear.bias",
            ):
                if src_k in sd:
                    rpt.dropped.append(src_k)
                    sd.pop(src_k)

        # Post layernorm  (DINOv2/ViT: layernorm.*)
        for src_k, tgt_k in (
            ("layernorm.weight", "post_layernorm.weight"),
            ("layernorm.bias",   "post_layernorm.bias"),
        ):
            if src_k in sd:
                _add(tgt_k, sd.pop(src_k), src_key=src_k, renamed=True)

    # ---- pass through any remaining keys (re-export or unknown) ----
    for k in list(sd):
        _add(k, sd.pop(k))
        logger.debug(f"Weight mapper: passthrough key {k}")

    return out, rpt


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_TOP_LEVEL_HEAD_PREFIXES = ("classifier.", "logit_layer.")


def _promote_heads_from_vision_prefix(
    mapped: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """
    ``Aloe*VisionModel`` state dicts use only ``vision_model.*``.  ``Aloe*ForImageClassification``
    also has top-level ``classifier.*`` / ``logit_layer.*``.  Per-backbone mappers wrongly
    prefix *all* leftover keys with ``vision_model.``; move head keys back to the root.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in mapped.items():
        if k.startswith("vision_model."):
            suffix = k[len("vision_model.") :]
            if suffix.startswith(_TOP_LEVEL_HEAD_PREFIXES):
                out[suffix] = v
                continue
        out[k] = v
    return out


def map_factory_weights(
    source_state_dict: dict[str, Any],
    target_model: "nn.Module",
    *,
    backbone: str,
    drop_position_embeddings: bool | None = None,
    for_image_classification: bool = False,
) -> tuple[dict[str, torch.Tensor], MigrationReport]:
    """
    Map a ModelFactory / BcosConverter state dict to native ALOE HF format.

    Parameters
    ----------
    source_state_dict :
        ``model.state_dict()`` from a ModelFactory-built model.
        Keys must NOT have a ``vision_model.`` prefix (ModelFactory strips it).
    target_model :
        The native ALOE model to load into.  Used only to count encoder layers.
    backbone : ``"siglip2"`` | ``"dinov3"`` | ``"vit"``
    drop_position_embeddings :
        Whether to drop position-embedding weights from the source.
        Defaults to ``True`` for ``dinov3`` (uses RoPE), ``False`` for ``vit``.
        Explicitly ignored for ``siglip2`` (always kept and renamed).
    for_image_classification :
        If ``True``, target is ``Aloe*ForImageClassification`` — keep ``classifier.*`` and
        ``logit_layer.*`` at the **top level** instead of under ``vision_model.``.

    Returns
    -------
    mapped_sd : dict
        State dict ready for ``target_model.load_state_dict(…, strict=False)``.
    report : MigrationReport
        Summary with ``.log()`` and ``.raise_if_critical()`` helpers.
    """
    # Count encoder layers from the target model structure (bare trunk *or* wrapped Aloe*VisionModel).
    num_layers = _encoder_layer_count_from_target(target_model)
    if num_layers is None or num_layers == 0:
        num_layers = _infer_num_layers_from_source_keys(source_state_dict)
        logger.warning(
            "Could not read encoder depth from target model; inferred {} unique layer indices from source keys.",
            num_layers,
        )

    if backbone == "siglip2":
        mapped, rpt = _map_siglip2(source_state_dict, num_layers)
    elif backbone in ("dinov3", "vit"):
        if drop_position_embeddings is None:
            drop_position_embeddings = (backbone == "dinov3")
        mapped, rpt = _map_dinov3_or_vit(
            source_state_dict,
            num_layers,
            backbone=backbone,
            drop_position_embeddings=drop_position_embeddings,
        )
    else:
        raise ValueError(f"Unknown backbone {backbone!r}. Expected 'siglip2', 'dinov3', or 'vit'.")

    if for_image_classification:
        mapped = _promote_heads_from_vision_prefix(mapped)

    rpt.log()
    return mapped, rpt
