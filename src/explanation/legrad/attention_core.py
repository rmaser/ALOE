"""Shared attention-attribution cores for classifier LeGrad and Chefer CAM.

The algorithms here intentionally know nothing about DINO, ViT, SigLIP, or ALOE
module layouts.  Adapters provide hidden states, attention tensors, pooling
seeds, and class scores for a concrete architecture.
"""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .utils import min_max


class ClassifierAttentionAdapter(Protocol):
    patch_size: int

    def run(self, image: Tensor) -> None: ...

    def layer_indices(self) -> list[int]: ...

    def all_layer_indices(self) -> list[int]: ...

    def attention_map(self, layer_idx: int) -> Tensor: ...

    def hidden_state(self, layer_idx: int) -> Tensor: ...

    def score_from_hidden(self, hidden_state: Tensor, target: Tensor) -> Tensor: ...

    def score_and_pool_attention_from_hidden(self, hidden_state: Tensor, target: Tensor) -> tuple[Tensor, Tensor]: ...

    def final_score_and_pool_attention(self, target: Tensor) -> tuple[Tensor, Tensor]: ...

    def target_score(self, target: Tensor) -> Tensor: ...

    def final_pool_attention(self) -> Tensor: ...

    def uses_attention_pooling(self) -> bool: ...

    def patch_indices(self, sequence_length: int) -> Tensor: ...

    def cls_attention_row(self) -> int: ...


def _mean_heads(x: Tensor) -> Tensor:
    """Return attention-like tensors as ``[B, Q, K]`` after head averaging."""
    if x.ndim == 4:
        return x.mean(dim=1)
    if x.ndim == 3:
        return x
    if x.ndim == 2:
        return x.unsqueeze(1)
    raise ValueError(f"Expected attention tensor with rank 2-4, got shape {tuple(x.shape)}")


def _patch_grid(num_patches: int) -> tuple[int, int]:
    side = int(num_patches**0.5)
    if side * side != num_patches:
        raise ValueError(f"Only square patch grids are supported for now, got {num_patches} patches")
    return side, side


def _patch_map(
    scores: Tensor,
    patch_size: int,
    normalize: bool,
    output_size: tuple[int, int] | None = None,
) -> Tensor:
    """Convert per-patch scores ``[B, P]`` into an image-space heatmap."""
    w, h = _patch_grid(scores.shape[-1])
    heatmap = rearrange(scores, "b (w h) -> b 1 w h", w=w, h=h)
    if normalize:
        heatmap = min_max(heatmap)
    if output_size is not None:
        return F.interpolate(heatmap, size=output_size, mode="bilinear", align_corners=False)
    return F.interpolate(heatmap, scale_factor=patch_size, mode="bilinear")


def _pool_relevance(attn: Tensor, grad: Tensor, *, use_attention_values: bool) -> Tensor:
    """Pooler query relevance as ``[B, P]``.

    LeGrad uses positive gradients.  Chefer uses positive gradient-weighted
    attention.  Adapters ensure pooler attention keys already correspond to
    patch tokens, or expose a compatible key axis.
    """
    signal = grad * attn if use_attention_values else grad
    signal = torch.clamp(signal, min=0.0)
    signal = _mean_heads(signal)
    return signal.mean(dim=1)


def _grad_for_differentiable_inputs(
    score: Tensor,
    inputs: list[Tensor],
    *,
    retain_graph: bool,
    create_graph: bool = False,
) -> tuple[Tensor | None, ...]:
    """Return autograd gradients, leaving non-differentiable inputs as ``None``."""
    grads: list[Tensor | None] = [None] * len(inputs)
    if not score.requires_grad:
        return tuple(grads)

    grad_positions: list[int] = []
    grad_inputs: list[Tensor] = []
    for idx, input_tensor in enumerate(inputs):
        if input_tensor.requires_grad:
            grad_positions.append(idx)
            grad_inputs.append(input_tensor)

    if not grad_inputs:
        return tuple(grads)

    computed_grads = torch.autograd.grad(
        score,
        grad_inputs,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
    )
    for idx, grad in zip(grad_positions, computed_grads, strict=True):
        grads[idx] = grad
    return tuple(grads)


def _require_encoder_attention_gradients(grads: tuple[Tensor | None, ...]) -> None:
    if any(grad is not None for grad in grads):
        return
    raise ValueError(
        "Chefer CAM could not obtain gradients for any encoder attention map. "
        "Run the explainer with torch gradients enabled and an eager attention "
        "implementation that returns differentiable attention weights."
    )


def compute_legrad_from_adapter(
    adapter: Any,
    image: Tensor,
    target: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """Compute LeGrad once for every classifier attention adapter."""
    adapter.run(image)
    accum: Tensor | None = None

    for layer_idx in adapter.layer_indices():
        hidden_state = adapter.hidden_state(layer_idx)
        if adapter.uses_attention_pooling():
            score, attn_map = adapter.score_and_pool_attention_from_hidden(hidden_state, target)
            patch_idx = adapter.patch_indices(hidden_state.shape[1])
            grad = _grad_for_differentiable_inputs(
                score,
                [attn_map],
                retain_graph=True,
            )[0]
            if grad is None:
                continue
            layer_scores = _pool_relevance(grad, grad, use_attention_values=False)
            if layer_scores.shape[-1] != patch_idx.numel():
                layer_scores = layer_scores[:, patch_idx]
        else:
            attn_map = adapter.attention_map(layer_idx)
            patch_idx = adapter.patch_indices(attn_map.shape[-1])
            score = adapter.score_from_hidden(hidden_state, target)
            grad = _grad_for_differentiable_inputs(
                score,
                [attn_map],
                retain_graph=True,
            )[0]
            if grad is None:
                continue
            layer_scores = torch.clamp(_mean_heads(grad), min=0.0)[:, :, patch_idx].mean(dim=1)

        accum = layer_scores if accum is None else accum + layer_scores

    if accum is None:
        last_hidden = adapter.hidden_state(adapter.layer_indices()[-1])
        patch_idx = adapter.patch_indices(last_hidden.shape[1])
        accum = last_hidden.new_zeros((last_hidden.shape[0], patch_idx.numel()))

    return _patch_map(accum, adapter.patch_size, normalize, output_size=tuple(image.shape[-2:]))


def _encoder_rollout(adapter: Any, grads: tuple[Tensor | None, ...]) -> Tensor:
    attn_maps = [adapter.attention_map(idx) for idx in adapter.all_layer_indices()]
    if not attn_maps:
        raise ValueError("No encoder attention maps captured")

    first_attn = attn_maps[0]
    seq_len = first_attn.shape[-1]
    batch_size = first_attn.shape[0]
    rollout = torch.eye(seq_len, device=first_attn.device, dtype=first_attn.dtype).unsqueeze(0)
    rollout = rollout.expand(batch_size, -1, -1)

    eye = torch.eye(seq_len, device=first_attn.device, dtype=first_attn.dtype).unsqueeze(0)
    for attn, grad in zip(attn_maps, grads, strict=True):
        if grad is None:
            continue
        cam = torch.clamp(attn * grad, min=0.0)
        cam = _mean_heads(cam)
        cam = cam + eye
        cam = cam / (cam.sum(dim=-1, keepdim=True) + 1e-9)
        rollout = torch.bmm(cam, rollout)
    return rollout


def compute_chefer_from_adapter(
    adapter: Any,
    image: Tensor,
    target: Tensor,
    *,
    normalize: bool = True,
) -> Tensor:
    """Compute Chefer CAM once for every classifier attention adapter."""
    adapter.run(image)
    layer_indices = adapter.all_layer_indices()
    attn_maps = [adapter.attention_map(idx) for idx in layer_indices]
    if not attn_maps:
        raise ValueError("No encoder attention maps captured")

    if adapter.uses_attention_pooling():
        score, pool_attn = adapter.final_score_and_pool_attention(target)
        grads = _grad_for_differentiable_inputs(
            score,
            [pool_attn, *attn_maps],
            retain_graph=False,
        )
        pool_grad = grads[0]
        if pool_grad is None:
            raise ValueError(
                "Chefer CAM could not obtain gradients for the attention-pooling head. "
                "Run the explainer with torch gradients enabled and an eager attention "
                "implementation that returns differentiable attention weights."
            )
        encoder_grads = grads[1:]
        _require_encoder_attention_gradients(encoder_grads)
        seed = _pool_relevance(pool_attn, pool_grad, use_attention_values=True)
        patch_idx = adapter.patch_indices(attn_maps[0].shape[-1])
        if seed.shape[-1] != patch_idx.numel():
            seed = seed[:, patch_idx]
        rollout = _encoder_rollout(adapter, encoder_grads)
        patch_rollout = rollout[:, patch_idx, :]
        full_scores = torch.bmm(seed.unsqueeze(1), patch_rollout).squeeze(1)
        scores = full_scores[:, patch_idx]
    else:
        score = adapter.target_score(target)
        grads = _grad_for_differentiable_inputs(score, attn_maps, retain_graph=False)
        _require_encoder_attention_gradients(grads)
        rollout = _encoder_rollout(adapter, grads)
        patch_idx = adapter.patch_indices(attn_maps[0].shape[-1])
        scores = rollout[:, adapter.cls_attention_row(), :][:, patch_idx]

    return _patch_map(scores, adapter.patch_size, normalize, output_size=tuple(image.shape[-2:]))
