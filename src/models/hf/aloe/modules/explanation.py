# Minimal explanation-mode infrastructure for the ALOE HF package.
# Self-contained — no dependency on project src.* or bcos PyPI package.

from __future__ import annotations

from typing import Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from .interpretability_utils import gradient_to_image


def _prepare_tensor_for_explain(in_tensor: torch.Tensor) -> torch.Tensor:
    """Ensure 6-channel B-cos input with gradients (duplicate RGB to 6 channels via complement if needed)."""
    if in_tensor.ndim == 3:
        raise ValueError("Expected 4-dimensional input tensor")
    if in_tensor.shape[1] == 3:
        rgb = in_tensor.detach().clone().requires_grad_(True)
        in_tensor = torch.cat([rgb, 1.0 - rgb], dim=1)
        in_tensor.retain_grad()
    elif in_tensor.shape[1] != 6:
        raise ValueError(f"Expected 3 or 6 input channels, got {in_tensor.shape[1]}")
    elif not in_tensor.requires_grad:
        in_tensor.requires_grad_(True)
    if in_tensor.is_leaf and in_tensor.grad is not None:
        in_tensor.grad.zero_()
    return in_tensor


def _explain_backward_target(
    logits: torch.Tensor,
    idx: int | list[int] | torch.Tensor | None,
) -> tuple[torch.Tensor, Any]:
    """Tensor to ``.sum().backward`` and explained-class id(s)."""
    pred = logits.max(1)
    if idx is None:
        return pred.values, pred.indices.detach().cpu()
    if isinstance(idx, torch.Tensor) and idx.shape[0] == logits.shape[0]:
        return torch.gather(logits, 1, idx.unsqueeze(1)), idx
    if not isinstance(idx, torch.Tensor):
        warnings.warn(
            "Using the same class index for all batch elements.",
            stacklevel=2,
        )
    return logits[:, idx], idx  # type: ignore[index]


def _pooled_representation_for_explain(out: Any, config: Any | None = None) -> torch.Tensor:
    """Project an HF model output to a single `(B, D)` tensor that can be explained."""
    if isinstance(out, torch.Tensor):
        return out

    pooler_output = getattr(out, "pooler_output", None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output

    last_hidden_state = getattr(out, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        if last_hidden_state.ndim != 3:
            raise TypeError(f"Expected last_hidden_state to have 3 dims, got {last_hidden_state.ndim}")
        n_reg = int(getattr(config, "aloe_num_registers", 0)) if config is not None else 0
        n_cls = 1 if getattr(config, "aloe_cls_token", False) else 0
        start = n_reg + n_cls
        if start >= last_hidden_state.shape[1]:
            start = 0
        return last_hidden_state[:, start:, :].mean(dim=1)

    if isinstance(out, (list, tuple)) and out:
        first = out[0]
        if not isinstance(first, torch.Tensor):
            raise TypeError(f"Unexpected sequence output type {type(first).__name__}; expected Tensor.")
        if first.ndim == 2:
            return first
        if first.ndim == 3:
            return first.mean(dim=1)

    raise TypeError(
        f"Unexpected model output type {type(out).__name__}; expected Tensor, sequence, "
        "or ModelOutput with pooler_output / last_hidden_state."
    )


# ---------------------------------------------------------------------------
# explanation_mode context manager
# ---------------------------------------------------------------------------

class explanation_mode:  # noqa: N801
    """
    Context manager that puts all :class:`DetachableModule` submodules inside
    *model* into explanation mode (``module.detach = True``) on entry and
    restores them on exit.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self._detachable: list[nn.Module] | None = None

    def _find(self) -> None:
        self._detachable = [
            m for m in self.model.modules() if hasattr(m, "set_explanation_mode")
        ]

    def __enter__(self) -> "explanation_mode":
        if self._detachable is None:
            self._find()
        for m in self._detachable:
            m.set_explanation_mode(True)
        return self

    def __exit__(self, *_: Any) -> None:
        for m in self._detachable or []:
            m.set_explanation_mode(False)

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)

        return wrapper


# ---------------------------------------------------------------------------
# BcosUtilMixin  (minimal HF-publishable subset)
# ---------------------------------------------------------------------------

class BcosUtilMixin:
    """
    Mixin that adds explanation-mode support to an ALOE vision model.

    * ``model.explanation_mode()`` — returns a reentrant context manager.
    * ``model.is_bcos_model`` — always ``True``.
    """

    is_bcos_model: bool = True

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__expl_ctx = explanation_mode(self)  # type: ignore[arg-type]

    def explanation_mode(self) -> explanation_mode:
        """Return the reentrant explanation-mode context manager for this model."""
        return self.__expl_ctx

    def explain(
        self,
        in_tensor: torch.Tensor,
        idx: int | list[int] | torch.Tensor | None = None,
        to_numpy: bool = True,
        map_size: int = 1,
        **grad2img_kwargs: Any,
    ) -> dict[str, Any]:
        """Generate a simple pooled-feature input×gradient explanation for HF ALOE vision models."""
        if map_size != 1:
            raise ValueError("HF ALOE explain only supports map_size=1")
        in_tensor = _prepare_tensor_for_explain(in_tensor)

        if self.training:  # type: ignore[attr-defined]
            warnings.warn(
                "Model is in training mode! This might lead to unexpected results! Use model.eval()!",
                stacklevel=2,
            )

        result: dict[str, Any] = {}
        with torch.enable_grad(), self.explanation_mode():
            out = self(in_tensor)  # type: ignore[operator]
            pooled = _pooled_representation_for_explain(out, getattr(self, "config", None))
            pred_out = pooled.max(1)
            result["prediction"] = pred_out.indices.detach().cpu()
            to_be_explained_logit, explained_idx = _explain_backward_target(pooled, idx)
            result["explained_class_idx"] = explained_idx
            to_be_explained_logit.sum().backward(inputs=[in_tensor])

        contribution_map = in_tensor * in_tensor.grad
        linear_mapping = in_tensor.grad
        result["dynamic_linear_weights"] = in_tensor.grad.detach().clone()
        result["contribution_map"] = contribution_map.sum(1).unsqueeze(1).detach().clone()
        result["explanation"] = gradient_to_image(
            in_tensor.detach().clone(),
            linear_mapping.detach().clone(),
            to_numpy=to_numpy,
            **grad2img_kwargs,
        )
        return result

    @torch.no_grad()
    def get_dynamic_linear_weights(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward *x* in explanation mode and collect B-cos dynamic weight matrices."""
        weights: dict[str, torch.Tensor] = {}
        with self.explanation_mode():
            _ = self(x)  # type: ignore[operator]
        for name, module in self.named_modules():  # type: ignore[operator]
            if hasattr(module, "linear") and hasattr(module.linear, "weight"):
                weights[name] = module.linear.weight.detach()
        return weights

    def explain_language_features(
        self,
        in_tensor: torch.Tensor,
        language_features: torch.Tensor,
        idx=None,
        to_numpy: bool = True,
        map_size: int = 1,
        b: float = 1.0,
        **grad2img_kwargs: Any,
    ) -> dict[str, Any]:
        """Explain cosine similarity against external language features."""
        if map_size != 1:
            raise ValueError("HF ALOE explain_language_features only supports map_size=1")
        in_tensor = _prepare_tensor_for_explain(in_tensor)
        if in_tensor.shape[0] != 1:
            raise ValueError("Expected batch size of 1")

        if self.training:  # type: ignore[attr-defined]
            warnings.warn(
                "Model is in training mode! This might lead to unexpected results! Use model.eval()!",
                stacklevel=2,
            )

        result: dict[str, Any] = {}
        with torch.enable_grad(), self.explanation_mode():
            img_feats = _pooled_representation_for_explain(
                self(in_tensor),  # type: ignore[operator]
                getattr(self, "config", None),
            )
            img_feats_norm = F.normalize(img_feats, p=2, dim=-1)
            language_features_norm = F.normalize(language_features, p=2, dim=-1)
            cosine_logits = img_feats_norm @ language_features_norm.T

            pred = cosine_logits.max(1)
            result["prediction"] = pred.indices.detach().cpu()
            if idx is None:
                to_be_explained_logit = pred.values
                result["explained_class_idx"] = pred.indices.detach().cpu()
            else:
                if isinstance(idx, torch.Tensor) and idx.shape[0] == cosine_logits.shape[0]:
                    to_be_explained_logit = torch.gather(cosine_logits, 1, idx.unsqueeze(1))
                    result["explained_class_idx"] = idx.detach().cpu()
                else:
                    if not isinstance(idx, torch.Tensor):
                        warnings.warn(
                            "Using the same class index for all batch elements.",
                            stacklevel=2,
                        )
                    to_be_explained_logit = cosine_logits[:, idx]  # type: ignore[index]
                    result["explained_class_idx"] = idx

            if b != 1:
                sign = torch.sign(to_be_explained_logit)
                to_be_explained_logit = sign * to_be_explained_logit.abs().pow(b)

            to_be_explained_logit.sum().backward(inputs=[in_tensor])

        contribution_map = in_tensor * in_tensor.grad
        linear_mapping = in_tensor.grad
        result["dynamic_linear_weights"] = in_tensor.grad.detach().clone()
        result["contribution_map"] = contribution_map.sum(1).unsqueeze(1).detach().clone()
        result["explanation"] = gradient_to_image(
            in_tensor.detach().clone(),
            linear_mapping.detach().clone(),
            to_numpy=to_numpy,
            **grad2img_kwargs,
        )
        return result
