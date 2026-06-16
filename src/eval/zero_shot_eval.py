from __future__ import annotations

import math
import re
import string
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, cast

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.modeling_outputs import BaseModelOutputWithPooling

# Adjust imports for Aloe
from src.eval.evaluator import Evaluator
from src.eval.zero_shot_templates import OPENAI_IMAGENET_TEMPLATES
from src.explanation.abstract import ExplanationOutput
from src.explanation.bcos_utils import gradient_to_image
from src.models.hf.aloe.modules.explanation import (
    explanation_mode as bcos_submodule_explanation_mode,
)


def _explanation_mode_ctx(model: nn.Module):
    """
    Prefer ``model.explanation_mode()`` when the HF wrapper (e.g. ``AloeSiglip2VisionModel``) is used.

    ``HfModelFactory.get_pretrained_HF_model`` often **extracts** ``vision_model`` for
    ``model_part=vision``, dropping the outer class that defines ``explanation_mode`` while
    keeping the same B-cos submodules. In that case use the utils context that scans
    ``set_explanation_mode`` on all descendants.
    """
    em = getattr(model, "explanation_mode", None)
    if callable(em):
        return em()
    return bcos_submodule_explanation_mode(model)


def _vision_output_to_feats(img_feats: object) -> torch.Tensor:
    """Match ``BcosUtilMixin.explain_language_features`` vision embedding extraction."""
    if isinstance(img_feats, torch.Tensor):
        return img_feats
    if hasattr(img_feats, "pooler_output") and img_feats.pooler_output is not None:
        return img_feats.pooler_output
    if isinstance(img_feats, (list, tuple)):
        return img_feats[0]
    raise TypeError(
        f"Unexpected vision forward output type {type(img_feats).__name__}; "
        "expected a Tensor, ModelOutput with pooler_output, or a sequence."
    )


def _resolve_zero_shot_explainable_model(model: nn.Module) -> nn.Module:
    to_check: list[nn.Module] = [model]
    seen: set[int] = set()
    while to_check:
        current = to_check.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))

        if hasattr(current, "explain_language_features"):
            return current
        for attr in ("module", "_orig_mod"):
            inner = getattr(current, attr, None)
            if isinstance(inner, nn.Module):
                to_check.append(inner)

    raise NotImplementedError(
        "Zero-shot explanations require a model with explain_language_features(...)."
    )


def _zero_shot_class_index_for_sample(
    idx: Optional[Union[int, list[int], torch.Tensor]],
    bi: int,
    batch_size: int,
) -> Optional[int]:
    if idx is None:
        return None
    if isinstance(idx, int):
        return idx
    if isinstance(idx, (list, tuple)):
        if len(idx) == 1:
            return int(idx[0])
        raise TypeError("idx as list must have length 1 for this code path")
    if isinstance(idx, torch.Tensor):
        if idx.ndim == 1 and idx.shape[0] == batch_size:
            return int(idx[bi].item())
        if idx.ndim == 0:
            return int(idx.item())
    raise TypeError(f"Unsupported idx type for zero-shot explain: {type(idx).__name__}")


def explain_language_features_minimal_bcos(
    model: nn.Module,
    in_tensor: torch.Tensor,
    language_features: torch.Tensor,
    *,
    idx: Optional[Union[int, list[int], torch.Tensor]] = None,
    to_numpy: bool = True,
    map_size: int = 1,
    b: float = 1.0,
    **grad2img_kwargs: object,
) -> ExplanationOutput:
    """
    Same behavior as full ``BcosUtilMixin.explain_language_features`` for ``map_size == 1``.

    Works when the runtime module is only the inner vision tower (no ``explanation_mode`` method)
    by applying :func:`bcos_submodule_explanation_mode` over ``model.modules()`` (see
    ``_explanation_mode_ctx``).
    """
    if map_size != 1:
        raise ValueError(
            "explain_language_features_minimal_bcos only supports map_size=1 "
            "(strided forward needs the full explainability mixin)."
        )
    if in_tensor.ndim == 3:
        raise ValueError("Expected a 4-D input tensor")
    if model.training:
        warnings.warn(
            "Model is in training mode — use model.eval() for explanations.",
            stacklevel=2,
        )

    device = in_tensor.device
    language_features = language_features.to(device)
    language_features_norm = F.normalize(language_features, p=2, dim=-1)

    batch_size = in_tensor.shape[0]
    explanations: list[torch.Tensor] = []
    contribution_maps: list[torch.Tensor] = []
    dyn_weights: list[torch.Tensor] = []
    explained_idxs: list[torch.Tensor] = []

    for bi in range(batch_size):
        x = in_tensor[bi : bi + 1].detach().clone()
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.zero_()

        class_idx = _zero_shot_class_index_for_sample(idx, bi, batch_size)

        with torch.enable_grad(), _explanation_mode_ctx(model):
            img_feats = _vision_output_to_feats(model(x))
            img_feats_norm = F.normalize(img_feats, p=2, dim=-1)
            cosine_logits = img_feats_norm @ language_features_norm.T

            if class_idx is None:
                predicted_idx = cosine_logits.argmax(dim=1)
                to_be_explained_logit = cosine_logits.gather(1, predicted_idx.unsqueeze(1))
                explained_idxs.append(predicted_idx.detach().cpu())
            else:
                to_be_explained_logit = cosine_logits[:, class_idx].unsqueeze(1)
                explained_idxs.append(torch.tensor([class_idx], dtype=torch.long))

            if b != 1:
                sign = torch.sign(to_be_explained_logit)
                to_be_explained_logit = sign * to_be_explained_logit.abs().pow(b)

            to_be_explained_logit.sum().backward(inputs=[x])

        contribution_map = x * x.grad
        linear_mapping = x.grad
        result_contrib = contribution_map.sum(1).unsqueeze(1).detach().clone()
        expl = gradient_to_image(
            x.detach().clone(),
            linear_mapping.detach().clone(),
            to_numpy=to_numpy,
            **grad2img_kwargs,  # type: ignore[arg-type]
        )
        if not isinstance(expl, torch.Tensor):
            expl = torch.as_tensor(expl)
        explanations.append(expl)
        contribution_maps.append(result_contrib)
        dyn_weights.append(x.grad.detach().clone())

    return ExplanationOutput(
        explanation=torch.cat(explanations, dim=0),
        contribution_map=torch.cat(contribution_maps, dim=0),
        explained_class_idx=torch.cat(explained_idxs, dim=0),
        dynamic_linear_weights=torch.cat(dyn_weights, dim=0),
    )


class ZeroShotEvaluator(Evaluator):
    """Lightweight zero-shot evaluator for precomputed image features.

    Usage:
        zs = ZeroShotEvaluator(language_model, tokenizer)
        zs.set_prompts_from_labels(label_names)  # or pass text_features directly to evaluate
        metrics = zs.evaluate(features_loader)   # loader yielding (image_features, targets)
    """

    def __init__(
        self,
        language_model: Optional[str] = None,
        tokenizer: Optional[str] = None,
        class_names: Optional[List[str]] = None,
        templates: Optional[List[str]] = None,
        feature_cache_dir: Optional[str] = None,
        force_recompute: bool = False,
        device: Optional[Union[torch.device, str]] = None,
        hf_model_reference: Optional[str] = None,
        *,
        logit_scale: float = 100.0,
        topk: Sequence[int] = (1, 5, 10),
        batch_size: int = 256,
    ) -> None:
        super().__init__(
            feature_cache_dir=feature_cache_dir, force_recompute=force_recompute
        )
        self.language_model_name = language_model
        self.tokenizer_name = tokenizer
        self.class_names = class_names
        self.templates = templates
        self.hf_model_reference = hf_model_reference
        self.logit_scale = float(logit_scale)
        self.topk = topk
        self.batch_size = batch_size
        self.language_model: Optional[nn.Module] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._class_prompts: Optional[List[str]] = None
        self._text_features: Optional[torch.Tensor] = None
        self.num_labels = 0
        self.num_templates = 0

        # Repo id passed to ``AutoTokenizer`` (may differ from ``hf_model_reference`` for vision-only Hub ids).
        self._tokenizer_pretrained_id: Optional[str] = None

        if self.language_model is None and self.hf_model_reference is not None:
            from transformers import AutoConfig, AutoModel, CLIPTextModel

            hf_ref = self.hf_model_reference
            load_kw: dict = {}
            if isinstance(hf_ref, str) and "/" in hf_ref:
                load_kw["trust_remote_code"] = True

            cfg = AutoConfig.from_pretrained(hf_ref, **load_kw)
            model_type = str(getattr(cfg, "model_type", "") or "")

            # Native ALOE SigLIP2 **vision** repos have no text tower or tokenizer; align text side with
            # the original SigLIP2 multimodal checkpoint named in ``aloe_base_model_name``.
            if model_type.startswith("aloe_siglip2"):
                base = getattr(cfg, "aloe_base_model_name", None) or (
                    "google/siglip2-base-patch16-224"
                )
                multimodal = AutoModel.from_pretrained(base, trust_remote_code=True)
                if not hasattr(multimodal, "text_model"):
                    raise RuntimeError(
                        f"Expected a SigLIP-style multimodal model with `.text_model` from {base!r} "
                        f"(vision reference repo {hf_ref!r})."
                    )
                self.language_model = multimodal.text_model
                self._tokenizer_pretrained_id = base
            else:
                try:
                    self.language_model = CLIPTextModel.from_pretrained(hf_ref, **load_kw)
                except Exception:
                    self.language_model = AutoModel.from_pretrained(hf_ref, **load_kw)
                self._tokenizer_pretrained_id = hf_ref

            self.language_model.to(self.device)
            self.language_model.eval()

        if self.language_model is not None:
            # If the model is a wrapper (like SiglipModel or CLIPModel), use the text model directly
            if hasattr(self.language_model, "text_model"):
                self.language_model = self.language_model.text_model

            self.language_model.to(self.device)
            self.language_model.eval()

        # Auto-initialize tokenizer from HF language model if not provided
        if self.tokenizer is None and self.language_model is not None:
            tok_id = self._tokenizer_pretrained_id or self.hf_model_reference
            assert tok_id is not None, (
                "hf_model_reference must be provided to auto-initialize tokenizer."
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                tok_id,
                use_fast=True,
                trust_remote_code=True,
            )

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = label.lower()
        label = label.translate(str.maketrans("", "", string.punctuation))
        label = re.sub(r"\s+", " ", label).strip()
        return label

    def set_prompts_from_labels(
        self, labels: List[str], template: list = OPENAI_IMAGENET_TEMPLATES
    ) -> None:
        normalized = [self._normalize_label(lbl) for lbl in labels]
        # Ensure prompts are grouped by label (all templates for label 0, then label 1, ...)
        self._class_prompts = [t(c=label) for label in normalized for t in template]
        self.num_labels = len(labels)
        self.num_templates = len(template)
        self._text_features = None  # invalidate cache
        # logger.info(f"Set {self.num_labels} labels with {self.num_templates} templates each ({len(self._class_prompts)} total prompts).")

    @torch.no_grad()
    def _ensure_text_features(self, batch_size: int = 100) -> torch.Tensor:
        if self._text_features is not None:
            return torch.as_tensor(self._text_features, device=self.device)
        if self._class_prompts is None:
            raise RuntimeError(
                "No class prompts set. Call set_prompts_from_labels() or pass text_features to evaluate()."
            )
        if self.language_model is None or self.tokenizer is None:
            raise RuntimeError(
                "language_model and tokenizer are required to compute text features."
            )

        toks = self.tokenizer(
            self._class_prompts,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        num_batches = math.ceil(len(self._class_prompts) / batch_size)
        output = []
        for idx in range(num_batches):
            start = idx * batch_size
            end = start + batch_size
            token_batch = {k: v[start:end].to(self.device) for k, v in toks.items()}

            out: BaseModelOutputWithPooling = self.language_model(**token_batch)  # type: ignore[misc]
            tf = out.pooler_output
            if tf is None:
                raise RuntimeError("Language model did not return pooled output.")
            output.append(tf)

        tf = torch.cat(output, dim=0)
        tf = tf / tf.norm(dim=-1, keepdim=True)

        expected = self.num_labels * self.num_templates
        if tf.shape[0] != expected:
            raise RuntimeError(
                f"Unexpected text feature count: got {tf.shape[0]}, expected {expected}"
            )

        tf = tf.view(self.num_labels, self.num_templates, -1).mean(dim=1)
        tf = tf / tf.norm(dim=-1, keepdim=True)
        self._text_features = tf
        return tf

    @torch.no_grad()
    def evaluate(
        self,
        data: Optional[
            DataLoader
            | Iterable[Tuple[torch.Tensor, torch.Tensor]]
            | Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        *,
        text_features: Optional[torch.Tensor] = None,
        topk: Optional[Sequence[int]] = None,
        normalize_features: bool = True,
        return_counts: bool = False,
        model: Optional[nn.Module] = None,
        datamodule: Optional[L.LightningDataModule] = None,
        trainer: Optional[L.Trainer] = None,
        split: str = "val",
    ) -> Dict[str, float]:
        """Compute zero-shot metrics from precomputed image features.

        data can be:
          - a DataLoader or iterable yielding (image_features, targets)
          - a tuple (all_image_features, all_targets)
          - None (if lightning_module and datamodule are provided)
        """
        # Resolve features
        if data is None:
            if model is None or datamodule is None:
                raise ValueError(
                    "If data is not provided, model and datamodule must be provided."
                )

            # Use shared extraction logic
            device = next(model.parameters()).device
            feats, labels = self.get_features(model, datamodule, split, device)
            data = (feats, labels)

        # Resolve text features
        if text_features is None:
            text_features = self._ensure_text_features()
        assert text_features is not None
        text_features = torch.as_tensor(text_features, device=self.device)
        text_feats = cast(torch.Tensor, text_features)
        if normalize_features:
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        # Prepare accumulators
        if topk is None:
            topk = self.topk

        topk_list = [int(k) for k in topk if int(k) >= 1]
        if len(topk_list) == 0:
            topk_list = [1]
        max_k = max(topk_list)
        total = 0
        correct_k: Dict[int, int] = {k: 0 for k in topk_list}

        def _eval_batch(img_feats: torch.Tensor, targets: torch.Tensor):
            nonlocal total
            img_feats = img_feats.to(self.device)
            targets = targets.to(self.device)
            if normalize_features:
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            logits = self.logit_scale * (img_feats @ text_feats.T)
            # compute top-k predictions once up to max_k, then accumulate
            top_idx = logits.topk(max_k, dim=1).indices
            matches = top_idx.eq(targets.unsqueeze(1))
            for k in topk_list:
                correct_k[k] += int(matches[:, :k].any(dim=1).sum().item())
            total += targets.numel()

        # Iterate data
        if isinstance(data, tuple):
            feats, labels = data
            feats = torch.as_tensor(feats)
            labels = torch.as_tensor(labels)
            _eval_batch(feats, labels)
        else:
            for batch in data:
                # Handle dict batch
                if isinstance(batch, dict):
                    if "features" in batch:
                        feats = batch["features"]
                    elif "image_features" in batch:
                        feats = batch["image_features"]
                    else:
                        # Fallback or error
                        feats = batch[0]  # Try positional

                    if "label" in batch:
                        labels = batch["label"]
                    elif "targets" in batch:
                        labels = batch["targets"]
                    else:
                        labels = batch[1]
                else:
                    feats, labels = batch

                _eval_batch(feats, labels)

        metrics: Dict[str, float] = {}
        denom = float(max(total, 1))
        for k in topk_list:
            metrics[f"acc@{k}"] = correct_k[k] / denom
        if return_counts:
            metrics.update({"total": float(total)})
            for k in topk_list:
                metrics[f"correct@{k}"] = float(correct_k[k])
        return metrics

    def explain_batch(
        self,
        images: torch.Tensor,
        model: nn.Module,
        b=1,
        idx: Optional[list[int]] = None,
        converted_model=True,
        pretrained_weights=None,
        **explain_kwargs,
    ) -> None:
        # Only support B-cos for now
        return self.explain_bcos_batch(images, model, b, idx, **explain_kwargs)

    def explain_bcos_batch(
        self,
        images: torch.Tensor,
        model: nn.Module,
        b=1,
        idx: Optional[list[int]] = None,
        **explain_kwargs,
    ) -> None:
        text_features = self._ensure_text_features()
        assert text_features is not None
        text_features = torch.as_tensor(text_features, device=self.device)
        text_feats = cast(torch.Tensor, text_features)

        explainable_model = _resolve_zero_shot_explainable_model(model)
        out = explainable_model.explain_language_features(
            in_tensor=images,
            language_features=text_feats,
            idx=idx,
            to_numpy=False,
            map_size=1,
            b=b,
            **explain_kwargs,
        )
        return out if isinstance(out, ExplanationOutput) else ExplanationOutput(**out)
