# ALOE vision models with a classification head (supervised ``logits`` + ``explain``).

from __future__ import annotations

import math
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutput

from src.modules.logit_layer import LogitLayer

from .configuration_aloe_vision import (
    AloeDinoV3VisionConfig,
    AloeSiglip2VisionConfig,
    AloeViTVisionConfig,
)
from .modeling_aloe_base import AloePreTrainedVisionModel, AloeVisionTransformer
from .modules.bcos_core import BcosUnnormedLinear, select_bcos_unnormed_linear
from .modules.explanation import _explain_backward_target, _prepare_tensor_for_explain
from .modules.interpretability_utils import gradient_to_image as _grad2img
from .modules.loss import UniformOffLabelsBCEWithLogitsLoss
from .modules.interpretability_utils import plot_contribution_map


def _pooled_representation(out: BaseModelOutputWithPooling, config: Any) -> torch.Tensor:
    """Pooled vector: use ``pooler_output`` if set, else mean over patch tokens (skip registers/CLS)."""
    if out.pooler_output is not None:
        return out.pooler_output
    hs = out.last_hidden_state
    n_reg = int(getattr(config, "aloe_num_registers", 0))
    n_cls = 1 if getattr(config, "aloe_cls_token", False) else 0
    start = n_reg + n_cls
    return hs[:, start:, :].mean(dim=1)


class AloeForImageClassificationBase(AloePreTrainedVisionModel):
    """Vision tower + B-cos classifier head; ``explain`` uses class logits."""

    _no_split_modules = ["AloeVisionEmbeddings", "AloeEncoderLayer"]

    gradient_to_image = staticmethod(_grad2img)
    plot_contribution_map = staticmethod(plot_contribution_map)

    def __init__(self, config: Any) -> None:
        """Vision tower + B-cos classifier; optional :class:`~src.modules.logit_layer.LogitLayer` for CE loss only."""
        super().__init__(config)
        self.vision_model = AloeVisionTransformer(config)
        _Lin = select_bcos_unnormed_linear(config)
        self.classifier = _Lin(
            config.hidden_size,
            config.num_labels,
            b=config.aloe_b_linear,
            detach_output=False,
        )
        n_labels = int(getattr(config, "num_labels", 0))
        use_logit = bool(getattr(config, "aloe_use_logit_layer", False)) and n_labels > 1
        if use_logit:
            bias = getattr(config, "aloe_logit_bias", None)
            if bias is None:
                bias = -math.log(n_labels - 1)
            temp = getattr(config, "aloe_logit_temperature", None)
            self.logit_layer = LogitLayer(logit_temperature=temp, logit_bias=bias)
        else:
            self.logit_layer = nn.Identity()
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        """Patch stem / projection used as HF ``get_input_embeddings``."""
        return self.vision_model.embeddings.patch_embedding

    def _init_aloe_submodules(self, module: nn.Module) -> bool:
        """Xavier for the classifier head; other ALOE blocks use the base hook."""
        if isinstance(module, BcosUnnormedLinear) and getattr(self, "classifier", None) is module:
            init.xavier_uniform_(module.linear.weight)
            return True
        return super()._init_aloe_submodules(module)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ) -> ImageClassifierOutput:
        """Encode image, pool, classify, optional CE/MSE loss when ``labels`` given.

        Returned ``logits`` follow the model's prediction semantics: when
        ``aloe_use_logit_layer`` is enabled, they are the calibrated
        ``logit_layer(classifier(...))`` outputs. This keeps supervised eval /
        inference consistent with LP checkpoints and avoids every downstream
        caller having to remember to apply ``logit_layer`` manually.

        Loss semantics match native ALOE classification:
        ``aloe_use_logit_layer=True`` with ``num_labels > 1`` uses
        :class:`bcos.modules.losses.UniformOffLabelsBCEWithLogitsLoss`,
        otherwise we fall back to standard CE / MSE behavior.
        """
        vm_out = self.vision_model(
            pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        pooled = _pooled_representation(vm_out, self.config)
        raw_logits = self.classifier(pooled)
        logits = (
            self.logit_layer(raw_logits)
            if (
                bool(getattr(self.config, "aloe_use_logit_layer", False))
                and not isinstance(self.logit_layer, nn.Identity)
            )
            else raw_logits
        )

        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze().float())
            elif (
                bool(getattr(self.config, "aloe_use_logit_layer", False))
                and not isinstance(self.logit_layer, nn.Identity)
            ):
                loss = UniformOffLabelsBCEWithLogitsLoss()(logits, labels)
            else:
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=vm_out.hidden_states,
            attentions=vm_out.attentions,
        )

    def explain(
        self,
        in_tensor: torch.Tensor,
        idx: Union[int, List[int], torch.Tensor, None] = None,
        to_numpy: bool = True,
        smooth: int = 15,
        alpha_percentile: float = 99.5,
        **vision_kwargs: Any,
    ) -> dict[str, Any]:
        """
        Input×gradient maps for a **class logit** (B-cos 6-channel convention).

        Uses the same logits as :meth:`forward` for prediction and ``backward``.
        """
        in_tensor = _prepare_tensor_for_explain(in_tensor)

        if self.training:
            warnings.warn("Model is in training mode; use eval() for explanations.", stacklevel=2)

        result: dict[str, Any] = {}
        with torch.enable_grad(), self.explanation_mode():
            out = self.forward(in_tensor, **vision_kwargs)
            logits = out.logits
            if logits is None:
                raise RuntimeError("forward did not return logits")
            pred_out = logits.max(1)
            result["prediction"] = pred_out.indices.detach().cpu()
            to_be_explained_logit, explained_idx = _explain_backward_target(logits, idx)
            result["explained_class_idx"] = explained_idx

            to_be_explained_logit.sum().backward(inputs=[in_tensor])

        contribution_map = in_tensor * in_tensor.grad
        linear_mapping = in_tensor.grad
        result["dynamic_linear_weights"] = in_tensor.grad.detach().clone()
        result["contribution_map"] = contribution_map.sum(1).unsqueeze(1).detach().clone()
        result["explanation"] = _grad2img(
            in_tensor.detach().clone(),
            linear_mapping.detach().clone(),
            smooth=smooth,
            alpha_percentile=alpha_percentile,
            to_numpy=to_numpy,
        )
        return result

    @classmethod
    def from_backbone_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        num_labels: int,
        **kwargs: Any,
    ) -> "AloeForImageClassificationBase":
        """Load backbone weights from Hub/local into a supervised ALOE model.

        Distinction from the config helpers:
        - ``config_class.from_backbone_pretrained(...)`` builds a supervised *config*
          (config-only; no weights loaded).
        - ``cls.from_pretrained(..., config=cfg, ...)`` is where Hugging Face
          actually loads the backbone weights. The classification head can be
          reinitialized because we use ``ignore_mismatched_sizes=True``.
        """
        load_kw = dict(kwargs)
        trust = bool(load_kw.pop("trust_remote_code", True))
        # 1) Build a supervised config object (config-only).
        cfg = cls.config_class.from_backbone_pretrained(
            pretrained_model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=trust,
            **load_kw,
        )
        # 2) Load backbone weights into the supervised model using the config above.
        return cls.from_pretrained(
            pretrained_model_name_or_path,
            config=cfg,
            ignore_mismatched_sizes=True,
            trust_remote_code=trust,
            **load_kw,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Default ``ignore_mismatched_sizes=True`` so classifier shape can differ from checkpoint."""
        kwargs.setdefault("ignore_mismatched_sizes", True)
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)


class AloeSiglip2ForImageClassification(AloeForImageClassificationBase):
    config_class = AloeSiglip2VisionConfig


class AloeDinoV3ForImageClassification(AloeForImageClassificationBase):
    config_class = AloeDinoV3VisionConfig

class AloeViTForImageClassification(AloeForImageClassificationBase):
    config_class = AloeViTVisionConfig


__all__ = [
    "AloeDinoV3ForImageClassification",
    "AloeSiglip2ForImageClassification",
    "AloeViTForImageClassification",
]
