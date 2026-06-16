# model_factory.py
#
# HF-native ALOE model construction: registered ALOE Hub models, optional HF checkpoint
# mapping, freeze / checkpoint loading, and compile.

from typing import Any, Dict, List, Literal, Optional
import re
import types

from beartype import beartype
from loguru import logger
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutput

from src.models.config import HFModelConfig, VitConfig
from src.models.model_factory.utils import (
    get_check_if_kept_trainable_fn,
    load_hf_automodel_state_dict,
    load_state_dict_from_checkpoint,
    load_state_dict_from_hf_model,
    load_state_dict_into_model,
)
from src.modules.pooler.cls_pooler import CLSPooler

from .config import CheckpointLoadConfig, ModelFactoryConfig
from .heuristics import CLASSIFIER_KEYS, MAPPING_KEYS, POOLER_KEYS


class PlainHFImageClassificationModel(nn.Module):
    """Plain HF backbone + linear classifier head for non-Bcos baseline models."""

    def __init__(self, vision_model: nn.Module, *, feature_dim: int, num_labels: int) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.classifier = nn.Linear(feature_dim, num_labels, bias=False)

    def forward(self, *args: Any, **kwargs: Any) -> ImageClassifierOutput:
        out = self.vision_model(*args, **kwargs)
        if not isinstance(out, BaseModelOutputWithPooling):
            raise TypeError(
                "PlainHFImageClassificationModel expects BaseModelOutputWithPooling from the backbone."
            )
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return ImageClassifierOutput(
            logits=logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
        )


class HfModelFactory:
    """HF-native factory for ALOE vision backbones and checkpoints."""

    def __init__(self, config: ModelFactoryConfig):
        self.config = config

    @beartype
    def create_model(self, model_type: str = "vision") -> nn.Module:
        """Create and fully configure the model according to experiment settings."""
        logger.info(f"Creating model with {type(self).__name__}")

        from src.models.hf.aloe_models import register_all_aloe_models

        register_all_aloe_models()
        logger.info(
            "ALOE Hugging Face autos ensured; models load via AutoModel / registered classes only."
        )

        self.model = self.get_pretrained_model(verbose=True)
        if model_type == "language":
            logger.info("Created language model: skipping all modifications")
            return self.model

        if self.config.strip_classifier:
            self.model = self._strip_classifier(self.model)
        if self.config.strip_pooler:
            self.model = self._strip_pooler(self.model)

        self.freeze_weights()

        self._load_supervised_checkpoints_from_target_config()

        for ckpt_config in self.config.load_weights_from_local_checkpoints:
            self.load_model_weights_from_checkpoint(
                ckpt_config.path,
                ckpt_config.patterns,
                ckpt_config.ignore_patterns,
            )

        for ckpt_config in self.config.load_weights_from_hf_checkpoints:
            if ckpt_config.map_hf_vision_to_native_aloe:
                self._load_hf_checkpoint_mapped_to_native_aloe(ckpt_config)
            else:
                self.load_model_weights_from_hf_checkpoint(
                    ckpt_config.path,
                    ckpt_config.patterns,
                    ckpt_config.ignore_patterns,
                )

        if self.config.hf_vision_init_checkpoint:
            if self._should_skip_hf_vision_init_checkpoint():
                logger.info(
                    "Skipping hf_vision_init_checkpoint={!r}: backbone or checkpoint lists already "
                    "loaded weights (applying HF vision init would overwrite them).",
                    self.config.hf_vision_init_checkpoint,
                )
            else:
                self._apply_hf_vision_init_checkpoint()

        if self.config.target_model_config.add_hidden_layer_forward_hooks:
            self.model = self._patch_forward_for_hidden_states(self.model)

        self._log_model_creation_summary()
        self.log_model()
        logger.info("Model creation and configuration complete")

        if self.config.compile:
            self.model = torch.compile(self.model, mode=self.config.compile_mode)
            logger.info(f"Model compiled with mode: {self.config.compile_mode}")
        else:
            logger.info("Model compilation disabled")

        return self.model

    def _filter_state_dict_by_patterns(
        self,
        state_dict: Dict[str, torch.Tensor],
        patterns: List[str],
        ignore_patterns: List[str],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if any(re.match(ip, key) for ip in ignore_patterns):
                logger.info(f"Ignored {key}")
                continue
            if any(re.match(pat, key) for pat in patterns):
                out[key] = tensor
        return out

    def _load_filtered_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        patterns: List[str],
        ignore_patterns: List[str] | None = None,
    ) -> None:
        ignore_patterns = ignore_patterns or []
        filtered_state_dict = self._filter_state_dict_by_patterns(
            state_dict,
            patterns,
            ignore_patterns,
        )
        renamed_state_dict = self._rename_sub(filtered_state_dict)
        logger.info(
            f"Added {len(renamed_state_dict)} keys to checkpoint_state_dict. "
            f"{len(state_dict) - len(renamed_state_dict)} keys were removed."
        )
        load_state_dict_into_model(self.model, renamed_state_dict)

    def _should_skip_hf_vision_init_checkpoint(self) -> bool:
        """
        ``hf_vision_init_checkpoint`` maps a vanilla HF vision tower into the native ALOE model.
        It must run only when the trunk is still random (scratch / no trunk weights). If we
        already loaded a backbone or checkpoint lists, applying it would overwrite those weights.
        """
        tmc = self.config.target_model_config
        if getattr(tmc, "backbone_checkpoint", None):
            return True
        if self.config.load_weights_from_local_checkpoints:
            return True
        if self.config.load_weights_from_hf_checkpoints:
            return True
        return False

    def _load_hf_checkpoint_mapped_to_native_aloe(self, ckpt_config: CheckpointLoadConfig) -> None:
        from src.models.hf.aloe.hf_vision_checkpoint_bridge import map_hf_vision_state_dict_to_native_model

        if not ckpt_config.native_aloe_backbone:
            raise ValueError(
                "CheckpointLoadConfig.native_aloe_backbone is required when map_hf_vision_to_native_aloe=True"
            )

        raw = load_hf_automodel_state_dict(
            ckpt_config.path,
            trust_remote_code=ckpt_config.map_hf_trust_remote_code,
            verbose=True,
        )
        subset = self._filter_state_dict_by_patterns(
            raw,
            ckpt_config.patterns,
            ckpt_config.ignore_patterns,
        )
        map_hf_vision_state_dict_to_native_model(
            self.model,
            subset,
            backbone=ckpt_config.native_aloe_backbone,
        )

    def _apply_hf_vision_init_checkpoint(self) -> None:
        from src.models.hf.aloe.hf_vision_checkpoint_bridge import (
            infer_native_aloe_backbone,
            map_hf_vision_state_dict_to_native_model,
        )

        backbone = self.config.hf_vision_init_backbone or infer_native_aloe_backbone(
            self.config.target_model_config
        )
        if backbone is None:
            raise ValueError(
                "Set ModelFactoryConfig.hf_vision_init_backbone or use a target_model_config with type "
                "siglip2 / dinov3 / vit when hf_vision_init_checkpoint is set."
            )

        raw = load_hf_automodel_state_dict(
            self.config.hf_vision_init_checkpoint,
            trust_remote_code=self.config.hf_vision_init_trust_remote_code,
            verbose=True,
        )
        map_hf_vision_state_dict_to_native_model(self.model, raw, backbone=backbone)

    def freeze_weights(self) -> None:
        logger.info("Freezing pretrained model weights")
        keep_trainable = get_check_if_kept_trainable_fn(self.config.keep_pretrained_weights_trainable)

        for name, param in self.model.named_parameters():
            if keep_trainable(name) or not self.config.freeze_pretrained_weights:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, module in self.model.named_modules():
            if keep_trainable(name) or not self.config.freeze_pretrained_weights:
                module.train()
            else:
                module.eval()

    @beartype
    def load_model_weights_from_hf_checkpoint(
        self,
        checkpoint: str,
        use_keys_from_pretrained_statedict: List[str],
        ignore_patterns: List[str] | None = None,
    ) -> None:
        checkpoint_state_dict = load_state_dict_from_hf_model(checkpoint, verbose=True)
        self._load_filtered_state_dict(
            checkpoint_state_dict,
            use_keys_from_pretrained_statedict,
            ignore_patterns,
        )

    @beartype
    def load_model_weights_from_checkpoint(
        self,
        checkpoint: str,
        use_keys_from_pretrained_statedict: List[str],
        ignore_patterns: List[str] | None = None,
    ) -> None:
        checkpoint_state_dict = load_state_dict_from_checkpoint(checkpoint, verbose=True)
        self._load_filtered_state_dict(
            checkpoint_state_dict,
            use_keys_from_pretrained_statedict,
            ignore_patterns,
        )

    def _rename_sub(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for key, replacement in MAPPING_KEYS.items():
            if key in state_dict:
                state_dict[replacement] = state_dict.pop(key)
                logger.info(f"Renamed {key} to {replacement}")
        return state_dict

    def _log_model_creation_summary(self) -> None:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"Model Architecture: {self.config.model_repo}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Frozen parameters: {total_params - trainable_params:,}")

        module_stats: dict[str, dict[str, int]] = {}
        for name, param in self.model.named_parameters():
            parts = name.split(".")
            group = parts[0] if len(parts) > 1 else "root"
            if group not in module_stats:
                module_stats[group] = {"trainable": 0, "frozen": 0, "total": 0}
            module_stats[group]["total"] += param.numel()
            if param.requires_grad:
                module_stats[group]["trainable"] += param.numel()
            else:
                module_stats[group]["frozen"] += param.numel()

        print()
        print("=" * 66)
        print(f"  {'Module':<30} {'Trainable':>10} {'Frozen':>10} {'Total':>10}")
        print("  " + "-" * 62)
        for module_name, stats in sorted(module_stats.items()):
            print(
                f"  {module_name:<30} {stats['trainable']:>10,} {stats['frozen']:>10,} {stats['total']:>10,}"
            )
        print("=" * 66)
        print()

        if self.config.load_weights_from_local_checkpoints:
            logger.info(
                f"Pretrained weights loaded from: {self.config.load_weights_from_local_checkpoints}"
            )
        else:
            logger.info("No pretrained weights loaded from local checkpoints")

        if self.config.freeze_pretrained_weights:
            logger.info("Pretrained weight freezing is ENABLED")
        else:
            logger.info("Pretrained weight freezing is DISABLED")

        if self.config.compile:
            logger.info(f"Model compiled with mode: {self.config.compile_mode}")
        else:
            logger.info("Model compilation disabled")

    def log_model(self) -> None:
        logger.info("Model Architecture\n{}", self.model)

    def _load_supervised_checkpoints_from_target_config(self) -> None:
        target_cfg = self.config.target_model_config
        backbone_checkpoint = getattr(target_cfg, "backbone_checkpoint", None)
        linear_probe_checkpoint = getattr(target_cfg, "linear_probe_checkpoint", None)

        if backbone_checkpoint:
            self.load_model_weights_from_checkpoint(
                backbone_checkpoint,
                [".*"],
                [".*classifier.*", ".*logit_layer.*"],
            )
        if linear_probe_checkpoint:
            self.load_model_weights_from_checkpoint(
                linear_probe_checkpoint,
                [r"classifier\..*", r"logit_layer\..*"],
                [],
            )

    def _wants_image_classification_model(self) -> bool:
        num_labels = getattr(self.config.target_model_config, "num_labels", None)
        return num_labels is not None and int(num_labels) > 0

    def _wants_bcos_head_model(self) -> bool:
        return bool(getattr(self.config.target_model_config, "use_bcos_head", True))

    def _native_aloe_backbone(self) -> Literal["siglip2", "dinov3", "vit"]:
        backbone_type = getattr(self.config.target_model_config, "type", None)
        if backbone_type == "google_vit":
            return "vit"
        if backbone_type in {"siglip2", "dinov3", "vit"}:
            return backbone_type
        raise ValueError(f"Unsupported backbone type for image classification: {backbone_type!r}")

    def get_pretrained_model(self, verbose: bool = False) -> nn.Module:
        if isinstance(self.config.target_model_config, HFModelConfig):
            model = self.get_pretrained_HF_model(
                model_config=self.config.target_model_config,
                verbose=verbose,
            )
        else:
            raise RuntimeError("Only Hugging Face-backed model configs are supported.")
        return model

    def get_pretrained_HF_model(self, model_config: HFModelConfig, verbose: bool = True) -> nn.Module:
        model = self._get_automodel(model_config.name)
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        if verbose:
            logger.info(f"Loaded model {model_config.name} from huggingface")

        if hasattr(model, "vision_model") and model_config.model_part == "vision":
            from src.models.hf.aloe.modeling_aloe_base import AloePreTrainedVisionModel
            from src.models.hf.aloe.modeling_aloe_for_image_classification import (
                AloeForImageClassificationBase,
            )

            if self._wants_image_classification_model() and isinstance(
                model, (AloeForImageClassificationBase, PlainHFImageClassificationModel)
            ):
                logger.info("Keeping Aloe ForImageClassification model for supervised logits / explanations")
            elif isinstance(model, AloeForImageClassificationBase):
                model = model.vision_model
                logger.info(
                    "Extracted vision_model from ALOE ForImageClassification (vision-only for factory)"
                )
            elif isinstance(model, AloePreTrainedVisionModel):
                logger.info("Native ALOE vision encoder; keeping root module")
            else:
                model = model.vision_model
                logger.info("Extracted vision_model from generic HuggingFace multimodal model")
        elif hasattr(model, "text_model") and model_config.model_part == "language":
            model = model.text_model
            logger.info("Extracted text_model from HuggingFace model")

        assert isinstance(model, nn.Module)
        return model

    def _apply_native_aloe_hub_config_overlays(self, config: Any, tmc: Any) -> tuple[bool, bool]:
        if not (isinstance(tmc, VitConfig) and isinstance(tmc, HFModelConfig)):
            return False, False

        overlay = False
        register_resize = False
        if hasattr(config, "aloe_num_registers"):
            desired_reg = int(tmc.register_tokens)
            current_reg = int(getattr(config, "aloe_num_registers", 0))
            if desired_reg != current_reg:
                config.aloe_num_registers = desired_reg
                overlay = True
                register_resize = True

        hid = getattr(tmc, "hidden_act", None)
        if hid is not None and getattr(config, "hidden_act", None) != hid:
            config.hidden_act = hid
            overlay = True

        bcos_impl = getattr(tmc, "aloe_bcos_impl", None)
        if bcos_impl is not None:
            current_impl = getattr(config, "aloe_bcos_impl", None)
            if current_impl != bcos_impl:
                setattr(config, "aloe_bcos_impl", bcos_impl)
                overlay = True

        return overlay, register_resize

    def _get_automodel(self, name: str) -> nn.Module:
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError("transformers required: pip install transformers") from exc

        tmc = self.config.target_model_config
        wants_classifier = self._wants_image_classification_model()
        extra: dict[str, Any] = {}
        if self.config.attn_implementation is not None:
            extra["attn_implementation"] = self.config.attn_implementation
        if isinstance(tmc, HFModelConfig) and tmc.trust_remote_code:
            extra["trust_remote_code"] = True

        if self.config.load_pretrained_weights:
            if wants_classifier:
                if self._wants_bcos_head_model():
                    from src.models.hf.aloe_models import build_aloe_config_from_factory

                    aloe_cfg = build_aloe_config_from_factory(
                        self.config,
                        backbone=self._native_aloe_backbone(),
                    )
                    aloe_cfg.num_labels = int(tmc.num_labels)
                    extra["config"] = aloe_cfg
                    extra["ignore_mismatched_sizes"] = True
                    model = AutoModelForImageClassification.from_pretrained(name, **extra)
                else:
                    backbone_model = AutoModel.from_pretrained(name, **extra)
                    model = PlainHFImageClassificationModel(
                        backbone_model,
                        feature_dim=int(getattr(tmc, "feature_dim", 768)),
                        num_labels=int(tmc.num_labels),
                    )
            else:
                if isinstance(tmc, VitConfig) and isinstance(tmc, HFModelConfig):
                    cfg_kw_pre: dict[str, Any] = {}
                    if tmc.trust_remote_code:
                        cfg_kw_pre["trust_remote_code"] = True
                    tmp_cfg = AutoConfig.from_pretrained(name, **cfg_kw_pre)
                    overlay, register_resize = self._apply_native_aloe_hub_config_overlays(tmp_cfg, tmc)
                    if overlay:
                        extra["config"] = tmp_cfg
                        extra["ignore_mismatched_sizes"] = register_resize
                model = AutoModel.from_pretrained(name, **extra)
            logger.info("Loaded pretrained weights from HuggingFace")
        else:
            model = self._get_model_from_autoconfig(name)
            logger.info(
                "No pretrained weights loaded from HuggingFace: instantiating fresh model."
            )

        return model

    def _get_model_from_autoconfig(self, name: str):
        try:
            from transformers import AutoConfig, AutoModel, AutoModelForImageClassification
        except ImportError as exc:
            raise ImportError("transformers required: pip install transformers") from exc

        tmc = self.config.target_model_config
        wants_classifier = self._wants_image_classification_model()
        if wants_classifier:
            if self._wants_bcos_head_model():
                from src.models.hf.aloe_models import build_aloe_config_from_factory

                config = build_aloe_config_from_factory(
                    self.config,
                    backbone=self._native_aloe_backbone(),
                )
                config.num_labels = int(tmc.num_labels)
                model = AutoModelForImageClassification.from_config(config)
                logger.info(f"Created image-classification model {name} from ALOE config with reinitialized parameters")
                return model

            cfg_kw: dict[str, Any] = {}
            if isinstance(tmc, HFModelConfig) and tmc.trust_remote_code:
                cfg_kw["trust_remote_code"] = True
            config = AutoConfig.from_pretrained(name, **cfg_kw)
            backbone_model = AutoModel.from_config(config)
            model = PlainHFImageClassificationModel(
                backbone_model,
                feature_dim=int(getattr(tmc, "feature_dim", 768)),
                num_labels=int(tmc.num_labels),
            )
            logger.info(
                f"Created image-classification model {name} from plain HF backbone + linear head with reinitialized parameters"
            )
            return model

        cfg_kw: dict[str, Any] = {}
        if isinstance(tmc, HFModelConfig) and tmc.trust_remote_code:
            cfg_kw["trust_remote_code"] = True
        config = AutoConfig.from_pretrained(name, **cfg_kw)
        overlay, register_resize = self._apply_native_aloe_hub_config_overlays(config, tmc)
        if overlay:
            logger.info(
                f"Instantiating {name} from AutoConfig with native overlays: "
                f"aloe_num_registers={getattr(config, 'aloe_num_registers', '?')}, "
                f"hidden_act={getattr(config, 'hidden_act', '?')}, "
                f"aloe_bcos_impl={getattr(config, 'aloe_bcos_impl', '?')}"
                + (
                    "; register count differs from Hub config.json"
                    if register_resize
                    else ""
                )
            )
        model = AutoModel.from_config(config)
        logger.info(f"Created model {name} from AutoConfig with reinitialized parameters")
        return model

    def _strip_classifier(self, model: nn.Module) -> nn.Module:
        module_path = None
        for name, module in model.named_modules():
            for pattern in CLASSIFIER_KEYS:
                if re.match(pattern, name):
                    module_path = name
                    break
            if module_path:
                break
        logger.info(f"Found classifier at {module_path}")
        if module_path is None:
            return model
        self._replace_module(model, module_path, nn.Identity())
        return model

    def _strip_pooler(self, model: nn.Module) -> nn.Module:
        tmc = self.config.target_model_config
        if getattr(tmc, "cls_token", True) is False:
            logger.info(
                "Skipping pooler stripping for {}: cls_token=False, so CLSPooler would discard the native pooled representation",
                type(tmc).__name__,
            )
            return model

        module_path = None
        for name, module in model.named_modules():
            for pattern in POOLER_KEYS:
                if re.match(pattern, name):
                    module_path = name
                    break
            if module_path:
                break
        logger.info(f"Found pooler at {module_path}")
        if module_path is None:
            return model
        self._replace_module(model, module_path, CLSPooler())
        return model

    @staticmethod
    def _replace_module(model: nn.Module, module_path: str, replacement: nn.Module) -> None:
        parent_path, _, child_name = module_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        setattr(parent, child_name, replacement)

    def _patch_forward_for_hidden_states(self, model: nn.Module) -> nn.Module:
        try:
            from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
        except ImportError:
            DINOv3ViTModel = None

        if DINOv3ViTModel is None or not isinstance(model, DINOv3ViTModel):
            logger.info(
                "Skipping hidden-states forward patch: {} is not DINOv3ViTModel",
                type(model).__name__,
            )
            return model

        def patched_forward(
            self_model: nn.Module,
            pixel_values: "torch.Tensor",
            bool_masked_pos=None,
            head_mask=None,
            output_hidden_states: Optional[bool] = None,
            **kwargs,
        ) -> "BaseModelOutputWithPooling":
            pixel_values = pixel_values.to(self_model.embeddings.patch_embeddings.weight.dtype)
            hidden_states = self_model.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
            position_embeddings = self_model.rope_embeddings(pixel_values)
            all_hidden_states = () if output_hidden_states else None

            for i, layer_module in enumerate(self_model.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                layer_head_mask = head_mask[i] if head_mask is not None else None
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=layer_head_mask,
                    position_embeddings=position_embeddings,
                )

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            sequence_output = self_model.norm(hidden_states)
            pooled_output = sequence_output[:, 0, :]
            return BaseModelOutputWithPooling(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                hidden_states=all_hidden_states,
            )

        model.forward = types.MethodType(patched_forward, model)
        logger.info(
            "Monkey-patched {}.forward to collect hidden states in-loop (no hooks)",
            type(model).__name__,
        )
        return model
ModelFactory = HfModelFactory
