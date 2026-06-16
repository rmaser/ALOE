from bcos.modules.losses import UniformOffLabelsBCEWithLogitsLoss
from typing import Tuple
from typing import Any
from typing import Dict
from typing import Union
from typing import Optional
import pytorch_lightning as pl
import torch
import torch.nn as nn
from src.training.config import (
    LightningModuleConfig,
    BcosifyLightningModuleConfig,
)
from src.training.utils import construct_optimizer, construct_scheduler
from src.models.model_factory import ModelFactory
from transformers.modeling_outputs import BaseModelOutputWithPooling
import beartype
from beartype import beartype as typechecker
from jaxtyping import Float, jaxtyped
from torch import Tensor
from torch.nn import functional as F
import torchmetrics


@jaxtyped(typechecker=typechecker)
def _pixel_values_for_hf_teacher(x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch _ height width"]:
    """
    Vanilla HF vision models (SigLIP, ViT, …) expect 3 input channels.

    The student dataloader typically uses the student's image processor (e.g.
    :class:`~src.models.hf.aloe.image_processing_aloe.AloeImageProcessor` with
    ``expand_to_6ch=True``) or ``HFTransform`` with ``is_bcos=True``, producing
    ``[x, 1-x]`` on the normalised RGB tensor.  The first three channels are the
    same normalised RGB the teacher backbone was trained on.
    """
    if x.ndim == 4 and x.shape[1] == 6:
        return x[:, :3].contiguous()
    return x


@beartype.beartype
def _num_register_tokens(module: nn.Module) -> int:
    """HF vision configs use ``num_register_tokens``; native ALOE uses ``aloe_num_registers``."""
    cfg = getattr(module, "config", None)
    if cfg is None:
        return 0
    n_hf = int(getattr(cfg, "num_register_tokens", 0) or 0)
    if n_hf > 0:
        return n_hf
    return int(getattr(cfg, "aloe_num_registers", 0) or 0)


@jaxtyped(typechecker=typechecker)
def _strip_leading_aloe_registers(
    student_states: Float[Tensor, "batch seq hidden"],
    student_model: nn.Module,
    teacher_model: nn.Module,
) -> Float[Tensor, "batch seq_out hidden"]:
    """
    Remove leading Aloe register tokens from student sequence when the teacher has
    no register tokens, so patch streams align for distillation.
    """
    s_cfg = getattr(student_model, "config", None)
    t_cfg = getattr(teacher_model, "config", None)
    n_student = int(getattr(s_cfg, "aloe_num_registers", 0) or 0) if s_cfg is not None else 0
    if n_student <= 0:
        return student_states

    n_teacher_aloe = int(getattr(t_cfg, "aloe_num_registers", 0) or 0) if t_cfg is not None else 0
    n_teacher_hf = int(getattr(t_cfg, "num_register_tokens", 0) or 0) if t_cfg is not None else 0
    if n_teacher_aloe > 0 or n_teacher_hf > 0:
        return student_states

    return student_states[:, n_student:, :]


def _set_scheduler_total_steps(scheduler_config: Any, total_steps: int) -> None:
    target = str(scheduler_config.target)
    if "OneCycleLR".lower() in target.lower():
        setattr(scheduler_config, "total_steps", total_steps)
    elif "CosineAnnealingLR".lower() in target.lower():
        setattr(scheduler_config, "T_max", total_steps)
    elif "create_warmup_cosine_scheduler" in target:
        setattr(scheduler_config, "total_epochs", total_steps)
    elif "create_warmup_constant_cosine_scheduler" in target:
        # Step counts are explicit in the Hydra config.
        pass



class DistillModel(pl.LightningModule):
    def __init__(self, cfg: LightningModuleConfig):
        super().__init__()
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg

        self.student_model_factory = ModelFactory(cfg.student_model_factory_config)
        self.student_model = self.student_model_factory.create_model()

        self.teacher_model_factory = ModelFactory(cfg.teacher_model_factory_config)
        self.teacher_model = self.teacher_model_factory.create_model()
        

        self.set_up_loss_fn()

    def explain(self, in_tensor: Tensor, idx=None, **kwargs) -> Dict[str, Any]:
        """
        Delegates explanation to the student model.
        """
        if hasattr(self.student_model, "explain"):
            return self.student_model.explain(in_tensor, idx=idx, **kwargs)
        else:
            raise NotImplementedError("Student model does not support explanation.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # Remove teacher model from checkpoint to save space
        # The teacher is frozen and can be reloaded from the factory/config
        keys_to_remove = [k for k in checkpoint["state_dict"].keys() if k.startswith("teacher_model.")]
        for k in keys_to_remove:
            del checkpoint["state_dict"][k]

    @staticmethod
    def _is_allowed_missing_teacher_key(key: str) -> bool:
        return key.startswith("teacher_model.")

    def load_state_dict(self, state_dict, strict: bool = True):
        incompatible_keys = super().load_state_dict(state_dict, strict=False)
        missing_keys = list(incompatible_keys.missing_keys)
        unexpected_keys = list(incompatible_keys.unexpected_keys)

        filtered_missing = [
            key for key in missing_keys if not self._is_allowed_missing_teacher_key(key)
        ]

        if strict and (filtered_missing or unexpected_keys):
            error_msgs = []
            if filtered_missing:
                error_msgs.append(
                    "Missing key(s) in state_dict: {}.".format(
                        ", ".join(f'"{key}"' for key in filtered_missing)
                    )
                )
            if unexpected_keys:
                error_msgs.append(
                    "Unexpected key(s) in state_dict: {}.".format(
                        ", ".join(f'"{key}"' for key in unexpected_keys)
                    )
                )
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    self.__class__.__name__, "\n\t".join(error_msgs)
                )
            )

        return incompatible_keys.__class__(filtered_missing, unexpected_keys)
        
    def training_step(self, batch, batch_idx, *args, **kwargs): # type: ignore[override]
        raise NotImplementedError("Need to implement training_step method")

    def configure_optimizers(self):
        optimizer = construct_optimizer(self.cfg.optimizer_config, self.student_model.parameters())
        _set_scheduler_total_steps(self.cfg.scheduler_config, self.trainer.estimated_stepping_batches)

        scheduler = construct_scheduler(self.cfg.scheduler_config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def set_up_loss_fn(self):
        # instantiate loss function directly from string name, can be local losses as well
        self.loss_fn = self.cfg.loss_fn

class DistillHFModel(DistillModel):
    def __init__(self, cfg: LightningModuleConfig):
        super().__init__(cfg)

    @beartype.beartype
    def forward_student(self, batch: Union[Dict[str, Any], Tuple[Tensor, Optional[int]]]) -> tuple[BaseModelOutputWithPooling, Optional[int | Tensor]]:
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch.get("label", None)
        else:
            x, y = batch
        
        student_prediction: BaseModelOutputWithPooling = self.student_model(x, output_hidden_states=True)
        return student_prediction, y
    
    @beartype.beartype
    def forward_teacher(self, batch: Union[Dict[str, Any], Tuple[Tensor, Optional[int]]]) -> tuple[BaseModelOutputWithPooling, Optional[int]]:
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch.get("label", None)
        else:
            x, y = batch
        
        # If required interpolate teacher image size
        interpolation_value = self.cfg.teacher_model_factory_config.target_model_config.interpolate_teacher_image_size
        if interpolation_value:
            x = F.interpolate(x, size=interpolation_value, mode="bilinear", align_corners=False, antialias=True)

        x = _pixel_values_for_hf_teacher(x)

        with torch.no_grad():
            teacher_prediction: BaseModelOutputWithPooling = self.teacher_model(x, output_hidden_states=True, interpolate_pos_encoding=True)
        return teacher_prediction, y
    
    def training_step(self, batch, batch_idx, *args, **kwargs): # type: ignore[override]
        student_prediction, y = self.forward_student(batch)
        teacher_prediction, _ = self.forward_teacher(batch)
            
        return self._compute_distillation_loss(student_prediction, teacher_prediction, y, "train")
    
    def validation_step(self, batch, batch_idx, *args, **kwargs): # type: ignore[override]
        student_prediction, y = self.forward_student(batch)
        teacher_prediction, _ = self.forward_teacher(batch)

        return self._compute_distillation_loss(student_prediction, teacher_prediction, y, "val")
    
    def _compute_distillation_loss(self, student_prediction, teacher_prediction, y, split="train"):
        loss: Tensor = torch.tensor(0, dtype=torch.float32, device=self.device)
        
        for layer in self.cfg.distillation_layers:
            student_states = student_prediction.hidden_states[layer]
            teacher_states = teacher_prediction.hidden_states[layer]

            if getattr(self.cfg, "strip_aloe_register_prefix", True):
                student_states = _strip_leading_aloe_registers(
                    student_states, self.student_model, self.teacher_model
                )

            if getattr(self.cfg, "drop_register_tokens", False):
                # DINOv2/v3 models typically place register tokens immediately after the CLS token
                num_reg = _num_register_tokens(self.teacher_model)
                if num_reg > 0:
                    teacher_states = torch.cat([teacher_states[:, :1], teacher_states[:, 1+num_reg:]], dim=1)

                num_reg_student = _num_register_tokens(self.student_model)
                if num_reg_student > 0:
                    student_states = torch.cat([student_states[:, :1], student_states[:, 1+num_reg_student:]], dim=1)

            # Align end-lengths (drops extra tokens from the front if lengths still mismatch)
            min_len = min(teacher_states.size(1), student_states.size(1))
            teacher_states = teacher_states[:, -min_len:]
            student_states = student_states[:, -min_len:]

            layer_loss = torch.mean(self.loss_fn(student_states, teacher_states))
            self.log(f"loss_layers/{split}_layer_{layer}", layer_loss.item())
            loss += layer_loss


        if self.cfg.distill_pooling_output:
            pooling_loss = torch.mean(self.loss_fn(student_prediction.pooler_output.unsqueeze(1), teacher_prediction.pooler_output.unsqueeze(1)))
            self.log(f"loss_pooler/{split}_pooler", pooling_loss.item())
            loss += pooling_loss
        
        self.log(f"loss_total/{split}_total", loss.item())
        return loss



class BcosifyModel(pl.LightningModule):
    def __init__(self, cfg: BcosifyLightningModuleConfig):
        super().__init__()
        self.save_hyperparameters(cfg.model_dump())
        self.cfg = cfg

        self.student_model_factory = ModelFactory(cfg.student_model_factory_config)
        self.student_model = self.student_model_factory.create_model()
        
        # Instantiate loss object
        self.loss_fn = UniformOffLabelsBCEWithLogitsLoss()
        target_cfg = self.cfg.student_model_factory_config.target_model_config
        num_classes = getattr(
            self.cfg.student_model_factory_config,
            "num_classes",
            None,
        ) or getattr(target_cfg, "num_labels", None)
        if num_classes is None:
            raise ValueError(
                "BcosifyHFModel requires num_labels on the target model config "
                "or num_classes on the student factory config."
            )
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=int(num_classes))
        
    def training_step(self, batch, batch_idx, *args, **kwargs): # type: ignore[override]
        raise NotImplementedError("Need to implement training_step method")

    def configure_optimizers(self):
        optimizer = construct_optimizer(self.cfg.optimizer_config, self.parameters())
        if self.cfg.scheduler_config is None:
            from .scheduler import create_warmup_cosine_scheduler

            scheduler = create_warmup_cosine_scheduler(
                optimizer,
                total_epochs=self.trainer.estimated_stepping_batches,
                warmup_epochs=10000,
            )
        else:
            _set_scheduler_total_steps(self.cfg.scheduler_config, self.trainer.estimated_stepping_batches)
            scheduler = construct_scheduler(self.cfg.scheduler_config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


class BcosifyHFModel(BcosifyModel):
    def __init__(self, cfg: BcosifyLightningModuleConfig):
        super().__init__(cfg)

    @beartype.beartype
    def forward_student(self, batch: Union[Dict[str, Any], Tuple[Tensor, Optional[int]]]) -> tuple[Tensor, Tensor]:
        if isinstance(batch, dict):
            x = batch["image"]
            y = batch.get("label", None)
        else:
            x, y = batch
        
        assert isinstance(y, Tensor)
        # Extract features
        out = self.student_model(x)
        logits = out.logits if hasattr(out, "logits") else out
        return logits, y
    
    
    def training_step(self, batch, batch_idx, *args, **kwargs): # type: ignore[override]
        student_prediction, y = self.forward_student(batch)
        prefix = "train_"
        loss = self.loss_fn(student_prediction, y)
        acc = self.accuracy(student_prediction, y)
        self.log(prefix + "loss", loss)
        self.log(prefix + "acc", acc)
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs): # type: ignore[override]
        student_prediction, y = self.forward_student(batch)
        prefix = "val_"
        loss = self.loss_fn(student_prediction, y)  
        acc = self.accuracy(student_prediction, y)
        self.log(prefix + "loss", loss)
        self.log(prefix + "acc", acc)
        return loss
