from typing import Any, Optional

from pydantic import BaseModel, ConfigDict

from src.models.model_factory import ModelFactoryConfig


class SchedulerConfig(BaseModel):
    target: str = "torch.optim.lr_scheduler.OneCycleLR"
    model_config = ConfigDict(extra="allow")


class OptimizerConfig(BaseModel):
    target: str = "torch.optim.Adam"
    lr: float
    weight_decay: Optional[float] = None


class LightningModuleConfig(BaseModel):
    optimizer_config: OptimizerConfig
    scheduler_config: SchedulerConfig
    student_model_factory_config: ModelFactoryConfig
    teacher_model_factory_config: ModelFactoryConfig
    loss_fn: Any
    distillation_layers: list[int] = []
    distill_pooling_output: bool = True
    drop_register_tokens: bool = False
    strip_aloe_register_prefix: bool = True


class BcosifyLightningModuleConfig(BaseModel):
    optimizer_config: OptimizerConfig
    scheduler_config: Optional[SchedulerConfig] = None
    student_model_factory_config: ModelFactoryConfig
