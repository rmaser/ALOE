
from src.training.config import OptimizerConfig, SchedulerConfig
import importlib
from typing import Any

def get_class_from_path(path: str) -> Any:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def construct_optimizer(cfg: OptimizerConfig, params):
    optimizer_cls = get_class_from_path(cfg.target)
    # exclude target from kwargs
    kwargs = cfg.model_dump(exclude_none=True, exclude={"target"})
    optimizer = optimizer_cls(params=params, **kwargs)
    return optimizer

def construct_scheduler(cfg: SchedulerConfig, optimizer):
    scheduler_cls = get_class_from_path(cfg.target)
    # exclude target from kwargs
    kwargs = cfg.model_dump(exclude_none=True, exclude={"target", "model_params"})
    scheduler = scheduler_cls(optimizer=optimizer, **kwargs)
    return scheduler