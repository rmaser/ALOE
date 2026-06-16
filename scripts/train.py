import hydra
import os
import sys

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from src.utils.checkpoint_safety import register_trusted_checkpoint_safe_globals
from src.utils.omegaconf_resolvers import register_aloe_omegaconf_resolvers

register_aloe_omegaconf_resolvers()


def _local_device_count(devices) -> int:
    if devices == "auto":
        return torch.cuda.device_count()
    if isinstance(devices, list | ListConfig):
        return len(devices)
    return int(devices)


def _scale_batch_size_for_distributed_training(cfg: DictConfig) -> None:
    if "data" not in cfg or "datamodule_config" not in cfg.data:
        return
    if "dataloader_config" not in cfg.data.datamodule_config:
        return

    global_batch_size = cfg.data.datamodule_config.dataloader_config.batch_size
    devices = cfg.trainer.get("devices", 1)
    num_devices = _local_device_count(devices)
    num_nodes = int(cfg.trainer.get("num_nodes", 1))
    world_size = num_devices * num_nodes

    if world_size <= 1:
        return
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} must be divisible by distributed "
            f"world size {world_size} ({num_devices} devices/node * {num_nodes} nodes)."
        )

    per_device_batch_size = global_batch_size // world_size
    print(
        "Scaling batch size: "
        f"Global {global_batch_size} / {world_size} processes "
        f"({num_devices} devices/node * {num_nodes} nodes) = "
        f"{per_device_batch_size} per device"
    )
    cfg.data.datamodule_config.dataloader_config.batch_size = per_device_batch_size


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Resolve config
    OmegaConf.resolve(cfg)
    
    # Unify logging
    # from src.utils.logging_utils import setup_loguru_logging_intercept
    # import logging
    # setup_loguru_logging_intercept(level=logging.INFO, modules=("pytorch_lightning", "wandb"))

    # Set matmul precision
    torch.set_float32_matmul_precision('high')

    pl.seed_everything(cfg.get("seed", 42))
    
    pl.seed_everything(cfg.get("seed", 42))
    
    # Scale configured global batch size to the per-process batch expected by DDP.
    _scale_batch_size_for_distributed_training(cfg)

    # Initialize datamodule
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # Initialize logger
    # Check for SLURM resumption nugget
    from src.slurm.resume import check_slurm_resume, create_slurm_nugget, find_latest_checkpoint_for_run_id
    check_slurm_resume(cfg)

    logger = hydra.utils.instantiate(cfg.get("logger"))
    
    # Create nugget if it doesn't exist
    create_slurm_nugget(cfg, logger)
    
    # Log the full config to WandB
    if logger:
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    
    # Initialize model
    # We use hydra.utils.instantiate to create the model from the config
    model = hydra.utils.instantiate(cfg.model)
    
    # Initialize callbacks
    callbacks = []
    if "callbacks" in cfg:
        for _, callback_cfg in cfg.callbacks.items():
            if "_target_" in callback_cfg:
                callbacks.append(hydra.utils.instantiate(callback_cfg))

    # Initialize trainer
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    
    # Resume from checkpoint if ID is provided.
    ckpt_path = None
    if cfg.get("logger") and cfg.logger.get("id"):
        from src.utils.env import get_cache_path
        run_id = cfg.logger.id
        cache_path = get_cache_path()
        ckpt = find_latest_checkpoint_for_run_id(run_id, cache_path=cache_path)
        if ckpt is None:
            msg = (
                f"No checkpoint found for WandB ID: {run_id} under {cache_path}. "
                "Looked for both wandb run dirs and direct run-id checkpoint dirs."
            )
            print(msg)
            raise FileNotFoundError(msg)
        ckpt_path = str(ckpt)
        print(f"Resuming from latest checkpoint: {ckpt_path}")
        register_trusted_checkpoint_safe_globals()

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
