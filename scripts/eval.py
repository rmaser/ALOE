import sys
from pathlib import Path
from typing import Any, Dict
import inspect
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.eval.zero_shot_templates import IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
from src.models.model_factory import ModelFactory
from src.models.model_factory.config import ModelFactoryConfig
from src.utils.omegaconf_resolvers import register_aloe_omegaconf_resolvers

register_aloe_omegaconf_resolvers()

torch.set_float32_matmul_precision("high")


def _finalize_and_wait_for_wandb(
    trainer,
    status: str = "success",
    timeout_s: int = 180,
    poll_interval_s: float = 2.0,
) -> None:
    """Best-effort flush: finalize W&B run and wait for completion state."""
    logger = getattr(trainer, "logger", None)
    if logger is None:
        return
    if not hasattr(logger, "experiment"):
        if hasattr(logger, "finalize"):
            logger.finalize(status)
        return

    experiment = logger.experiment
    if experiment is None:
        return

    # Ensure finalize is called (Lightning API), then ask the W&B run to finish.
    if hasattr(logger, "finalize"):
        logger.finalize(status)
    finish_fn = getattr(experiment, "finish", None)
    if callable(finish_fn):
        try:
            finish_fn(exit_code=0, quiet=True)
        except TypeError:
            finish_fn()

    # Busy-wait loop (requested): wait for run state to settle.
    # Note: ``finish()`` is usually blocking; this loop is a safety net.
    deadline = time.time() + max(0, int(timeout_s))
    while time.time() < deadline:
        is_finished = bool(getattr(experiment, "_is_finished", False))
        is_running = bool(getattr(experiment, "_is_running", False))
        if is_finished or not is_running:
            print("W&B finalize complete.")
            return
        remaining = int(deadline - time.time())
        print(f"Waiting for W&B to report metrics... ({remaining}s left)")
        time.sleep(max(0.1, float(poll_interval_s)))

    print("Warning: Timed out waiting for W&B finalize; exiting anyway.")


def run_single_evaluator(
    evaluator_name: str,
    evaluator_cfg: DictConfig,
    model: torch.nn.Module,
    datamodule,
    trainer,
    cfg: DictConfig = None,
) -> Dict[str, Any]:
    """Run a single evaluator and return metrics."""
    print(f"\n{'=' * 50}")
    print(f"Running evaluator: {evaluator_name}")
    print(f"Target: {evaluator_cfg._target_}")
    print(f"{'=' * 50}")

    # Check if this evaluator requires the model in __init__ (e.g., GridPGEvaluator, PixelDeletionEvaluator)
    evaluator_target = evaluator_cfg._target_
    needs_model_in_init = any(
        name in evaluator_target
        for name in ["GridPGEvaluator", "PixelDeletionEvaluator"]
    )

    metrics = {}
    try:
        if needs_model_in_init:
            # These evaluators take lightning_module in __init__
            evaluator = hydra.utils.instantiate(evaluator_cfg, model=model)
            # Get dataloader for evaluation
            if "grid" in evaluator_name:
                dataloader = datamodule.grid_pg_val_dataloader()
            else:
                dataloader = datamodule.val_dataloader()
            if isinstance(dataloader, list):
                dataloader = dataloader[0]
            evaluate_sig = inspect.signature(evaluator.evaluate)
            evaluate_kwargs = {}
            if "logger" in evaluate_sig.parameters and getattr(trainer, "logger", None):
                evaluate_kwargs["logger"] = trainer.logger
            if "metric_prefix" in evaluate_sig.parameters:
                evaluate_kwargs["metric_prefix"] = evaluator_name
            metrics = evaluator.evaluate(dataloader, **evaluate_kwargs)
            # Convert tensor outputs to scalar metrics if needed
            if isinstance(metrics, dict):
                processed_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        processed_metrics[f"mean_{k}"] = v.mean().item() if v.numel() > 0 else 0.0
                    else:
                        processed_metrics[k] = v
                metrics = processed_metrics
            elif isinstance(metrics, (int, float)):
                metrics = {"score": metrics}
        else:
            # Standard evaluators that take model via evaluate()
            evaluator = hydra.utils.instantiate(evaluator_cfg)

            # Special case for ZeroShotEvaluator: needs class prompts
            if "ZeroShotEvaluator" in evaluator_target or evaluator_name == "zero_shot":
                if "data" in cfg and "dataset" in cfg.data:
                    ds_conf = cfg.data.dataset
                    if isinstance(ds_conf, dict):
                        dataset_name = ds_conf.get("dataset_name", "")
                    elif isinstance(ds_conf, DictConfig):
                        dataset_name = ds_conf.get("dataset_name", "")
                    else:
                        dataset_name = str(ds_conf)
                else:
                    dataset_name = ""

                if dataset_name in ["imagenet1k", "ILSVRC/imagenet-1k"]:
                    print("Configuring ZeroShotEvaluator for ImageNet1k...")

                    if hasattr(evaluator, "set_prompts_from_labels"):
                        # We can also pass templates if the method supports it?
                        # Usually set_prompts_from_labels(labels, templates)
                        # Let's try passing templates if supported.
                        sig = inspect.signature(evaluator.set_prompts_from_labels)
                        if "templates" in sig.parameters:
                            evaluator.set_prompts_from_labels(
                                IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES
                            )
                        else:
                            evaluator.set_prompts_from_labels(IMAGENET_CLASSNAMES)
                else:
                    print(
                        f"Skipping ZeroShotEvaluator: dataset '{dataset_name}' is not imagenet1k."
                    )
                    metrics = {"skipped": True}
                    prefixed_metrics = {
                        f"{evaluator_name}/{k}": v for k, v in metrics.items()
                    }
                    print(f"Results for {evaluator_name}: {metrics}")
                    return prefixed_metrics

            metrics = evaluator.evaluate(
                model=model, datamodule=datamodule, trainer=trainer
            )
    except Exception as e:
        print(f"Error running evaluator {evaluator_name}: {e}")
        import traceback

        traceback.print_exc()
        metrics = {"error": str(e)}

    # Prefix metrics with evaluator name
    prefixed_metrics = {f"{evaluator_name}/{k}": v for k, v in metrics.items()}
    print(f"Results for {evaluator_name}: {metrics}")

    return prefixed_metrics


@hydra.main(version_base=None, config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    # Print config
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Creating model with ModelFactory...")

    # Check if we should load config from a checkpoint
    if "ckpt_path" in cfg and cfg.ckpt_path is not None:
        print(f"Loading ModelFactoryConfig from checkpoint: {cfg.ckpt_path}")
        model_factory_config = cfg.model_factory_config

        print(f"Injecting checkpoint path into config to load weights: {cfg.ckpt_path}")
        model_factory_config.load_weights_from_local_checkpoints.append(
            {"path": cfg.ckpt_path, "patterns": [".*"], "ignore_patterns": []}
        )

    else:
        if "model_factory" not in cfg:
            raise ValueError(
                "Config must include 'model_factory' (e.g. experiment=eval/embeddings/base_models)."
            )

        model_factory_dict: Dict[str, Any] = OmegaConf.to_container(
            cfg.model_factory, resolve=True
        )  # type: ignore

        if (
            "target_model_config" in model_factory_dict
            and "_target_" in model_factory_dict["target_model_config"]
        ):
            model_factory_dict["target_model_config"] = hydra.utils.instantiate(
                cfg.model_factory.target_model_config
            )
        model_factory_config = ModelFactoryConfig(**model_factory_dict)

    def create_fresh_model():
        factory = ModelFactory(model_factory_config)
        model = factory.create_model()
        model.eval()
        model.to(device)
        model.cfg = cfg
        print(f"Model created and moved to {device}")
        return model

    # Instantiate datamodule
    print("Instantiating datamodule")
    datamodule = hydra.utils.instantiate(cfg.data)
    if hasattr(datamodule, "setup"):
        try:
            datamodule.setup(stage="validate")
        except TypeError:
            datamodule.setup()

    # Initialize Logger and Trainer
    logger = True
    if "logger" in cfg:
        logger = hydra.utils.instantiate(cfg.logger)

    import lightning.pytorch as L

    trainer = L.Trainer(
        accelerator="auto", devices=1, logger=logger, enable_checkpointing=False
    )

    # Update WandB config if available
    if hasattr(trainer.logger, "experiment") and hasattr(
        trainer.logger.experiment, "config"
    ):
        print("Updating WandB config with Hydra configuration...")
        try:
            # Resolve config to simple dict
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)

            # if "explainer" in cfg_dict and "_target_" in cfg_dict["explainer"]:
            #     explainer_class = cfg_dict["explainer"]["_target_"].split(".")[-1]
            #     cfg_dict["explainer_name"] = explainer_class

            trainer.logger.experiment.config.update(cfg_dict, allow_val_change=True)
        except Exception as e:
            print(f"Warning: Failed to update WandB config: {e}")

    # Collect all metrics
    all_metrics = {}

    # Check for list-based config (experiment configs)
    if "evaluators" in cfg and "eval_list" in cfg:
        eval_list = list(cfg.eval_list)
        print(f"\nRunning {len(eval_list)} evaluators: {eval_list}")

        for evaluator_name in eval_list:
            if evaluator_name not in cfg.evaluators:
                print(
                    f"Warning: Evaluator '{evaluator_name}' not found in evaluators config, skipping."
                )
                continue

            evaluator_cfg = cfg.evaluators[evaluator_name]

            # Create fresh model for this evaluator
            print(f"Creating fresh model for {evaluator_name}...")
            model = create_fresh_model()

            metrics = run_single_evaluator(
                evaluator_name, evaluator_cfg, model, datamodule, trainer, cfg
            )

            # Clean up model to ensure next evaluator gets a fresh one (and free GPU memory)
            del model
            torch.cuda.empty_cache()

            all_metrics.update(metrics)
            if trainer.logger:
                trainer.logger.log_metrics(metrics)

    elif "evaluator" in cfg:
        # Single-evaluator fallback
        model = create_fresh_model()
        metrics = run_single_evaluator(
            "evaluator", cfg.evaluator, model, datamodule, trainer, cfg
        )
        all_metrics.update(metrics)
        if trainer.logger:
            trainer.logger.log_metrics(metrics)

    else:
        raise ValueError(
            "No evaluator(s) configured. Use 'evaluator' or 'evaluators' + 'eval_list'."
        )

    # Print summary
    print(f"\n{'=' * 50}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 50}")
    for k, v in all_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    # Log metrics to W&B
    if all_metrics and trainer.logger:
        trainer.logger.log_metrics(all_metrics)
    _finalize_and_wait_for_wandb(trainer, status="success")

    return all_metrics


if __name__ == "__main__":
    main()
