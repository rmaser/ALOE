# ruff: noqa: F722

from functools import partial
from typing import Optional, Dict, Callable, Any
import torch
import torch.nn.functional as F
import tqdm
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from pathlib import Path

from src.explanation.abstract import BaseExplainer
from src.explainability.grid_score.grid_pg_processor import GridPGProcessor
from src.explainability.grid_score.config import GridProcessorConfig


class GridPGEvaluator:
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        explainer: Optional["BaseExplainer"] = None,
        max_batches: Optional[int] = None,
        confidence_threshold: float = 0.95,
        prediction_fn: Optional[Callable] = None,
        map_size: int = 2,
        log_every_n_batches: int = 10,
    ):
        self.model = model
        self.device = device
        self.explainer = explainer
        self.max_batches = max_batches
        self.confidence_threshold = confidence_threshold
        self.log_every_n_batches = max(1, int(log_every_n_batches))
        
        # Determine prediction function if not provided.
        # Match repo convention:
        # - ALOE/B-cos classifier heads (BCE-style) -> sigmoid
        # - Plain HF CE classifiers -> softmax
        if prediction_fn is None:
            uses_bce_style_head = hasattr(self.model, "logit_layer")
            if uses_bce_style_head:
                prediction_fn = torch.sigmoid
            else:
                prediction_fn = partial(F.softmax, dim=-1)

        def explain_wrapper(inp, explain_class=None, **kwargs):
            # Filter kwargs for generic explainers
            # Generic explainers (Captum-based) do not support mode/map_size and forward them to gradient_to_image which crashes.
            # ``return_explanation`` is a shared metric-path hint and should be preserved so explainers can skip unused renders.
            # Only BcosExplainer supports map_size.
            # However, even BcosExplainer's default utilities (gradient_to_image) don't seem to support 'mode'.
            kwargs.pop('mode', None)
            
            is_bcos = self.explainer.__class__.__name__ == "BcosExplainer"
            is_legrad_or_chefer = self.explainer.__class__.__name__ in ["LeGradExplainer", "CheferExplainer"]

            if is_legrad_or_chefer:
                # Disable normalization for LeGrad/Chefer during Grid PG (or whenever called via this wrapper in grid/strided mode)
                # We want raw attribution mass to preserve relative importance across grid cells
                kwargs["normalize"] = False

            # logger.info(f"Using explainer {self.explainer.__class__.__name__}")

            if not is_bcos:
                kwargs.pop('map_size', None)

            # Ensure mapping of explain_class -> target
            return self.explainer.explain(self.model, inp, target=explain_class, **kwargs)

        self.processor_config = GridProcessorConfig(
            model=self.model,
            explain_fn=explain_wrapper,
            grid_pg_scale=2,  # Default scale, usually matching dataset
            confidence_threshold=self.confidence_threshold,
            prediction_fn=prediction_fn,
            map_size=map_size,
        )
        # Note: GridPGProcessor now takes config only
        self.grid_pg_processor = GridPGProcessor(self.processor_config)

    @jaxtyped(typechecker=typechecker)
    def evaluate_batch(
        self,
        images: Float[torch.Tensor, "batch channels height width"],
        labels: Int[torch.Tensor, "batch label_axes"],
    ) -> float:
        """
        Run Grid PG on a single ``(images, labels)`` batch; return mean score in ``[0, 1]``, or ``0.0`` if none confident.

        Intended for unit tests and quick checks (full eval uses :meth:`evaluate`).
        """
        self.model.eval()
        images = images.to(self.device)
        labels = labels.to(self.device)
        with torch.inference_mode(False):
            with torch.enable_grad():
                score, _n_conf = self.grid_pg_processor.process_validation_step(images, labels)
        if score is None:
            return 0.0
        return float(score)

    def evaluate(
        self,
        dataloader,
        save_path: Optional[str | Path] = None,
        max_batches: Optional[int] = None,
        logger: Optional[Any] = None,
        metric_prefix: str = "grid_pg",
    ) -> Dict[str, float]:
        self.model.eval()
        
        if max_batches is None:
            max_batches = self.max_batches

        total_score_sum = 0.0
        num_batches = 0
        total_confident = 0
        total_samples = 0
        
        if max_batches is not None:
            if hasattr(dataloader, "__len__"):
                total_iters = min(max_batches, len(dataloader))
            else:
                total_iters = max_batches
        else:
            total_iters = len(dataloader) if hasattr(dataloader, "__len__") else None
            
        for i, batch in tqdm.tqdm(enumerate(dataloader), desc="Evaluating Grid PG", unit="batch", total=total_iters):
            if max_batches is not None and i >= max_batches:
                break
            
            # Handle batch format
            if isinstance(batch, (list, tuple)):
                images, labels = batch
            elif isinstance(batch, dict):
                images = batch.get('image', batch.get('img'))
                labels = batch.get('label', batch.get('target'))
                if images is None or labels is None:
                    raise ValueError("Batch must contain 'image' or 'img' and 'label' or 'target'")
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            total_samples += images.size(0)
                
            # Use GridPGProcessor
            with torch.inference_mode(False):
                with torch.enable_grad():
                    score, n_conf = self.grid_pg_processor.process_validation_step(images, labels)
            
            if score is not None:
                # score is the mean over the confident samples in the batch.
                # Accumulate the sum so we can calculate a proper dataset-level mean
                total_score_sum += score * n_conf
                num_batches += 1
                total_confident += n_conf

            should_log_progress = (
                logger is not None
                and (
                    ((i + 1) % self.log_every_n_batches == 0)
                    or (total_iters is not None and (i + 1) == total_iters)
                )
            )
            if should_log_progress:
                running_score = float(total_score_sum / total_confident) if total_confident > 0 else 0.0
                progress_metrics = {
                    f"{metric_prefix}/running_score": running_score,
                    f"{metric_prefix}/processed_batches": float(i + 1),
                    f"{metric_prefix}/scored_batches": float(num_batches),
                    f"{metric_prefix}/confident_samples_running": float(total_confident),
                    # Keep a lightweight heartbeat in W&B even when no batches pass confidence.
                    f"{metric_prefix}/heartbeat": float(i + 1),
                }
                try:
                    logger.log_metrics(progress_metrics, step=i + 1)
                except TypeError:
                    logger.log_metrics(progress_metrics)

        final_score = float(total_score_sum / total_confident) if total_confident > 0 else 0.0
        relative_confident = float(total_confident / total_samples) if total_samples > 0 else 0.0
        
        print(f"Mean Grid PG Score: {final_score:.4f} (from {num_batches} batches, {total_confident} confident samples)")
        print(f"Relative Confident Samples: {relative_confident:.4f} ({total_confident}/{total_samples})")

        results = {
            "score": final_score, 
            "confident_samples": float(total_confident),
            "relative_confident_samples": float(relative_confident)
        }

        if logger is not None:
            final_metrics = {
                f"{metric_prefix}/score": float(final_score),
                f"{metric_prefix}/confident_samples": float(total_confident),
                f"{metric_prefix}/relative_confident_samples": float(relative_confident),
                f"{metric_prefix}/num_batches": float(num_batches),
                f"{metric_prefix}/processed_batches": float(total_iters if total_iters is not None else num_batches),
            }
            logger.log_metrics(final_metrics)

        if save_path:
            self.save_results(results, save_path)
            
        return results
    
    def save_results(self, results: Dict[str, float], save_path: str | Path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
