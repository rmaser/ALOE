from functools import partial
from pathlib import Path
from typing import Optional, Dict, TYPE_CHECKING, Any, Tuple
import torch
import torch.nn.functional as F
import tqdm
from beartype import beartype as typechecker
from jaxtyping import Float, Int, jaxtyped
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from loguru import logger

if TYPE_CHECKING:
    from src.explanation.abstract import BaseExplainer

def is_bcos(model: nn.Module) -> bool:
    return bool(getattr(model, "is_bcos_model", False) or hasattr(model, "explain"))


def _normalize_contribution_map(current_map: torch.Tensor, _batch_size: int) -> torch.Tensor:
    """[B,1,H,W] single-channel map for pooling / masking."""
    if current_map.dim() == 3:
        return current_map.unsqueeze(1)
    if current_map.dim() == 4:
        if current_map.size(1) == 1:
            return current_map
        return current_map.mean(dim=1, keepdim=True)
    raise ValueError(f"contribution_map must be 3D or 4D, got shape {tuple(current_map.shape)}")


def _model_output_to_logits(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    if isinstance(outputs, tuple):
        return outputs[0]
    if torch.is_tensor(outputs):
        return outputs
    raise TypeError(f"Unsupported model output type for pixel deletion: {type(outputs).__name__}")


def _select_config_value(cfg: Any, path: str) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.select(cfg, path, default=None)
    except Exception:
        pass

    current = cfg
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        elif hasattr(current, part):
            current = getattr(current, part)
        else:
            return None
    return current


def _loss_target_name(loss_config: Any) -> str:
    if loss_config is None:
        return ""
    if isinstance(loss_config, str):
        return loss_config
    if hasattr(loss_config, "get"):
        target = loss_config.get("_target_", None)
        if target is not None:
            return str(target)
    target = getattr(loss_config, "_target_", None)
    if target is not None:
        return str(target)
    func = getattr(loss_config, "func", None)
    if func is not None:
        return getattr(func, "__name__", str(func))
    return getattr(loss_config, "__name__", type(loss_config).__name__)


def _configured_loss_name(cfg: Any) -> str:
    for path in (
        "default_loss_fn",
        "default_loss_fn._target_",
        "evaluators.linear_probe.loss_fn",
        "evaluators.linear_probe.loss_fn._target_",
        "evaluator.loss_fn",
        "evaluator.loss_fn._target_",
        "loss_fn",
        "loss_fn._target_",
    ):
        loss_name = _loss_target_name(_select_config_value(cfg, path))
        if loss_name:
            return loss_name
    return ""


def _uses_bce_style_head(model: nn.Module) -> bool:
    return bool(getattr(model, "is_bcos_model", False) or hasattr(model, "logit_layer"))


def _pad_hw_to_multiple(x_b1hw: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, int, int, int, int]:
    """Pad bottom/right so H,W are divisible by block_size. Returns padded tensor and nh,nw."""
    _, _, h, w = x_b1hw.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h or pad_w:
        x_b1hw = F.pad(x_b1hw, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    _, _, hp, wp = x_b1hw.shape
    nh, nw = hp // block_size, wp // block_size
    return x_b1hw, nh, nw, h, w


def _block_scores_for_mode(x_b1hw: torch.Tensor, mode: str, block_size: int) -> torch.Tensor:
    """Per-block scalar scores [B, nh*nw] for ranking (x already padded)."""
    if mode == "abs_topk":
        pooled = F.avg_pool2d(x_b1hw.abs(), block_size, stride=block_size)
    elif mode == "topk":
        pooled = F.avg_pool2d(x_b1hw, block_size, stride=block_size)
    elif mode == "positive_only":
        block_max = F.max_pool2d(x_b1hw, kernel_size=block_size, stride=block_size)
        pooled = F.avg_pool2d(x_b1hw.clamp(min=0), block_size, stride=block_size)
        pooled = pooled.masked_fill(block_max <= 0, float("-inf"))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return pooled.view(x_b1hw.size(0), -1)


def _block_mask_from_topk(
    scores_flat: torch.Tensor,
    nh: int,
    nw: int,
    block_size: int,
    height: int,
    width: int,
    deletion_percentage: float,
    largest: bool,
    device: torch.device,
) -> torch.Tensor:
    """Binary mask [B,1,H,W], 1=keep, 0=delete. Fraction is over blocks, not pixels."""
    batch_size = scores_flat.size(0)
    num_blocks = nh * nw
    k = int(round(deletion_percentage * num_blocks))
    k = max(0, min(num_blocks, k))

    block_keep = torch.ones(batch_size, num_blocks, device=device)
    if k > 0:
        for b in range(batch_size):
            row = scores_flat[b]
            valid = torch.isfinite(row)
            k_b = min(k, int(valid.sum().item()))
            if k_b > 0:
                sub = row.masked_fill(~valid, float("-inf") if largest else float("inf"))
                _, idx = torch.topk(sub, k_b, largest=largest)
                block_keep[b, idx] = 0.0

    mk = block_keep.view(batch_size, 1, nh, nw)
    mk = mk.repeat_interleave(block_size, dim=-2).repeat_interleave(block_size, dim=-1)
    mk = mk[:, :, :height, :width]
    return mk


class PixelDeletionEvaluator:
    def __init__(
        self, 
        model: nn.Module, 
        device: str = 'cuda',
        explainer: Optional["BaseExplainer"] = None,
        deletion_percentages: list[float] = [0.1],
        save_path: Optional[str | Path] = None,
        max_batches: Optional[int] = None,
        mode = "abs_topk",
        largest = True,
        correct_only: bool = False,
        confidence_threshold: Optional[float] = None,
        smoothing_sigma: float = 0.0,
        deletion_unit: str = "pixel",
        block_size: int = 16,
    ):
        if deletion_unit not in ("pixel", "block"):
            raise ValueError(f"deletion_unit must be 'pixel' or 'block', got {deletion_unit!r}")
        if block_size < 1:
            raise ValueError(f"block_size must be >= 1, got {block_size}")

        self.lightning_module = model.to(device)
        self.device = device
        self.explainer = explainer
        self.deletion_percentages = [float(dp) for dp in deletion_percentages]
        self.save_path = save_path
        self.max_batches = max_batches
        self.mode = mode
        self.largest: bool = largest
        self.correct_only: bool = correct_only
        self.confidence_threshold: Optional[float] = confidence_threshold
        self.smoothing_sigma = smoothing_sigma
        self.deletion_unit = deletion_unit
        self.block_size = block_size

        # Determine the underlying model to evaluate (student model)
        if hasattr(self.lightning_module, "student_model"):
            self.model = self.lightning_module.student_model
        else:
            self.model = self.lightning_module
        
        loss_name = _configured_loss_name(getattr(self.lightning_module, "cfg", None))
        loss_name_lower = loss_name.lower()

        if "bce" in loss_name_lower or "binary_cross_entropy" in loss_name_lower:
            self.activation_fn = torch.sigmoid
        elif "crossentropy" in loss_name_lower or "cross_entropy" in loss_name_lower:
            self.activation_fn = partial(torch.softmax, dim=-1)
        elif _uses_bce_style_head(self.lightning_module) or _uses_bce_style_head(self.model):
            logger.warning(
                "Could not find a pixel-deletion loss config; using sigmoid because the model exposes a BCE/B-cos-style head."
            )
            self.activation_fn = torch.sigmoid
        else:
            logger.warning(f"Unknown or missing loss function config ({loss_name}), defaulting to softmax")
            self.activation_fn = partial(torch.softmax, dim=-1)
            
        logger.info(f"Using activation function determined from Hydra config '{loss_name}': {self.activation_fn}")
        if self.deletion_unit == "block":
            logger.info(
                f"Block deletion: {self.block_size}x{self.block_size} patches; "
                "deletion_percentages are fractions of non-overlapping blocks (not pixels)."
            )

    @jaxtyped(typechecker=typechecker)
    def evaluate_batch(
        self,
        images: Float[torch.Tensor, "batch channels height width"],
        labels: Int[torch.Tensor, "batch *label_axes"],
    ) -> Dict[float, torch.Tensor]:
        self.lightning_module.eval()
        images = images.to(self.device)
        labels = labels.to(self.device).long()
        
        # Filter to correct samples and/or by confidence threshold if requested
        if self.correct_only or self.confidence_threshold is not None:
            with torch.no_grad():
                outputs = self.lightning_module(images)
                logits = _model_output_to_logits(outputs)
                
                probabilities = self.activation_fn(logits)
                predictions = logits.argmax(dim=-1)
                max_confidence = probabilities.max(dim=-1).values
                
                # Build filter mask
                filter_mask = torch.ones(len(images), dtype=torch.bool, device=self.device)
                
                if self.correct_only:
                    filter_mask = filter_mask & (predictions == labels)
                
                if self.confidence_threshold is not None:
                    filter_mask = filter_mask & (max_confidence >= self.confidence_threshold)
                
                if not filter_mask.any():
                    # No samples pass the filter in this batch
                    return {dp: torch.empty(0) for dp in self.deletion_percentages}
                
                images = images[filter_mask]
                labels = labels[filter_mask]
                logger.debug(f"Filtered to {filter_mask.sum().item()}/{len(filter_mask)} samples (correct_only={self.correct_only}, confidence_threshold={self.confidence_threshold})")
        
        batch_size, channels, height, width = images.shape
        total_pixels = height * width

        # Get attribution using the explainer or model's explain method
        if self.explainer is not None:
            explain_kwargs: dict[str, Any] = {"return_explanation": False}
            explanation = self.explainer.explain(self.model, images, target=labels, **explain_kwargs)
        else:
            # Fallback to model's explain method if no explainer provided (legacy support)
            if hasattr(self.model, "explain"):
                explanation = self.model.explain(images, idx=labels)
            elif hasattr(self.lightning_module, "explain"):
                explanation = self.lightning_module.explain(images, idx=labels)
            else:
                raise ValueError("No explainer provided and model does not support 'explain'.")

        
        # Start with raw map
        current_map = explanation['contribution_map']

        if self.smoothing_sigma > 0.0:
            # Apply Gaussian Blur
            k_size = int(4 * self.smoothing_sigma) + 1
            if k_size % 2 == 0: k_size += 1
            
            current_map = gaussian_blur(
                current_map, 
                kernel_size=[k_size, k_size], 
                sigma=[self.smoothing_sigma, self.smoothing_sigma]
            )


        map_b1hw = _normalize_contribution_map(current_map, batch_size)
        if map_b1hw.shape[-2:] != (height, width):
            map_b1hw = F.interpolate(
                map_b1hw, size=(height, width), mode="bilinear", align_corners=False
            )

        target_probability = {}

        for deletion_percentage in self.deletion_percentages:
            if self.deletion_unit == "block":
                x_pad, nh, nw, h_img, w_img = _pad_hw_to_multiple(map_b1hw, self.block_size)
                scores_flat = _block_scores_for_mode(x_pad, self.mode, self.block_size)
                mask = _block_mask_from_topk(
                    scores_flat,
                    nh,
                    nw,
                    self.block_size,
                    h_img,
                    w_img,
                    deletion_percentage,
                    self.largest,
                    self.device,
                )
            else:
                mask = self._pixel_deletion_mask(
                    map_b1hw, batch_size, height, width, total_pixels, deletion_percentage
                )

            mask = mask.to(device=images.device, dtype=images.dtype)

            with torch.no_grad():
                deleted_images = images * mask

                deleted_outputs = self.lightning_module(deleted_images)
                logits = _model_output_to_logits(deleted_outputs)

                probabilities = self.activation_fn(logits)

                gathered = probabilities.gather(1, labels.unsqueeze(1)).squeeze(1).detach().cpu()
                target_probability[deletion_percentage] = gathered

        return target_probability

    def _pixel_deletion_mask(
        self,
        map_b1hw: torch.Tensor,
        batch_size: int,
        height: int,
        width: int,
        total_pixels: int,
        deletion_percentage: float,
    ) -> torch.Tensor:
        """[B, 1, H, W] with 1=keep, 0=delete."""
        contribution_map = map_b1hw.view(batch_size, -1)
        k = int(round(deletion_percentage * total_pixels))

        mask = torch.ones(batch_size, 1, total_pixels, device=self.device)
        if k > 0 or self.mode == "positive_only":
            if self.mode == "positive_only":
                contrib_values = contribution_map.clone()
                contrib_values[contrib_values <= 0] = float("-inf")

                k_per_sample = torch.full((batch_size, 1), k, device=self.device, dtype=torch.long)

                _, sorted_indices = torch.sort(contrib_values, dim=1, descending=self.largest)

                ranks = torch.arange(total_pixels, device=self.device).unsqueeze(0)
                should_delete_sorted = ranks < k_per_sample

                delete_mask = torch.zeros_like(contribution_map, dtype=torch.bool)
                delete_mask.scatter_(1, sorted_indices, should_delete_sorted)

                mask.view(batch_size, -1).masked_fill_(delete_mask, 0.0)

            elif self.mode == "abs_topk":
                fixed_k = int(round(deletion_percentage * total_pixels))
                fixed_k = max(0, min(total_pixels, fixed_k))

                if fixed_k > 0:
                    ranking_values = contribution_map.abs()
                    _, indices = torch.topk(ranking_values, fixed_k, dim=1, largest=self.largest)
                    indices = indices.to(self.device)
                    mask.scatter_(2, indices.unsqueeze(1), 0.0)

            elif self.mode == "topk":
                fixed_k = int(round(deletion_percentage * total_pixels))
                fixed_k = max(0, min(total_pixels, fixed_k))

                if fixed_k > 0:
                    ranking_values = contribution_map
                    _, indices = torch.topk(ranking_values, fixed_k, dim=1, largest=self.largest)
                    indices = indices.to(self.device)
                    mask.scatter_(2, indices.unsqueeze(1), 0.0)

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

        return mask.view(batch_size, 1, height, width)

    def evaluate(self, dataloader, save_path: Optional[str | Path]=None, max_batches: Optional[int] = None):
        if save_path is None:
            save_path = self.save_path
        
        if max_batches is None:
            max_batches = self.max_batches

        self.lightning_module.eval()

        outputs: Dict[Any, Any] = {dp: torch.empty(0) for dp in self.deletion_percentages}
        
        if max_batches is not None:
            if hasattr(dataloader, "__len__"):
                total_batches = min(max_batches, len(dataloader))
            else:
                total_batches = max_batches
        else:
            total_batches = len(dataloader) if hasattr(dataloader, "__len__") else None
            
        for i, batch in tqdm.tqdm(enumerate(dataloader), desc="Evaluating Pixel Deletion", unit="batch", total=total_batches):
            if max_batches is not None and i >= max_batches:
                break
            
            # Handle batch format
            if isinstance(batch, dict):
                images = batch["image"]
                labels = batch["label"]
            elif isinstance(batch, (list, tuple)):
                images, labels = batch
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            batch_outputs = self.evaluate_batch(images, labels)

            # Append batch outputs to overall outputs
            for key, out in batch_outputs.items():
                outputs[key] = torch.cat((outputs[key], out), dim=0)
            
        curve = []
        sorted_dps = sorted([k for k in outputs.keys() if isinstance(k, (int, float))])
        
        unit = "blocks" if self.deletion_unit == "block" else "pixels"
        for dp in sorted_dps:
            val = outputs[dp]
            if val.numel() == 0:
                mean_val = 0.0
                print(f"Mean probability after deleting {dp*100:.0f}% {unit}: no evaluated samples")
            else:
                mean_val = val.mean().item()
                print(f"Mean probability after deleting {dp*100:.0f}% {unit}: {mean_val:.4f}")
            curve.append([dp, mean_val])

        outputs["pixel_deletion_curve"] = curve
        outputs["num_evaluated_samples"] = max((int(outputs[dp].numel()) for dp in sorted_dps), default=0)

        if save_path:
            self.save_results(outputs, save_path)
            
        return outputs
    
    def save_results(self, outputs: dict[float, torch.Tensor], save_path: str | Path):
        save_path = Path(save_path)
        # Append mode, largest, correct_only, and confidence_threshold to filename
        largest_str = "largest" if self.largest else "smallest"
        correct_str = "correct_only" if self.correct_only else "all_samples"
        conf_str = f"conf{self.confidence_threshold}" if self.confidence_threshold is not None else "no_conf"
        smooth_str = f"_smooth{self.smoothing_sigma}" if self.smoothing_sigma > 0 else ""
        unit_str = f"block{self.block_size}" if self.deletion_unit == "block" else "pixel"
        new_filename = (
            f"{save_path.stem}_{self.mode}_{largest_str}_{correct_str}_{conf_str}_{unit_str}{smooth_str}{save_path.suffix}"
        )
        save_path = save_path.with_name(new_filename)

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(outputs, save_path)
        print(f"Saved results to {save_path}")
