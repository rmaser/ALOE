from typing import TYPE_CHECKING, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch import Tensor

if TYPE_CHECKING:
    import matplotlib
    import matplotlib.pyplot as plt

TensorLike = Union[Tensor, np.ndarray]

__all__ = [
    "gradient_to_image",
    "joint_alpha_quantile_value",
    "plot_contribution_map",
    "contribution_map_to_image",
]


def joint_alpha_quantile_value(
    image: Tensor,
    linear_mapping: Tensor,
    *,
    alpha_percentile: float = 99.5,
    smooth: int = 15,
    smooth_sigma: Optional[float] = None,
) -> Tensor:
    """
    One alpha quantile over every batch item and spatial location (blurred alpha, same
    recipe as ``gradient_to_image``). Use as ``alpha_quantile_value=`` when rendering each
    token so saliency strength is comparable across output steps.
    """
    assert image.ndim == 4 and linear_mapping.ndim == 4
    contribs = (image * linear_mapping).sum(1, keepdim=True)
    alpha = linear_mapping.norm(p=2, dim=1, keepdim=True)
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    if smooth:
        sigma_vals = [float(smooth_sigma), float(smooth_sigma)] if smooth_sigma else None
        alpha = TF.gaussian_blur(alpha, kernel_size=[smooth, smooth], sigma=sigma_vals)
    flat = alpha.flatten()
    q = float(alpha_percentile) / 100.0
    return torch.quantile(flat, q)


def gradient_to_image(
    image: Tensor,
    linear_mapping: Tensor,
    smooth: int = 15,
    alpha_percentile: float = 99.5,
    to_numpy: bool = True,
    smooth_sigma: Optional[float] = None,
    use_original_colors: bool = False,
    alpha_quantile_value: Optional[float] = None,
) -> TensorLike:
    assert len(linear_mapping.shape) == 4 and len(image.shape) == 4

    img_rgb = image[:, :3].clone()
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    img_rgb = img_rgb * std + mean

    contribs = (image * linear_mapping).sum(1, keepdim=True)

    if use_original_colors:
        rgb_grad = img_rgb.clamp(0, 1)
    else:
        denominator = linear_mapping.abs().max(1, keepdim=True).values + 1e-12
        rgb_grad = linear_mapping * torch.reciprocal(denominator)
        rgb_grad = rgb_grad.clamp(min=0)
        if rgb_grad.shape[1] >= 6:
            denominator2 = rgb_grad[:, :3] + rgb_grad[:, 3:6] + 1e-12
            rgb_grad = rgb_grad[:, :3] * torch.reciprocal(denominator2)
        elif rgb_grad.shape[1] == 1:
            rgb_grad = rgb_grad.repeat(1, 3, 1, 1)

    alpha = linear_mapping.norm(p=2, dim=1, keepdim=True)
    alpha = torch.where(contribs < 0, 1e-12, alpha)

    if smooth:
        sigma_vals = [float(smooth_sigma), float(smooth_sigma)] if smooth_sigma else None
        alpha = TF.gaussian_blur(alpha, kernel_size=[smooth, smooth], sigma=sigma_vals)

    if alpha.numel() == 0:
        raise ValueError("Alpha tensor is empty.")

    if alpha_quantile_value is not None:
        scale = torch.as_tensor(alpha_quantile_value, device=alpha.device, dtype=alpha.dtype)
        alpha = (alpha / (scale + 1e-12)).clip(0, 1)
    else:
        B = alpha.shape[0]
        alpha_reshaped = alpha.view(B, -1)
        quantiles_per_image = torch.quantile(
            alpha_reshaped, q=float(alpha_percentile) / 100.0, dim=1, keepdim=True
        )
        alpha = alpha * torch.reciprocal(quantiles_per_image.view(B, 1, 1, 1) + 1e-12)
        alpha = alpha.clip(0, 1)

    final_out = torch.concatenate([rgb_grad, alpha], dim=1).permute(0, 2, 3, 1)
    final_out = final_out.clamp(0, 1)
    return final_out.cpu().numpy() if to_numpy else final_out


def plot_contribution_map(
    contribution_map: TensorLike,
    ax: Optional["plt.Axes"] = None,
    vrange: Optional[float] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    hide_ticks: bool = True,
    cmap: str = "bwr",
    percentile: float = 99.5,
) -> "Tuple[plt.Axes, matplotlib.image.AxesImage]":
    assert contribution_map.ndim == 2, "Contribution map is supposed to only have 2 spatial dimensions."
    if isinstance(contribution_map, torch.Tensor):
        contribution_map = contribution_map.detach().cpu().numpy()
    cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1)

    if vrange is None or vrange == "auto":
        vrange = np.max(np.abs(contribution_map.flatten()))
    im = ax.imshow(
        contribution_map,
        cmap=cmap,
        vmin=-vrange if vmin is None else vmin,
        vmax=vrange if vmax is None else vmax,
    )

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

    return ax, im


def contribution_map_to_image(
    contribution_map: TensorLike,
    percentile: float = 99.5,
    cmap: str = "bwr",
    *,
    global_cutoff: Optional[float] = None,
    global_vrange: Optional[float] = None,
) -> np.ndarray:
    """
    Renders a contribution map to an RGBA numpy array without needing a matplotlib figure.
    This mimics the normalization of ``plot_contribution_map`` exactly.
    """
    if isinstance(contribution_map, torch.Tensor):
        contribution_map = contribution_map.detach().cpu().numpy()

    contribution_map = np.squeeze(contribution_map)
    assert contribution_map.ndim == 2, "Contribution map is supposed to only have 2 spatial dimensions."

    if global_cutoff is not None:
        cutoff = global_cutoff
    else:
        cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)

    if global_vrange is not None:
        vrange = global_vrange if global_vrange > 0 else 1e-12
    else:
        vrange = np.max(np.abs(contribution_map))
        if vrange == 0:
            vrange = 1e-12

    norm_map = (contribution_map + vrange) / (2 * vrange)
    norm_map = np.clip(norm_map, 0, 1)

    import matplotlib.cm as cm

    colormap = cm.get_cmap(cmap)
    return colormap(norm_map)
