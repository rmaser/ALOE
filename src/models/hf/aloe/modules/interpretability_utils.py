# Visualization helpers for ALOE ``explain`` (B-cos 6-channel RGB + inverse).
# Vendored from ``src/models/mixins/bcos_model.py``; kept here for Hub self-containment.
#
# No ``beartype`` / ``jaxtyping`` here: this file is loaded on the Hub via
# ``trust_remote_code`` and must stay transformers-only at runtime.

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as TF
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

TensorLike = Union[torch.Tensor, np.ndarray]


def _check_non_neg_int(name: str, n: int) -> None:
    if not isinstance(n, int) or n < 0:
        raise TypeError(f"{name} must be a non-negative int, got {n!r}")


def _check_closed_percent(name: str, x: float) -> None:
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__}")
    v = float(x)
    if not 0.0 <= v <= 100.0:
        raise ValueError(f"{name} must be in [0, 100], got {v}")


def _check_open_percent(name: str, x: float) -> None:
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__}")
    v = float(x)
    if not 0.0 < v <= 100.0:
        raise ValueError(f"{name} must be in (0, 100], got {v}")


def _check_pos_float(name: str, x: float) -> None:
    if not isinstance(x, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(x).__name__}")
    if float(x) <= 0.0:
        raise ValueError(f"{name} must be > 0, got {x}")


def _check_non_empty_str(name: str, s: str) -> None:
    if not isinstance(s, str) or len(s) == 0:
        raise TypeError(f"{name} must be a non-empty str, got {s!r}")


def gradient_to_image(
    image: torch.Tensor,
    linear_mapping: torch.Tensor,
    smooth: int = 15,
    alpha_percentile: float = 99.5,
    to_numpy: bool = True,
    smooth_sigma: Optional[float] = None,
) -> torch.Tensor | np.ndarray:
    """
    Build an RGBA explanation tensor (or NumPy array if ``to_numpy``).

    Expects ``image`` and ``linear_mapping`` of shape ``(N, C, H, W)`` with ``C >= 3``
    (typically ``C == 6`` for B-cos).
    """
    _check_non_neg_int("smooth", smooth)
    _check_closed_percent("alpha_percentile", alpha_percentile)
    if smooth_sigma is not None:
        _check_pos_float("smooth_sigma", smooth_sigma)

    if linear_mapping.ndim != 4 or image.ndim != 4:
        raise ValueError(
            f"Expected 4-D tensors; got image.ndim={image.ndim}, linear_mapping.ndim={linear_mapping.ndim}"
        )
    if image.shape != linear_mapping.shape:
        raise ValueError(
            f"image shape {tuple(image.shape)} != linear_mapping shape {tuple(linear_mapping.shape)}"
        )

    contribs = (image * linear_mapping).sum(1, keepdim=True)
    denominator = linear_mapping.abs().max(1, keepdim=True).values + 1e-12
    rgb_grad = linear_mapping * torch.reciprocal(denominator)
    rgb_grad = rgb_grad.clamp(min=0)

    if rgb_grad.shape[1] >= 6:
        denominator2 = rgb_grad[:, :3] + rgb_grad[:, 3:6] + 1e-12
        rgb_grad = rgb_grad[:, :3] * torch.reciprocal(denominator2)

    alpha = linear_mapping.norm(p=2, dim=1, keepdim=True)
    # Zero alpha where the per-pixel contribution is negative (no positive evidence).
    alpha = torch.where(contribs < 0, 0.0, alpha)
    if smooth:
        sigma_vals = None
        if smooth_sigma is not None:
            sigma_vals = [float(smooth_sigma), float(smooth_sigma)]
        alpha = TF.gaussian_blur(alpha, kernel_size=[smooth, smooth], sigma=sigma_vals)

    if alpha.numel() > 0:
        B = alpha.shape[0]
        alpha_reshaped = alpha.view(B, -1)

        # One scalar per batch row: shape (B, 1). ``alpha`` is (B, 1, H, W) — not RGB.
        quantiles_per_image = torch.quantile(
            alpha_reshaped,
            q=float(alpha_percentile) / 100.0,
            dim=1,
            keepdim=True,
        )
        # (B,1,1,1) so the per-image divisor broadcasts over H×W (same value every spatial location).
        quantiles_for_division = quantiles_per_image.view(B, 1, 1, 1)
        alpha = alpha * torch.reciprocal(quantiles_for_division + 1e-12)
        alpha = alpha.clip(0, 1)
    else:
        raise ValueError("Alpha tensor is empty; cannot compute quantiles.")

    rgb_grad = torch.concatenate([rgb_grad, alpha], dim=1)
    grad_image = rgb_grad.permute(0, 2, 3, 1)
    if to_numpy:
        return grad_image.cpu().numpy()
    return grad_image


def plot_contribution_map(
    contribution_map: TensorLike,
    ax: Optional[Axes] = None,
    vrange: Optional[Union[float, Literal["auto"]]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    hide_ticks: bool = True,
    cmap: str = "bwr",
    percentile: float = 99.5,
) -> tuple[Axes, AxesImage]:
    """Plot a 2-D contribution map; returns ``(axes, image_artist)``."""
    _check_non_empty_str("cmap", cmap)
    _check_open_percent("percentile", percentile)

    if isinstance(contribution_map, torch.Tensor):
        if contribution_map.ndim != 2:
            raise ValueError(f"Contribution map must be 2-D (H, W); got ndim={contribution_map.ndim}")
        contribution_map = contribution_map.detach().cpu().numpy()
    else:
        if contribution_map.ndim != 2:
            raise ValueError(f"Contribution map must be 2-D (H, W); got ndim={contribution_map.ndim}")

    cutoff = np.percentile(np.abs(contribution_map), percentile)
    contribution_map = np.clip(contribution_map, -cutoff, cutoff)

    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots(1)

    if vrange is None or vrange == "auto":
        vrange_f = float(np.max(np.abs(contribution_map.flatten())))
    else:
        vrange_f = float(vrange)

    im = ax.imshow(
        contribution_map,
        cmap=cmap,
        vmin=-vrange_f if vmin is None else vmin,
        vmax=vrange_f if vmax is None else vmax,
    )
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax, im
