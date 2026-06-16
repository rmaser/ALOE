import torch
from torch import Tensor
from typing import Optional

def gradient_to_image(
    image: Tensor,
    linear_mapping: Tensor,
    smooth: int = 15,
    alpha_percentile: float = 99.5,
    to_numpy: bool = True,
    clamping_percentile_floor: float = 0.0,
    smooth_sigma: Optional[float] = None,
) -> Tensor:
    """
    Computing color image from dynamic linear mapping of B-cos models.
    
    Parameters
    ----------
    image: Tensor
        Original input image (encoded with 6 color channels)
        Shape: [B, C, H, W] with C=6 or C=3
    linear_mapping: Tensor
        Linear mapping W_{1->l} of the B-cos model
        Shape: [B, C, H, W] same as image
    smooth: int
        Kernel size for smoothing the alpha values
    alpha_percentile: float
        Cut-off percentile for the alpha value. In range [0, 100].
    
    Returns
    -------
    torch.Tensor or np.ndarray
        image explanation of the B-cos model.
        Shape: [B, H, W, C] (C=4 ie RGBA)
    """
    assert(len(linear_mapping.shape) == 4 and len(image.shape) == 4)

    # shape of img and linmap is [B, C, H, W], summing over first dimension gives the contribution map per location
    # contribs = (image * linear_mapping).sum(1, keepdim=True)  # [B, H, W]
    
    # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
    denominator = linear_mapping.abs().max(1, keepdim=True).values + 1e-12
    rgb_grad = linear_mapping * torch.reciprocal(denominator)

    rgb_grad = rgb_grad.clamp(min=0)

    # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
    
    # Handle both 6-channel (with inverse) and 3-channel inputs
    if rgb_grad.shape[1] >= 6:
        # Original 6-channel B-cos case: r, g, b, 1-r, 1-g, 1-b
        denominator2 = rgb_grad[:, :3] + rgb_grad[:, 3:6] + 1e-12
        rgb_grad = rgb_grad[:, :3] * torch.reciprocal(denominator2)  # [B, 3, H, W]
    elif rgb_grad.shape[1] == 1:
        # Handle 1-channel heatmap (e.g. from LeGrad)
        rgb_grad = rgb_grad.repeat(1, 3, 1, 1)

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = linear_mapping.norm(p=2, dim=1, keepdim=True)
    
    # Smooth alpha
    if smooth > 1:
        # Use simple average pooling for smoothing if no sigma provided
        pad = smooth // 2
        alpha = torch.nn.functional.avg_pool2d(alpha, kernel_size=smooth, stride=1, padding=pad)

    # Normalize alpha
    B = alpha.shape[0]
    alpha = alpha.view(B, -1)
    
    # Calculate percentile for each item in batch
    k = int(alpha.shape[1] * (alpha_percentile / 100.0))
    k = min(k, alpha.shape[1] - 1)
    
    alpha_max = torch.kthvalue(alpha, k, dim=1).values
    alpha_max = alpha_max.view(B, 1, 1, 1)
    alpha = alpha.view(B, 1, image.shape[2], image.shape[3])
    
    alpha = alpha / (alpha_max + 1e-12)
    alpha = alpha.clamp(0, 1)
    
    # Combine RGB and Alpha
    # rgb_grad is [B, 3, H, W], alpha is [B, 1, H, W]
    rgba = torch.cat([rgb_grad, alpha], dim=1)
    
    # Permute to [B, H, W, C]
    rgba = rgba.permute(0, 2, 3, 1)
    
    if to_numpy:
        return rgba.detach().cpu().numpy()
    
    return rgba

class ModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        out = self.model(x)
        if hasattr(out, 'logits'):
            return out.logits
        elif hasattr(out, 'pooler_output'):
            return out.pooler_output
        elif isinstance(out, tuple):
            return out[0]
        return out
