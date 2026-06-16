import cv2
import numpy as np
from typing import Tuple, Optional

def process_lime_attribution(
    attribution: np.ndarray, 
    target_size: Optional[Tuple[int, int]] = None, 
    blur_sigma: float = 0.0
) -> np.ndarray:
    """
    Process LIME attribution image.
    
    Args:
        attribution: Input image (H, W) or (H, W, C).
        target_size: (H, W) to resize to.
        blur_sigma: Sigma for Gaussian blur.
        
    Returns:
        Processed image.
    """
    img = attribution.copy()
    
    img = attribution.copy()
    
    if target_size is not None:
        h, w = target_size
        # Downsample to original grid size (14x14 for 224x224 with patch 16)
        # We assume patch size 16 as per user description
        patch_size = 16
        small_h = h // patch_size
        small_w = w // patch_size
        
        # 1. Downsample with Nearest to recover/simulate the grid
        img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Upsample with Bilinear to smooth it
        img = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
        
    return img
