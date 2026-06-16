from typing import List
from typing import Callable
import torch
import torch.nn as nn

from src.modules.defaults import DEFAULT_ACT_LAYER, DEFAULT_CONV_LAYER, DEFAULT_NORM_LAYER, DEFAULT_LINEAR_LAYER
# from src.models.bcos_vit import make_conv_stem
from einops.layers.torch import Rearrange



def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_conv_stem(
    in_channels: int,
    out_channels: List[int],
    conv2d_layer: Callable[..., nn.Module] = None,
    norm2d_layer: Callable[..., nn.Module] = None,
    act_layer: Callable[..., nn.Module] = None,
):
    """
    Following the conv stem design in Early Convolutions Help Transformers See Better (Xiao et al.)
    """
    model = []
    for outc in out_channels:
        conv = conv2d_layer(
            in_channels,
            outc,
            kernel_size=3,
            stride=(2 if outc > in_channels else 1),
            padding=1,
        )
        in_channels = outc
        norm = norm2d_layer(in_channels)
        act = act_layer()
        model += [conv, norm, act]
    return nn.Sequential(*model)



class ConvStem(nn.Module):
    """
    Convolutional stem that replaces the patch embedding.
    Follows the same pattern as SimpleViT: conv -> rearrange to spatial patches -> linear projection.
    """
    def __init__(
        self,
        conv_layers: list[int],
        patch_size: int = 16,
        target_dim: int = 768,
        conv2d_layer=None,
        norm2d_layer=None,
        act_layer=None,
        linear_layer=None,
        in_channels: int = 6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.target_dim = target_dim
        
        # Create conv stem
        self.stem = make_conv_stem(
            in_channels=in_channels,
            out_channels=conv_layers,
            conv2d_layer=conv2d_layer or DEFAULT_CONV_LAYER,
            norm2d_layer=norm2d_layer or DEFAULT_NORM_LAYER,
            act_layer=act_layer or DEFAULT_ACT_LAYER,
        )
        
        # Calculate actual downsampling from conv stem
        # Each layer with stride=2 halves the spatial dimensions
        downsampling_factor = 1
        prev_channels = 6
        for outc in conv_layers:
            stride = 2 if outc > prev_channels else 1
            downsampling_factor *= stride
            prev_channels = outc
        
        patch_size_after_downsampling = patch_size // downsampling_factor
        # Calculate input dim for linear layer (per patch)
        patch_height, patch_width = pair(patch_size_after_downsampling)
        patch_dim = conv_layers[-1] * patch_height * patch_width
        
        # Rearrange to spatial patches: (batch, channels, height, width) -> (batch, h, w, patch_dim)
        # This matches SimpleViT's approach: "b c (h p1) (w p2) -> b h w (p1 p2 c)"
        self.to_patches = Rearrange(
            "b c (h p1) (w p2) -> b h w (p1 p2 c)",
            p1=patch_height,
            p2=patch_width,
        )
        
        # Project each patch to target dimension
        self.patch_projection = (linear_layer or DEFAULT_LINEAR_LAYER)(patch_dim, target_dim)
        
        # Initialize the patch projection layer properly
        self._init_weights()
    
    
    def _init_weights(self):
        """Initialize weights similar to HuggingFace ViT"""
        # Initialize the patch projection layer
        if hasattr(self.patch_projection, 'weight'):
            # Standard linear layer
            nn.init.xavier_uniform_(self.patch_projection.weight)
        elif hasattr(self.patch_projection, 'linear'):
            # B-COS layer with wrapped linear
            if hasattr(self.patch_projection.linear, 'weight'):
                nn.init.xavier_uniform_(self.patch_projection.linear.weight)
            elif hasattr(self.patch_projection.linear, 'linear') and hasattr(self.patch_projection.linear.linear, 'weight'):
                nn.init.xavier_uniform_(self.patch_projection.linear.linear.weight)
    
    def forward(self, x):
        x = self.stem(x)  # Apply conv layers: (B, 6, 224, 224) -> (B, C, H', W')
        x = self.to_patches(x)  # Rearrange to spatial patches: (B, C, H', W') -> (B, H', W', patch_dim)
        x = self.patch_projection(x)  # Project each patch: (B, H', W', patch_dim) -> (B, H', W', 768)
        
        # Rearrange to match ViTPatchEmbeddings output format: (B, H', W', 768) -> (B, 768, H', W')
        # ViTPatchEmbeddings outputs (batch, hidden_size, height, width)
        x = x.permute(0, 3, 1, 2)  # (B, H', W', 768) -> (B, 768, H', W')
        
        return x
    
    @property
    def weight(self):
        '''
        Needs to be present to avoid errors with HF transformers ViTForImageClassification model when loading pretrained weights.
        '''
        # Handle different types of layers that might not have direct weight access
        if hasattr(self.patch_projection, 'weight'):
            return self.patch_projection.weight
        elif hasattr(self.patch_projection, 'linear') and hasattr(self.patch_projection.linear, 'weight'):
            # For B-COS layers that wrap a linear layer
            return self.patch_projection.linear.weight
        elif hasattr(self.patch_projection, 'linear') and hasattr(self.patch_projection.linear, 'linear'):
            # For doubly wrapped B-COS layers
            return self.patch_projection.linear.linear.weight
        else:
            # Fallback: create a dummy parameter if no weight is found
            import warnings
            warnings.warn("ConvStem: patch_projection layer has no accessible weight, creating dummy parameter")
            return nn.Parameter(torch.empty(self.target_dim, 1))  # Dummy parameter