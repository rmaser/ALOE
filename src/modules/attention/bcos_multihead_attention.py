from typing import Type

import torch.nn as nn
from bcos.modules import DetachableModule

from src.modules.bcos import BcosUnnormedLinear

class BcosMultiHeadAttention(DetachableModule, nn.Module):
    def __init__( 
        self,
        embedding_dim: int,
        num_heads: int,
        out_b: float = 1.0,
        bcos_linear_class: type[nn.Module] = BcosUnnormedLinear,
        **kwargs
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.head_dim = embedding_dim // num_heads

        self.q_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.out_proj = bcos_linear_class(embedding_dim, embedding_dim, b=out_b)

    def forward(self, query, key, value, output_attentions=False, **kwargs):
        B, T_q, C = query.size()
        T_kv = key.size(1)  # Get actual key/value sequence length
        H = self.num_heads
        D = self.head_dim

        q = self.q_proj(query).view(B, T_q, H, D).transpose(1, 2)
        k = self.k_proj(key).view(B, T_kv, H, D).transpose(1, 2)
        v = self.v_proj(value).view(B, T_kv, H, D).transpose(1, 2)

        if self.detach:
            k = k.detach()
            q = q.detach()

        # Compute scaled dot-product attention
        if not output_attentions:
            attn_output = nn.functional.scaled_dot_product_attention(q, k, v)
            attn_weights = None
        else:
            import math
            # Manual attention for explainability
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
            attn_weights = scores.softmax(dim=-1)
            attn_output = (attn_weights @ v)
             
        # Concatenate heads and project output (sequence length follows the query)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, C)  # (B, T_q, C)
        output = self.out_proj(attn_output)  # (B, T, C)

        return output, attn_weights

    @classmethod
    def from_mha(cls, attn: nn.Module, out_b: float = 1.0, bcos_linear_class: Type[nn.Module] = BcosUnnormedLinear) -> 'BcosMultiHeadAttention':
        if not isinstance(attn, nn.MultiheadAttention):
            raise ValueError("attn must be an instance of nn.MultiheadAttention")
        
        assert attn.batch_first, "attn must have batch_first=True"

        in_proj_weights = attn.in_proj_weight
        q_proj_weight, k_proj_weight, v_proj_weight = in_proj_weights.chunk(3, dim=0)
        
        new_attn = cls(
            embedding_dim=attn.embed_dim,
            num_heads=attn.num_heads,
            out_b=out_b,
            bcos_linear_class=bcos_linear_class
        )

        new_attn.q_proj.weight.data.copy_(q_proj_weight)
        new_attn.k_proj.weight.data.copy_(k_proj_weight)
        new_attn.v_proj.weight.data.copy_(v_proj_weight)
        new_attn.out_proj.linear.weight.data.copy_(attn.out_proj.weight)  # type: ignore[attr-defined]

        return new_attn
    
