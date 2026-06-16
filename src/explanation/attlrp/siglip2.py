from typing import Callable, Optional
import torch

from transformers.models.dinov3_vit.modeling_dinov3_vit import eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS


def siglip2_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Input shape: Batch x Time x Channel"""

    batch_size, seq_length, embed_dim = hidden_states.shape

    # At this point query and key have to be detached for CP-LRP
    queries = self.q_proj(hidden_states).detach()
    keys = self.k_proj(hidden_states).detach()
    values = self.v_proj(hidden_states)

    queries = queries.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    keys = keys.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    values = values.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        queries,
        keys,
        values,
        attention_mask,
        is_causal=self.is_causal,
        scaling=self.scale,
        dropout=0.0 if not self.training else self.dropout,
    )

    attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights
