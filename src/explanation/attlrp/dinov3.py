from typing import Callable, Optional, Unpack
import torch

from transformers.utils import TransformersKwargs
from transformers.models.dinov3_vit.modeling_dinov3_vit import apply_rotary_pos_emb, eager_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def dinov3_attn_forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """ 
    For the CP-LRP variant, no gradient is allowed to flow through the softmax function.
    We patch the torch.nn.MultiheadAttention.forward such that the gradient flow
    at the query and key tensors is stopped, which are directly connected to the softmax function.
    """
    batch_size, patches, _ = hidden_states.size()

    # At this point query and key have to be detached for CP-LRP
    query_states = self.q_proj(hidden_states).detach()
    key_states = self.k_proj(hidden_states).detach()
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, patches, self.num_heads, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings # type: ignore
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, patches, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
