from typing import Optional

import torch


def aloe_attn_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """AttnLRP/CP-LRP attention rule for native ALOE's fused-QKV self-attention."""
    batch_size, seq_length, embed_dim = hidden_states.shape

    q, k, v = self.qkv_proj(hidden_states).split(self.embed_dim, dim=-1)
    q = q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2).detach()
    k = k.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2).detach()
    v = v.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is not None:
        from src.models.hf.aloe.modules.rope import apply_rotary_pos_emb

        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

    attn_output, attn_weights = self._attention_fn(
        self,
        q,
        k,
        v,
        attention_mask,
        is_causal=self.is_causal,
        scaling=self.scale,
        dropout=0.0 if not self.training else self.dropout,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, seq_length, embed_dim).contiguous()
    attn_output = self.out_proj(attn_output)
    return attn_output, (attn_weights if output_attentions else None)


def aloe_pooler_attn_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """AttnLRP/CP-LRP attention rule for native ALOE's learned-probe pooler attention."""
    del value
    batch_size, query_length, embed_dim = query.shape
    key_value_length = key.shape[1]
    num_heads, head_dim = self.num_heads, self.head_dim

    q = self.q_proj(query).view(batch_size, query_length, num_heads, head_dim).transpose(1, 2).detach()
    k, v = self.kv_proj(key).split(self.embedding_dim, dim=-1)
    k = k.view(batch_size, key_value_length, num_heads, head_dim).transpose(1, 2).detach()
    v = v.view(batch_size, key_value_length, num_heads, head_dim).transpose(1, 2)

    if output_attentions:
        from src.models.hf.aloe.modules.attention_utils import eager_attention_forward

        attention_fn = eager_attention_forward
    else:
        attention_fn = self._attention_fn

    attn_output, attn_weights = attention_fn(
        self,
        q,
        k,
        v,
        attn_mask,
        is_causal=self.is_causal,
        scaling=self.scale,
        dropout=0.0 if not self.training else self.dropout,
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, query_length, embed_dim).contiguous()
    return self.out_proj(attn_output), (attn_weights if output_attentions else None)
