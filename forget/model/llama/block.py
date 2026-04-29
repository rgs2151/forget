"""
Wrapper for the block to save activations and unembed them
"""

import torch as t
from .attention import AttnWrapper
from ..abstract import AbstractTokenizer
from typing import Optional


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """
    save_internal_decodings: bool = False
    steering_op = None

    activations: Optional[t.Tensor] = None
    from_position: Optional[t.Tensor] = None   # (batch,) tensor or None

    attn_out_unembedded: Optional[t.Tensor] = None
    intermediate_resid_unembedded: Optional[t.Tensor] = None
    mlp_out_unembedded: Optional[t.Tensor] = None
    block_out_unembedded: Optional[t.Tensor] = None


    def __init__(
        self,
        block,
        unembed_matrix,
        norm,
        tokenizer: AbstractTokenizer
    ) -> None:
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

    def forward(
        self,
        *args,
        **kwargs,
    ) -> tuple[t.Tensor, ...]:
        output = self.block(*args, **kwargs)
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        self.activations = hidden

        if self.steering_op is not None:
            delta = self._compute_steering_delta(
                hidden, kwargs.get("position_ids")
            )
            hidden = hidden + delta
            output = (hidden,) + output[1:] if is_tuple else hidden

        if not self.save_internal_decodings:
            return output

        self.save_block_internals(output, args[0])
        return output

    def _compute_steering_delta(self, hidden: t.Tensor, position_ids) -> t.Tensor:
        raw = self.steering_op(hidden)          # (batch, seq, hidden) or (batch, 1, hidden)

        seq_len = hidden.shape[-2]
        if seq_len == 1:                        # KV-cache generation step: always steer
            return raw.expand_as(hidden)

        # Prefill: mask out padding + pre-instruction tokens
        col = t.arange(seq_len, device=hidden.device).unsqueeze(0)              # (1, seq)
        content_start = (position_ids == 0).long().cumsum(-1).argmax(-1)        # (batch,)
        mask = col >= content_start.unsqueeze(-1)                               # (batch, seq)
        if self.from_position is not None:
            mask = mask & (col >= self.from_position.unsqueeze(-1))             # (batch, seq)
        return mask.unsqueeze(-1).to(hidden.dtype) * raw                        # (batch, seq, 1) * raw

    def save_block_internals(
        self,
        output: t.Tensor,
        attn_output: t.Tensor,
    ) -> None:
        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += attn_output
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        # Final residual unembedded
        self.block_out_unembedded = self.unembed_matrix(self.norm(output[0]))

    def add(self, activations: t.Tensor) -> None:
        """Wrap raw activations in AddSteer for backward compat."""
        from .steering import AddSteer
        self.steering_op = AddSteer(vec=activations, scale=1.0)

    def reset(self) -> None:
        self.steering_op = None
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
