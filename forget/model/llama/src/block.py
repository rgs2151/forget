"""
Wrapper for the block to save activations and unembed them
"""

import torch as t
from .attention import AttnWrapper
from ...abstract import AbstractTokenizer
from typing import Optional

class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """
    save_internal_decodings: bool = False
    add_activations: Optional[t.Tensor] = None
    steering_op = None
    
    activations: Optional[t.Tensor] = None
    from_position: Optional[int] = None
    
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
        """
        Forward pass for the block
        """
        output = self.block(*args, **kwargs)
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        self.activations = hidden

        if self.add_activations is not None or self.steering_op is not None:
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
        # raw delta from steering_op or plain add_activations
        if self.steering_op is not None:
            raw = self.steering_op(hidden)
        else:
            raw = self.add_activations.to(hidden.dtype)

        # position masking
        from_id = self.from_position
        if position_ids is not None:
            if from_id is None:
                from_id = position_ids.min().item() - 1
            mask = (position_ids >= from_id).unsqueeze(-1)
            while mask.dim() > hidden.dim():
                mask = mask.squeeze(0)
            return mask.to(hidden.dtype) * raw
        return raw.expand_as(hidden)
    
    def save_block_internals(
        self,
        output: t.Tensor,
        attn_output: t.Tensor,
    ) -> None:
        """
        Save the internals of the block
        """
        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += attn_output
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        # Final residual unembedded. (this is actually also the output of the block)
        self.block_out_unembedded = self.unembed_matrix(self.norm(output[0]))
    
    def add_vector_from_position(
        self,
        matrix: t.Tensor,
        vector: t.Tensor,
        position_ids: t.Tensor,
        from_pos: Optional[int] = None,
    ) -> t.Tensor:
        """
        Add a vector to the matrix from a given position
        """
        from_id = from_pos
        if from_id is None:
            from_id = position_ids.min().item() - 1

        mask = position_ids >= from_id
        mask = mask.unsqueeze(-1)
        # squeeze batch dims that newer transformers drops from hidden states
        while mask.dim() > matrix.dim():
            mask = mask.squeeze(0)

        matrix += mask.float() * vector
        return matrix
    
    def add(self, activations: t.Tensor) -> None:
        """
        Add activations to the block
        """
        self.add_activations = activations

    def reset(self) -> None:
        """
        Reset the block
        """
        self.add_activations = None
        self.steering_op = None
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
