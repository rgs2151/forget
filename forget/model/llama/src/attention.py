"""
Wrapper for the attention mechanism to save activations
"""

import torch as t
from typing import Optional, Union

class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """
    activations = None

    def __init__(
        self,
        attn,
    ):
        super().__init__()
        self.attn = attn

    def forward(
        self, 
        *args,
        **kwargs,
    ):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output