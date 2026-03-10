"""
Llama model wrappers
"""

from .block import BlockOutputWrapper
from .attention import AttnWrapper

__all__ = ["BlockOutputWrapper", "AttnWrapper"]