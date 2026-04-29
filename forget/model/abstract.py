"""
Abstract wrapper class
"""

from abc import ABC, abstractmethod
import torch as t
from typing import List, Union, Dict
    
class AbstractTokenizer(ABC):
    """
    Abstract tokenizer class
    """
    
    # Class attributes that must be defined by subclasses
    tokenizer: any
    use_chat: bool  # Whether to use chat formatting
    
    # Chat formatting tags
    B_SYS: str  # System message start
    E_SYS: str  # System message end
    B_USER: str  # User message start
    E_USER: str  # User message end
    B_ASSISTANT: str  # Assistant message start
    E_ASSISTANT: str  # Assistant message end
    
    # Base formatting tags
    BASE_INPUT: str  # Base input start
    BASE_RESPONSE: str  # Base response start
    
    # Position markers for activation steering
    ADD_FROM_POS_CHAT: str  # Position to add activations in chat mode
    ADD_FROM_POS_BASE: str  # Position to add activations in base mode
    
    def tokenize(self, prompt: str) -> List[int]:
        """Tokenize a formatted prompt."""
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def batch_decode(self, tokens: Union[List[int], t.Tensor]) -> str:
        """Decode tokens back to text"""
        if isinstance(tokens, t.Tensor):
            if tokens.dim() > 1:
                tokens = tokens[0]  # Take first sequence from batch
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def encode(self, text: str) -> List[int]:
        """Encode text to tokens"""
        return self.tokenizer.encode(text)

    def decode(self, token_id: Union[int, t.Tensor]) -> str:
        """Decode a single token"""
        if isinstance(token_id, t.Tensor):
            token_id = int(token_id.item())
        return self.tokenizer.decode([token_id])

    def get_position_markers(self) -> Dict[str, str]:
        """Get position markers for activation steering"""
        return {
            "ADD_FROM_POS_CHAT": self.ADD_FROM_POS_CHAT,
            "ADD_FROM_POS_BASE": self.ADD_FROM_POS_BASE
        }

    def tokenize_batch(self, prompts: List[str]) -> Dict[str, t.Tensor]:
        """Left-pad a list of formatted prompts into a batch.

        Returns:
            {"input_ids": (batch, max_seq_len), "attention_mask": (batch, max_seq_len)}
        """
        token_lists = [self.tokenize(prompt) for prompt in prompts]
        max_len = max(len(tl) for tl in token_lists)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids = []
        attention_mask = []
        for tl in token_lists:
            pad_len = max_len - len(tl)
            input_ids.append([pad_id] * pad_len + tl)
            attention_mask.append([0] * pad_len + [1] * len(tl))

        return {
            "input_ids": t.tensor(input_ids),
            "attention_mask": t.tensor(attention_mask),
        }

    def set_end_str(self, gpu_id: int = 0) -> None:
        """Set the end string for tokenization"""
        device = f"cuda:{gpu_id}" if t.cuda.is_available() else "cpu"
        pos = self.ADD_FROM_POS_CHAT if self.use_chat else self.ADD_FROM_POS_BASE
        self.END_STR = t.tensor(self.encode(pos)[1:]).to(device)