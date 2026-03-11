"""
Abstract wrapper class
"""

from abc import ABC, abstractmethod
import torch as t
from typing import Optional, List, Union, Dict
from transformers.tokenization_utils import PreTrainedTokenizer
from ..chat import Chat

class AbstractWrapper(ABC):
    """
    Abstract wrapper class
    """
    
    @abstractmethod
    def generate(self, tokens: t.Tensor, max_new_tokens: int = 100) -> t.Tensor:
        """
        Generate tokens
        """
        pass
    
class AbstractTokenizer(ABC):
    """
    Abstract tokenizer class
    """
    
    # Class attributes that must be defined by subclasses
    tokenizer: PreTrainedTokenizer
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
    
    @abstractmethod
    def interpret_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Interpret messages in chat mode format
        """
        pass
    
    @abstractmethod
    def interpret_base(self, messages: List[Dict[str, str]]) -> str:
        """
        Interpret messages in base mode format
        """
        pass
    
    def tokenize(self, chat: Chat) -> List[int]:
        """
        Unified tokenize method that converts Chat to token IDs
        """
        messages = chat.get_conversation()
        if self.use_chat:
            input_content = self.interpret_chat(messages)
        else:
            input_content = self.interpret_base(messages)
        return self.tokenizer.encode(input_content, add_special_tokens=False)

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

    def set_end_str(self, gpu_id: int = 0) -> None:
        """Set the end string for tokenization"""
        device = f"cuda:{gpu_id}" if t.cuda.is_available() else "cpu"
        pos = self.ADD_FROM_POS_CHAT if self.use_chat else self.ADD_FROM_POS_BASE
        self.END_STR = t.tensor(self.encode(pos)[1:]).to(device)