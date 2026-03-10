"""
Wrapper for Llama 2 models
"""

from typing import Optional, Literal
from transformers import AutoTokenizer
from .base import BaseLlamaWrapper
from .src.tokenize import Llama2Tokenizer

class Llama2Wrapper(BaseLlamaWrapper):
    """
    Wrapper for Llama 2 models
    """
    
    def __init__(
        self,
        hf_token: Optional[str],
        size: Literal["7b"] = "7b",
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
        gpu_id: int = 0
    ):
        """
        Initialize Llama 2 wrapper
        
        Args:
            hf_token: HuggingFace token for accessing models
            size: Model size (e.g., "7b", "13b", "70b")
            use_chat: Whether to use chat formatting
            override_model_weights_path: Optional path to override model weights
        """
        model_path = f"meta-llama/Llama-2-{size}-chat-hf" if use_chat else f"meta-llama/Llama-2-{size}-hf"
        
        # Initialize tokenizer and its wrapper
        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        tokenizer_wrapper = Llama2Tokenizer(tokenizer, use_chat=use_chat)
        tokenizer_wrapper.set_end_str(gpu_id=gpu_id)

        super().__init__(
            hf_token=hf_token,
            model_path=model_path,
            tokenizer=tokenizer_wrapper,
            override_model_weights_path=override_model_weights_path,
            gpu_id=gpu_id
        ) 