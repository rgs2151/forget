"""
Wrapper for Llama 3 / 3.1 models
"""

from typing import Optional, Literal
import torch as t
from transformers import AutoTokenizer
from .base import BaseLlamaWrapper
from ..abstract import AbstractTokenizer

class Llama3Wrapper(BaseLlamaWrapper):

    def __init__(
        self,
        hf_token: Optional[str],
        size: Literal["8b", "70b"] = "8b",
        use_chat: bool = True,
        override_model_weights_path: Optional[str] = None,
        gpu_id: int = 0,
    ):
        model_path = (
            f"meta-llama/Llama-3.1-{size.upper()}-Instruct"
            if use_chat
            else f"meta-llama/Llama-3.1-{size.upper()}"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, token=hf_token)
        tokenizer_wrapper = Llama3Tokenizer(tokenizer, use_chat=use_chat)
        tokenizer_wrapper.set_end_str(gpu_id=gpu_id)

        super().__init__(
            hf_token=hf_token,
            model_path=model_path,
            tokenizer=tokenizer_wrapper,
            override_model_weights_path=override_model_weights_path,
            gpu_id=gpu_id,
        )





class Llama3Tokenizer(AbstractTokenizer):
    """
    Llama 3 / 3.1 Tokenizer Wrapper

    chat format:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>

        system text<|eot_id|><|start_header_id|>user<|end_header_id|>

        user text<|eot_id|><|start_header_id|>assistant<|end_header_id|>

        assistant text<|eot_id|>
    """

    # Chat formatting tags
    B_SYS = "<|start_header_id|>system<|end_header_id|>\n\n"
    E_SYS = "<|eot_id|>"
    B_USER = "<|start_header_id|>user<|end_header_id|>\n\n"
    E_USER = "<|eot_id|>"
    B_ASSISTANT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    E_ASSISTANT = "<|eot_id|>"

    # Base formatting tags
    BASE_INPUT = "Input:"
    BASE_RESPONSE = "\nResponse:"

    # Position markers for activation steering
    # Steer from the assistant header onwards
    ADD_FROM_POS_CHAT = "<|start_header_id|>assistant<|end_header_id|>"
    ADD_FROM_POS_BASE = BASE_RESPONSE

    # End of prompt sequence token tensor
    END_STR: Optional[t.Tensor] = None

    # Special tokens
    BOS = "<|begin_of_text|>"
    EOS = "<|eot_id|>"
    
    def __init__(self, tokenizer: AutoTokenizer, use_chat: bool = True):
        self.tokenizer = tokenizer
        self.use_chat = use_chat

