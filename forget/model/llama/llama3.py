"""
Wrapper for Llama 3 / 3.1 models
"""

from typing import Optional, Literal, List, Dict
import torch as t
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from .base import BaseLlamaWrapper
from ..abstract import AbstractTokenizer
from ..chat import Chat

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

    # extras
    last_input_size: Optional[int] = None

    def __init__(self, tokenizer: PreTrainedTokenizer, use_chat: bool = True):
        self.tokenizer = tokenizer
        self.use_chat = use_chat

    def interpret_chat(self, messages: List[Dict[str, str]]) -> str:
        input_content = self.BOS
        for m in messages:
            role, content = m["role"], m["content"]
            if role == "system":
                input_content += f"{self.B_SYS}{content.strip()}{self.E_SYS}"
            elif role == "user":
                input_content += f"{self.B_USER}{content.strip()}{self.E_USER}"
            elif role == "assistant" and content is not None:
                input_content += f"{self.B_ASSISTANT}{content.strip()}{self.E_ASSISTANT}"
        # Open the assistant turn for generation
        input_content += self.B_ASSISTANT
        self.last_input_size = len(input_content)
        return input_content

    def interpret_base(self, messages: List[Dict[str, str]]) -> str:
        users = [m for m in messages if m["role"] == "user"]
        assists = [m for m in messages if m["role"] == "assistant"]
        if not users:
            return ""
        last_user = users[-1]["content"].strip()
        last_ass = assists[-1]["content"].strip() if assists else None
        input_content = f"{self.BASE_INPUT} {last_user}"
        if last_ass:
            input_content += f"{self.BASE_RESPONSE} {last_ass}"
        return input_content
