from typing import List, Optional, Dict
import torch as t
from transformers.tokenization_utils import PreTrainedTokenizer

from ...abstract import AbstractTokenizer
from ....chat import Chat


class Llama2Tokenizer(AbstractTokenizer):
    """
    Llama 2 Tokenizer Wrapper
    
    chat format:
    .. code-block:: python
        <s> [INST] 
        <<SYS>>system text<</SYS>>
        user text 
        [/INST]
        assistant text </s>
        <s> [INST] 
        user text 
        [/INST]
    
    base format:
    .. code-block:: python
        <s>
        Input: user text
        Response: assistant text
        </s>
    """
    
    # Chat formatting tags
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
    B_USER = B_INST
    E_USER = E_INST
    B_ASSISTANT = ""
    E_ASSISTANT = ""

    # Base formatting tags
    BASE_INPUT = "Input:"
    BASE_RESPONSE = "\nResponse:"

    # Position markers for activation steering
    ADD_FROM_POS_CHAT = E_INST
    ADD_FROM_POS_BASE = BASE_RESPONSE

    # End of prompt sequence token tensor
    END_STR: Optional[t.Tensor] = None
    
    # ChatML special tokens
    BOS, EOS = "<s>", "</s>"
    
    # extras
    last_input_size: Optional[int] = None

    def __init__(self, tokenizer: PreTrainedTokenizer, use_chat: bool = True):
        self.tokenizer = tokenizer
        self.use_chat = use_chat

    def interpret_chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Interpret messages in chat mode format
        """
        input_content = ""
        for i, m in enumerate(messages):
            role, content = m["role"], m["content"]
            
            if role == "user":
                
                if messages[i - 1]["role"] == "system":
                    sys_text = f"{self.B_SYS}{messages[i - 1]['content']}{self.E_SYS}"
                    input_content += f"{self.B_USER} {sys_text} {content.strip()} {self.E_USER}"
                else: 
                    input_content += f"{self.BOS}{self.B_USER} {content.strip()} {self.E_USER}"
        
            elif role == "assistant" and content is not None:
                input_content += f"{self.B_ASSISTANT}{content.strip()}{self.E_ASSISTANT}{self.EOS}"
        
        self.last_input_size = len(input_content) + 3
        return input_content

    def interpret_base(self, messages: List[Dict[str, str]]) -> str:
        """
        Interpret messages in base mode format
        """
        # base mode: use last user and assistant
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
