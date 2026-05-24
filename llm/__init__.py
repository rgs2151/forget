import transformers.utils.logging

transformers.utils.logging.set_verbosity_error()

from .chat_templates import (
    EXACT_MATCHES,
    LLAMA3,
    MISTRAL,
    QWEN,
    ChatTemplate,
    detect_template,
)
from .gpu import GPUPool
from .model import load_llm
