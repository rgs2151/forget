import os

from steering.base import AutoModelForCausalLMWrapper

from .chat_templates import detect_template


def load_llm(model_path, gpu_id=0, template=None, hf_token=None):
    if template is None:
        template = detect_template(model_path)
    return AutoModelForCausalLMWrapper(
        hf_token=hf_token or os.getenv("HF_TOKEN"),
        model_path=model_path,
        instruction_end_marker=template.instruction_end_marker,
        tokenizer_path=model_path,
        gpu_id=gpu_id,
    )
