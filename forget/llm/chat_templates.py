from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTemplate:
    bos: str
    b_sys: str
    e_sys: str
    b_user: str
    e_user: str
    b_assistant: str
    e_assistant: str

    def render(self, system, user, assistant=None):
        pieces = [self.bos, self.b_sys, system, self.e_sys,
                  self.b_user, user, self.e_user]
        if assistant is None:
            pieces.append(self.b_assistant)
        else:
            pieces.extend([self.b_assistant, assistant, self.e_assistant])
        return "".join(pieces)

    def trim_to_last_assistant(self, raw):
        idx = raw.rfind(self.b_assistant)
        text = raw[idx + len(self.b_assistant):] if idx != -1 else raw
        return text.replace(self.e_assistant, "").strip()

    def sanitize(self, text):
        return text.replace(self.e_assistant, "").strip()

    @property
    def instruction_end_marker(self):
        return self.b_assistant

    @property
    def assistant_end_marker(self):
        return self.e_assistant

    @property
    def idk_answer(self):
        return f"I don't know.{self.e_assistant}"


LLAMA3 = ChatTemplate(
    bos="<|begin_of_text|>",
    b_sys="<|start_header_id|>system<|end_header_id|>\n\n",
    e_sys="<|eot_id|>",
    b_user="<|start_header_id|>user<|end_header_id|>\n\n",
    e_user="<|eot_id|>",
    b_assistant="<|start_header_id|>assistant<|end_header_id|>\n\n",
    e_assistant="<|eot_id|>",
)

QWEN = ChatTemplate(
    bos="",
    b_sys="<|im_start|>system\n",
    e_sys="<|im_end|>\n",
    b_user="<|im_start|>user\n",
    e_user="<|im_end|>\n",
    b_assistant="<|im_start|>assistant\n",
    e_assistant="<|im_end|>",
)

MISTRAL = ChatTemplate(
    bos="<s>",
    b_sys="[INST] ",
    e_sys="\n\n",
    b_user="",
    e_user=" ",
    b_assistant="[/INST] ",
    e_assistant="</s>",
)

MISTRAL_SMALL = ChatTemplate(
    bos="<s>",
    b_sys="[SYSTEM_PROMPT]",
    e_sys="[/SYSTEM_PROMPT]",
    b_user="[INST]",
    e_user="",
    b_assistant="[/INST]",
    e_assistant="</s>",
)

PHI4 = ChatTemplate(
    bos="",
    b_sys="<|im_start|>system<|im_sep|>",
    e_sys="<|im_end|>",
    b_user="<|im_start|>user<|im_sep|>",
    e_user="<|im_end|>",
    b_assistant="<|im_start|>assistant<|im_sep|>",
    e_assistant="<|im_end|>",
)

PHI4_MINI = ChatTemplate(
    bos="",
    b_sys="<|system|>",
    e_sys="<|end|>",
    b_user="<|user|>",
    e_user="<|end|>",
    b_assistant="<|assistant|>",
    e_assistant="<|end|>",
)

EXACT_MATCHES = {
    "meta-llama/Llama-3.1-8B-Instruct": LLAMA3,
    "meta-llama/Llama-3.2-1B-Instruct": LLAMA3,
    "meta-llama/Llama-3.2-3B-Instruct": LLAMA3,
    "mistralai/Mistral-7B-Instruct-v0.3": MISTRAL,
    "mistralai/Mistral-Small-24B-Instruct-2501": MISTRAL_SMALL,
    "Qwen/Qwen2.5-0.5B-Instruct": QWEN,
    "Qwen/Qwen2.5-3B-Instruct": QWEN,
    "Qwen/Qwen2.5-7B-Instruct": QWEN,
    "Qwen/Qwen2.5-14B-Instruct": QWEN,
    "AtlaAI/Selene-1-Mini-Llama-3.1-8B": LLAMA3,
    "microsoft/phi-4": PHI4,
    "microsoft/Phi-4-mini-instruct": PHI4_MINI,
}


def detect_template(model_path):
    if model_path in EXACT_MATCHES:
        return EXACT_MATCHES[model_path]
    raise ValueError(
        f"No template registered for {model_path!r}. "
        f"Either use one of the verified models {sorted(EXACT_MATCHES)}, "
        f"or pass template= explicitly to override."
    )
