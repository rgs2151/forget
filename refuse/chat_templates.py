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

TEMPLATES = {
    "llama3": LLAMA3,
    "qwen": QWEN,
}


def detect_template(model_path):
    name = model_path.lower()
    if "llama-3" in name or "llama3" in name:
        return LLAMA3
    if "qwen" in name:
        return QWEN
    raise ValueError(
        f"No chat template registered for {model_path!r}; "
        f"pass template= explicitly or register one in chat_templates.TEMPLATES."
    )
