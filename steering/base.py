"""
Base wrapper for AutoModelForCausalLM models
"""

from typing import Optional, List, Union
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from .block import BlockOutputWrapper
from .find import find_instruction_end_positions_batch


class AutoModelForCausalLMWrapper():
    """
    Base wrapper class for AutoModelForCausalLM models that contains common functionality
    """
    def __init__(
        self,
        hf_token: Optional[str],
        model_path: str,
        instruction_end_marker: str,
        tokenizer_path: str,
        override_model_weights_path: Optional[str] = None,
        gpu_id: int = 0,
        trust_remote_code: bool = False,
    ):
        self.device = f"cuda:{gpu_id}" if t.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.instruction_end_marker = instruction_end_marker

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None: self.pad_token_id = self.tokenizer.eos_token_id
        self.END_STR = t.tensor(
            self.encode(instruction_end_marker, add_special_tokens=False),
            device=self.device,
        )

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

        # Load custom weights if provided
        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))

        # Move model to device
        self.model = self.model.to(self.device)

        # Clear HuggingFace generation config defaults to avoid warnings
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.model.generation_config.max_length = None

        # Instrument model layers without replacing the underlying HF blocks.
        self.layer_wrappers = [
            BlockOutputWrapper(
                layer,
                self.model.lm_head,
                self.model.model.norm,
            )
            for layer in self.model.model.layers
        ]
    
    # ------------------------------------------------------------------ #
    #  Tokenizer / decoder methods
    # ------------------------------------------------------------------ #
    
    def tokenize(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def tokenize_batch(self, prompts: List[str]) -> dict[str, t.Tensor]:
        """Left-pad a list of formatted prompts into a batch.

        Returns:
            {"input_ids": (batch, max_seq_len), "attention_mask": (batch, max_seq_len)}
        """
        token_lists = [self.tokenize(prompt) for prompt in prompts]
        max_len = max(len(token_list) for token_list in token_lists)

        input_ids = []
        attention_mask = []
        for token_list in token_lists:
            pad_len = max_len - len(token_list)
            input_ids.append([self.pad_token_id] * pad_len + token_list)
            attention_mask.append([0] * pad_len + [1] * len(token_list))

        return {
            "input_ids": t.tensor(input_ids),
            "attention_mask": t.tensor(attention_mask),
        }

    def decode_batch(self, tokens: t.Tensor, skip_special_tokens: bool = False) -> List[str]:
        return [
            self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in tokens
        ]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    # ------------------------------------------------------------------ #
    #  Batch methods
    # ------------------------------------------------------------------ #

    def batch_forward(self, prompts: List[str]) -> None:
        """Batched forward — for activation collection. Skips lm_head (only the layer hooks matter)."""
        self.reset_all()
        batch = self.tokenize_batch(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with t.no_grad():
            self.model.model(input_ids=input_ids, attention_mask=attention_mask)

        del input_ids, attention_mask
        t.cuda.empty_cache()

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        top_k: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
        intervention_start: str = "assistant",
    ) -> List[str]:
        """Batched generation with steering. Auto-detects per-sample from_positions.
        """
        batch = self.tokenize_batch(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        if intervention_start == "assistant":
            fps = find_instruction_end_positions_batch(input_ids, self.END_STR, attention_mask)
            self.set_from_positions(fps)
        elif intervention_start != "prefill":
            raise ValueError(f"unknown intervention_start {intervention_start!r}")
        with t.no_grad():
            generated = self.model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                pad_token_id=self.pad_token_id,
                use_cache=True,
            )
            decoded = self.decode_batch(generated, skip_special_tokens=False)
            
        del input_ids, attention_mask, generated
        t.cuda.empty_cache()
        return decoded

    def batch_next_token_option_probs(self, prompts: List[str], option_token_ids: dict[str, list[int]]):
        self.set_capture_activations(False)
        batch = self.tokenize_batch(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = list(option_token_ids)
        with t.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            hidden = outputs.last_hidden_state[:, -1, :].contiguous()
            label_logits = []
            hidden = hidden.float()
            for label in labels:
                ids = t.tensor(option_token_ids[label], device=self.device)
                weight = self.model.lm_head.weight[ids].float()
                logits = hidden @ weight.T
                if self.model.lm_head.bias is not None:
                    logits = logits + self.model.lm_head.bias[ids].float()
                label_logits.append(logits.max(dim=1).values)
            probs = t.softmax(t.stack(label_logits, dim=1).float(), dim=1).detach().cpu()

        rows = []
        for row in probs:
            rows.append({label: float(row[i]) for i, label in enumerate(labels)})

        del input_ids, attention_mask, outputs, hidden, logits, probs
        self.set_capture_activations(True)
        t.cuda.empty_cache()
        return rows

    # ------------------------------------------------------------------ #
    #  Accessors / mutators
    # ------------------------------------------------------------------ #

    def get_last_activations(self, layer: int) -> t.Tensor:
        return self.layer_wrappers[layer].activations

    def set_save_internal_decodings(self, value: bool) -> None:
        for layer in self.layer_wrappers:
            layer.save_internal_decodings = value

    def set_capture_activations(self, value):
        for layer in self.layer_wrappers:
            layer.capture_activations = value

    def set_from_positions(self, pos: Union[int, t.Tensor]) -> None:
        """Set from_position on all layers. Accepts int or (batch,) tensor."""
        if not isinstance(pos, t.Tensor):
            pos = t.tensor([pos], device=self.device)
        for layer in self.layer_wrappers:
            layer.from_position = pos

    def set_steering_op(self, layer: int, op) -> None:
        self.layer_wrappers[layer].steering_op = op

    def reset_all(self) -> None:
        for layer in self.layer_wrappers:
            layer.reset()
