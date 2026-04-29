"""
Base wrapper for Llama models
"""

from typing import Optional, List, Union
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from .block import BlockOutputWrapper
from .find import find_instruction_end_postion, find_instruction_end_positions_batch


class BaseLlamaWrapper():
    """
    Base wrapper class for Llama models that contains common functionality
    """

    # Stored after batch calls so user can inspect per-sample instruction end positions
    last_from_positions: Optional[t.Tensor] = None

    def __init__(
        self,
        hf_token: Optional[str],
        model_path: str,
        instruction_end_marker: str,
        tokenizer_path: str,
        override_model_weights_path: Optional[str] = None,
        gpu_id: int = 0
    ):
        self.device = f"cuda:{gpu_id}" if t.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.instruction_end_marker = instruction_end_marker

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            token=hf_token,
        )
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        self.END_STR = t.tensor(
            self.encode(instruction_end_marker, add_special_tokens=False),
            device=self.device,
        )

        # Initialize model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, token=hf_token
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

        # Wrap model layers for activation tracking
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer,
                self.model.lm_head,
                self.model.model.norm,
            )
    
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

    def batch_decode(self, tokens: Union[List[int], t.Tensor]) -> str:
        if isinstance(tokens, t.Tensor):
            if tokens.dim() > 1:
                tokens = tokens[0]
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    def decode_batch(self, tokens: t.Tensor, skip_special_tokens: bool = False) -> List[str]:
        return [
            self.tokenizer.decode(seq, skip_special_tokens=skip_special_tokens)
            for seq in tokens
        ]

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, token_id: Union[int, t.Tensor]) -> str:
        if isinstance(token_id, t.Tensor):
            token_id = int(token_id.item())
        return self.tokenizer.decode([token_id])

    # ------------------------------------------------------------------ #
    #  Single-sample methods (existing API)
    # ------------------------------------------------------------------ #

    def generate(
        self,
        tokens: t.Tensor,
        max_new_tokens: int = 100,
        top_k: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> str:
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            attention_mask = t.ones_like(tokens)
            generated = self.model.generate(
                inputs=tokens,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_k=top_k,
                temperature=temperature,
                pad_token_id=self.pad_token_id,
                use_cache=True,
            )
            return self.batch_decode(generated)

    def generate_from_prompt(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        top_k: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> str:
        tokens_list = self.tokenize(prompt)
        tokens = t.tensor(tokens_list).unsqueeze(0).to(self.device)
        return self.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            do_sample=do_sample,
            temperature=temperature,
        )

    def forward_from_prompt(self, prompt: str) -> t.Tensor:
        tokens_list = self.tokenize(prompt)
        tokens = t.tensor(tokens_list).unsqueeze(0).to(self.device)
        with t.no_grad():
            outputs = self.model(tokens)
            return outputs.logits

    def get_logits(self, tokens: t.Tensor) -> t.Tensor:
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits

    def get_logits_from_prompt(self, prompt: str) -> t.Tensor:
        tokens_list = self.tokenize(prompt)
        tokens = t.tensor(tokens_list).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    # ------------------------------------------------------------------ #
    #  Batch methods
    # ------------------------------------------------------------------ #

    def forward_from_prompts(self, prompts: List[str]) -> t.Tensor:
        """Batched forward — for activation collection (no steering)."""
        batch = self.tokenize_batch(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        with t.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def get_logits_from_prompts(self, prompts: List[str]) -> t.Tensor:
        """Batched logits with steering. Auto-detects per-sample from_positions."""
        batch = self.tokenize_batch(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        fps = find_instruction_end_positions_batch(input_ids, self.END_STR, attention_mask)
        self.set_from_positions(fps)
        self.last_from_positions = fps
        with t.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def generate_from_prompts(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        top_k: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> List[str]:
        """Batched generation with steering. Auto-detects per-sample from_positions.

        Returns list of decoded strings (one per sample, full sequence).
        Access llm.last_from_positions for the (batch,) tensor of instruction end column indices.
        """
        batch = self.tokenize_batch(prompts)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        fps = find_instruction_end_positions_batch(input_ids, self.END_STR, attention_mask)
        self.set_from_positions(fps)
        self.last_from_positions = fps
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
            return self.decode_batch(generated, skip_special_tokens=False)

    # ------------------------------------------------------------------ #
    #  Accessors / mutators
    # ------------------------------------------------------------------ #

    def get_last_activations(self, layer: int) -> t.Tensor:
        return self.model.model.layers[layer].activations

    def set_save_internal_decodings(self, value: bool) -> None:
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: Union[int, t.Tensor]) -> None:
        """Set from_position on all layers. Accepts int or (batch,) tensor."""
        if not isinstance(pos, t.Tensor):
            pos = t.tensor([pos], device=self.device)
        for layer in self.model.model.layers:
            layer.from_position = pos

    def set_add_activations(self, layer: int, activations: t.Tensor) -> None:
        """Wrap in AddSteer and set as steering op."""
        self.model.model.layers[layer].add(activations)

    def set_steering_op(self, layer: int, op) -> None:
        self.model.model.layers[layer].steering_op = op

    def reset_all(self) -> None:
        for layer in self.model.model.layers:
            layer.reset()
