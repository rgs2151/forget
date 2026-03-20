"""
Base wrapper for Llama models
"""

from typing import Optional
import torch as t
from transformers import AutoModelForCausalLM

from ..abstract import AbstractWrapper, AbstractTokenizer
from .src import BlockOutputWrapper
from ...utils.helpers import find_instruction_end_postion
from ...chat import Chat

class BaseLlamaWrapper(AbstractWrapper):
    """
    Base wrapper class for Llama models that contains common functionality
    """
    
    def __init__(
        self,
        hf_token: Optional[str],
        model_path: str,
        tokenizer: AbstractTokenizer,
        override_model_weights_path: Optional[str] = None,
        gpu_id: int = 0
    ):
        """
        Initialize the base Llama wrapper
        
        Args:
            hf_token: HuggingFace token for accessing models
            model_path: Path to the model on HuggingFace
            tokenizer: The HuggingFace tokenizer instance
            tokenizer: The appropriate tokenizer wrapper for the model
            override_model_weights_path: Optional path to override model weights
        """
        self.device = f"cuda:{gpu_id}" if t.cuda.is_available() else "cpu"
        self.model_path = model_path
        
        # Set the tokenizer wrapper
        self.tokenizer = tokenizer
        
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
        
        # self.model.config.use_cache = False
        # self.model.config.output_hidden_states = True
        # then outputs.hidden_states[layer_index+1] is [1, prompt_len+10, D]
        
        # Wrap model layers for activation tracking
        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, 
                self.model.lm_head, 
                self.model.model.norm, 
                self.tokenizer
            )

        # Register the model end of prompt sequence token
        self.END_STR = self.tokenizer.END_STR

    def generate(
        self,
        tokens: t.Tensor,
        max_new_tokens: int = 100,
        top_k: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text from input tokens
        
        Args:
            tokens: Input token tensor
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Number of highest probability tokens to keep for sampling
            temperature: Sampling temperature (1.0 = no scaling)
            
        Returns:
            Generated text
        """
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
                pad_token_id=self.tokenizer.tokenizer.eos_token_id,
                use_cache=True,
            )
            return self.tokenizer.batch_decode(generated)

    def generate_from_chat(
        self,
        chat: Chat,
        max_new_tokens: int = 50,
        top_k: int = 50,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate text given a Chat instance
        
        Args:
            chat: Chat instance containing the conversation history
            max_new_tokens: Maximum number of new tokens to generate
            top_k: Number of highest probability tokens to keep for sampling
            do_sample: Whether to sample from the model. Set to False for greedy decoding.
            temperature: Sampling temperature (1.0 = no scaling)
            
        Returns:
            Generated text
        """
        # tokenize via wrapper
        tokens_list = self.tokenizer.tokenize(chat)
        tokens = t.tensor(tokens_list).unsqueeze(0).to(self.device)
        return self.generate(
            tokens,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            do_sample=do_sample,
            temperature=temperature,
        )
        
    def forward_from_chat(
        self,
        chat: Chat,
    ) -> t.Tensor:
        tokens_list = self.tokenizer.tokenize(chat)
        tokens = t.tensor(tokens_list).unsqueeze(0).to(self.device)
        with t.no_grad():
            outputs = self.model(tokens)
            return outputs.logits

    # Get functions
    def get_logits(self, tokens: t.Tensor) -> t.Tensor:
        """Get logits for input tokens"""
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits

    def get_logits_from_chat(self, chat: Chat) -> t.Tensor:
        """Get logits given a Chat instance"""
        tokens_list = self.tokenizer.tokenize(chat)
        tokens = t.tensor(tokens_list).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer: int) -> t.Tensor:
        """Get activations from a specific layer"""
        return self.model.model.layers[layer].activations

    # Set functions
    def set_save_internal_decodings(self, value: bool) -> None:
        """Enable/disable saving internal decodings"""
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int) -> None:
        """Set the position to start from for each layer"""
        for layer in self.model.model.layers:
            layer.from_position = pos

    def set_add_activations(self, layer: int, activations: t.Tensor) -> None:
        """Set activations to add to a specific layer"""
        self.model.model.layers[layer].add(activations)

    def set_steering_op(self, layer: int, op) -> None:
        self.model.model.layers[layer].steering_op = op

    def reset_all(self) -> None:
        """Reset all layer states"""
        for layer in self.model.model.layers:
            layer.reset()