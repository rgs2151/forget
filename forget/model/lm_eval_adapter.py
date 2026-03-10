"""
lm-evaluation-harness adapter for steered models.

Usage:
    # Programmatic (recommended)
    from model.lm_eval_adapter import SteeredHFLM
    
    lm = SteeredHFLM(
        wrapper=your_llama_wrapper,        # BaseLlamaWrapper with blocks already wrapped
        steering_vectors=steering_vectors,  # dict: {layer_idx: tensor}
        steering_scale=1.0,
        steer_from_position=None,          # None = steer all positions
    )
    
    import lm_eval
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag", "mmlu", ...],
        batch_size=1,
    )

    # CLI (after registering)
    # lm_eval --model steered_hf --model_args wrapper_path=...,vectors_path=...,scale=1.0 --tasks hellaswag
"""

from typing import Optional, Dict
import torch as t
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model


@register_model("steered_hf")
class SteeredHFLM(HFLM):
    """
    lm-eval-harness compatible wrapper around a steered Llama model.
    
    Accepts an already-configured BaseLlamaWrapper (with BlockOutputWrappers 
    on each layer), applies steering vectors, and exposes it to lm-eval.
    """

    def __init__(
        self,
        wrapper=None,
        steering_vectors: Optional[Dict[int, t.Tensor]] = None,
        steering_scale: float = 1.0,
        steer_from_position: Optional[int] = None,
        batch_size: int = 1,
        max_length: Optional[int] = None,
        **kwargs,
    ):
        """
        Args:
            wrapper: A BaseLlamaWrapper instance (model already loaded with block wrappers).
            steering_vectors: Dict mapping layer index -> steering vector tensor.
            steering_scale: Multiplier for steering vectors.
            steer_from_position: Token position to start steering from.
                                 None = steer all positions (default for evals).
            batch_size: Batch size for evaluation.
            max_length: Max sequence length. If None, uses model's config.
        """
        assert wrapper is not None, "Must provide a BaseLlamaWrapper instance"
        
        self._wrapper = wrapper

        # Apply steering vectors
        self._wrapper.reset_all()
        if steering_vectors is not None:
            for layer_idx, vec in steering_vectors.items():
                scaled = vec * steering_scale
                self._wrapper.set_add_activations(layer_idx, scaled)

        # Set position for steering (None = all positions)
        if steer_from_position is not None:
            self._wrapper.set_from_positions(steer_from_position)

        # Initialize HFLM with the pre-loaded model and tokenizer
        # pretrained="__manual__" signals we're providing the model ourselves
        super().__init__(
            pretrained="__manual__",
            batch_size=batch_size,
            max_length=max_length,
            **kwargs,
        )

    def _create_model(self, *args, **kwargs):
        """Override to use our already-wrapped model instead of loading from HF."""
        self._model = self._wrapper.model

    def _create_tokenizer(self, *args, **kwargs):
        """Override to use the wrapper's tokenizer."""
        self.tokenizer = self._wrapper.tokenizer.tokenizer  # The underlying HF tokenizer

    @property
    def device(self):
        return t.device(self._wrapper.device)

    def update_steering(
        self,
        steering_vectors: Dict[int, t.Tensor],
        scale: float = 1.0,
        from_position: Optional[int] = None,
    ):
        """
        Update steering vectors on the fly (e.g., to sweep scales).
        Useful for running evals at multiple steering strengths.
        """
        self._wrapper.reset_all()
        for layer_idx, vec in steering_vectors.items():
            self._wrapper.set_add_activations(layer_idx, vec * scale)
        if from_position is not None:
            self._wrapper.set_from_positions(from_position)

    def clear_steering(self):
        """Remove all steering to get vanilla model behavior."""
        self._wrapper.reset_all()
