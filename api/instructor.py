"""
LLM provider implementations using instructor for structured outputs.

Supports OpenAI, Anthropic, Together, and other providers via instructor.
"""

import instructor
from pydantic import BaseModel
import asyncio
from typing import Any, List, Optional, Type, TypeVar
from tqdm.asyncio import tqdm

T = TypeVar('T', bound=BaseModel)


class InstructorLLM:
    """
    LLM implementation using instructor for structured outputs.
    
    Pass the fully initialized instance to agents (architects, writers).
    
    Examples:
        api = InstructorLLM("openai/gpt-4o-mini")
        api = InstructorLLM("anthropic/claude-3-5-sonnet-20241022")
        api = InstructorLLM("together/meta-llama/Llama-4-Scout-17B-16E-Instruct")
        
        # With options
        api = InstructorLLM("openai/gpt-4o", concurrency=20)
        
        # Usage with agents
        architect = metricArchitect(api=api)
        writer = spatialWriter(api=api)
    """
    
    provider_model: str
    concurrency: int
    model: str
    
    def __init__(
        self,
        provider_model: str,
        concurrency: int = 10,
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            provider_model: Provider and model in format "provider/model"
                           e.g. "openai/gpt-4o-mini", "anthropic/claude-3-5-sonnet-20241022"
            concurrency: Max concurrent requests for batch operations (default: 10)
            **kwargs: Additional arguments passed to instructor.from_provider()
        """
        self.provider_model = provider_model
        self.concurrency = concurrency
        self.client = instructor.from_provider(provider_model, async_client=True, **kwargs)
        
        # Extract just the model name for the API call
        parts = provider_model.split("/", 1)
        self.model = parts[1] if len(parts) > 1 else provider_model
    
    def _make_messages(self, prompt: str, system: str = "") -> list[dict]:
        """Build messages list with optional system prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    async def respond(
        self,
        prompt: str,
        response_model: Type[T],
        system: str = "",
        max_retries: int = 0,
    ) -> T:
        return await self.client.create(
            model=self.model,
            response_model=response_model,
            messages=self._make_messages(prompt, system),
            max_retries=max_retries,
        )
    
    async def _rate_limited_respond(
        self,
        prompt: str,
        response_model: Type[T],
        semaphore: asyncio.Semaphore,
        system: str = "",
        max_retries: int = 0,
    ) -> T:
        async with semaphore:
            return await self.respond(prompt, response_model, system, max_retries=max_retries)
    
    async def batch_respond(
        self,
        prompts: List[str],
        response_models: List[Type[T]],
        system: str = "",
        concurrency: Optional[int] = None,
        desc: str = "Processing",
        max_retries: int = 0,
    ) -> List[T]:
        """Order is GUARANTEED: results[i] corresponds to prompts[i]."""
        limit = concurrency or self.concurrency
        semaphore = asyncio.Semaphore(limit)
        
        tasks = [
            self._rate_limited_respond(prompt, model, semaphore, system, max_retries=max_retries)
            for prompt, model in zip(prompts, response_models, strict=True)
        ]
        
        return await tqdm.gather(*tasks, desc=desc)



