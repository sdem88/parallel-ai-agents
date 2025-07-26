"""
OpenAI GPT Provider with enterprise features
"""

import os
import asyncio
from typing import Dict, Any, Optional
import openai
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from core.interfaces import AIProvider


logger = structlog.get_logger(__name__)


class OpenAIProvider(AIProvider):
    """Enterprise-grade OpenAI GPT provider"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 60
    ):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key is required")
        
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        
        # Initialize client
        self._client = openai.AsyncOpenAI(
            api_key=self._api_key,
            timeout=timeout
        )
        
        self._logger = logger.bind(provider="openai", model=model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Execute prompt with OpenAI API"""
        context = context or {}
        
        try:
            self._logger.info("Executing prompt", prompt_length=len(prompt))
            
            # Build messages
            messages = []
            
            # Add system message if context provided
            system_prompt = self._build_system_prompt(context)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            # Make API call
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature
            )
            
            result = response.choices[0].message.content
            
            self._logger.info(
                "Prompt executed successfully",
                response_length=len(result),
                usage_prompt_tokens=getattr(response.usage, 'prompt_tokens', 0),
                usage_completion_tokens=getattr(response.usage, 'completion_tokens', 0),
                usage_total_tokens=getattr(response.usage, 'total_tokens', 0)
            )
            
            return result
            
        except openai.APIError as e:
            self._logger.error("OpenAI API error", error=str(e), error_type=type(e).__name__)
            raise
        except Exception as e:
            self._logger.error("Unexpected error in OpenAI provider", error=str(e))
            raise
    
    def _build_system_prompt(self, context: Dict[str, Any]) -> str:
        """Build system prompt from context"""
        if not context:
            return "You are a helpful AI assistant."
        
        system_parts = []
        
        if "role" in context:
            system_parts.append(f"You are {context['role']}.")
        
        if "capabilities" in context:
            caps = ", ".join(context["capabilities"])
            system_parts.append(f"Your capabilities include: {caps}.")
        
        if "instructions" in context:
            system_parts.append(context["instructions"])
        
        if "constraints" in context:
            system_parts.append(f"Constraints: {context['constraints']}")
        
        return " ".join(system_parts) if system_parts else "You are a helpful AI assistant."
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self._model
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported"""
        return self._max_tokens
    
    async def health_check(self) -> bool:
        """Check if the provider is healthy"""
        try:
            await self.execute("Hello", {})
            return True
        except Exception:
            return False