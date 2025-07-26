"""
Anthropic Claude AI Provider with enterprise features
"""

import os
import asyncio
from typing import Dict, Any, Optional
import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from core.interfaces import AIProvider


logger = structlog.get_logger(__name__)


class AnthropicProvider(AIProvider):
    """Enterprise-grade Anthropic Claude provider"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-opus-20240229",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        timeout: int = 60
    ):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic API key is required")
        
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._timeout = timeout
        
        # Initialize client with retry configuration
        self._client = anthropic.Anthropic(
            api_key=self._api_key,
            timeout=timeout
        )
        
        self._logger = logger.bind(provider="anthropic", model=model)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Execute prompt with Claude API"""
        context = context or {}
        
        try:
            self._logger.info("Executing prompt", prompt_length=len(prompt))
            
            # Build system prompt if context provided
            system_prompt = self._build_system_prompt(context)
            
            # Make API call
            message = await asyncio.to_thread(
                self._client.messages.create,
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            result = message.content[0].text
            
            self._logger.info(
                "Prompt executed successfully",
                response_length=len(result),
                usage_input_tokens=getattr(message.usage, 'input_tokens', 0),
                usage_output_tokens=getattr(message.usage, 'output_tokens', 0)
            )
            
            return result
            
        except anthropic.APIError as e:
            self._logger.error("Anthropic API error", error=str(e), error_type=type(e).__name__)
            raise
        except Exception as e:
            self._logger.error("Unexpected error in Anthropic provider", error=str(e))
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