"""
Simulation AI Provider for testing and development
"""

import asyncio
import random
from typing import Dict, Any
import structlog

from core.interfaces import AIProvider


logger = structlog.get_logger(__name__)


class SimulationProvider(AIProvider):
    """Simulation provider for testing without real AI APIs"""
    
    def __init__(
        self,
        model: str = "simulation-v1",
        max_tokens: int = 1000,
        delay_range: tuple = (0.5, 2.0),
        failure_rate: float = 0.0
    ):
        self._model = model
        self._max_tokens = max_tokens
        self._delay_range = delay_range
        self._failure_rate = failure_rate
        
        self._logger = logger.bind(provider="simulation", model=model)
    
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Simulate AI execution with realistic delays"""
        context = context or {}
        
        self._logger.info("Simulating AI execution", prompt_length=len(prompt))
        
        # Simulate processing delay
        delay = random.uniform(*self._delay_range)
        await asyncio.sleep(delay)
        
        # Simulate occasional failures
        if random.random() < self._failure_rate:
            raise Exception("Simulated AI provider failure")
        
        # Generate simulated response based on context
        response = self._generate_response(prompt, context)
        
        self._logger.info(
            "Simulation completed",
            response_length=len(response),
            simulated_delay=delay
        )
        
        return response
    
    def _generate_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate a realistic simulation response"""
        role = context.get("role", "AI Assistant")
        capabilities = context.get("capabilities", [])
        
        # Base response
        response_parts = [f"✅ {role} simulation response:"]
        
        # Add capability-specific responses
        if "research" in str(capabilities).lower():
            response_parts.append("📊 Research findings: [Simulated data analysis complete]")
        
        if "code" in str(capabilities).lower():
            response_parts.append("💻 Code implementation: [Simulated code generation complete]")
        
        if "test" in str(capabilities).lower():
            response_parts.append("🧪 Testing results: [Simulated test execution passed]")
        
        if "review" in str(capabilities).lower():
            response_parts.append("📋 Review findings: [Simulated code review complete]")
        
        # Add prompt-specific response
        response_parts.append(f"Task '{prompt[:50]}...' completed successfully")
        
        # Add some realistic metrics
        response_parts.append(f"⏱️ Processing time: {random.randint(100, 2000)}ms")
        response_parts.append(f"📈 Confidence score: {random.randint(85, 98)}%")
        
        return "\n".join(response_parts)
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self._model
    
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported"""
        return self._max_tokens
    
    async def health_check(self) -> bool:
        """Simulation provider is always healthy"""
        return True