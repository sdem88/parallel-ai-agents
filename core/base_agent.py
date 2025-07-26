"""
Base agent implementation with common functionality
"""

import asyncio
import time
from datetime import datetime
from typing import List, Optional
import structlog

from .interfaces import (
    AgentInterface, 
    AIProvider, 
    Task, 
    TaskStatus, 
    AgentCapability, 
    AgentMetrics
)


logger = structlog.get_logger(__name__)


class BaseAgent(AgentInterface):
    """Base implementation for all agents"""
    
    def __init__(
        self, 
        agent_id: str,
        name: str,
        capabilities: List[AgentCapability],
        ai_provider: Optional[AIProvider] = None,
        max_concurrent_tasks: int = 1
    ):
        self._agent_id = agent_id
        self._name = name
        self._capabilities = capabilities
        self._ai_provider = ai_provider
        self._max_concurrent_tasks = max_concurrent_tasks
        
        # State management
        self._current_tasks: List[str] = []
        self._metrics = AgentMetrics()
        self._is_healthy = True
        
        # Logging context
        self._logger = logger.bind(agent_id=agent_id, agent_name=name)
    
    @property
    def agent_id(self) -> str:
        return self._agent_id
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        return self._capabilities.copy()
    
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the task based on requirements"""
        if not task.agent_requirements:
            return True
        
        # Check if we have at least one required capability
        return any(cap in self._capabilities for cap in task.agent_requirements)
    
    async def execute_task(self, task: Task) -> Task:
        """Execute a task with full lifecycle management"""
        if not self.can_handle_task(task):
            task.status = TaskStatus.FAILED
            task.error = f"Agent {self.name} cannot handle task requirements: {task.agent_requirements}"
            return task
        
        # Check capacity
        if len(self._current_tasks) >= self._max_concurrent_tasks:
            task.status = TaskStatus.FAILED
            task.error = f"Agent {self.name} is at capacity ({self._max_concurrent_tasks} tasks)"
            return task
        
        # Start task execution
        task.status = TaskStatus.IN_PROGRESS
        task.assigned_agent = self.agent_id
        task.started_at = datetime.now()
        
        self._current_tasks.append(task.id)
        self._logger.info("Starting task execution", task_id=task.id, task_type=task.type)
        
        start_time = time.time()
        
        try:
            # Execute the actual work
            result = await self._execute_work(task)
            
            # Update task with results
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Update metrics
            execution_time = int((time.time() - start_time) * 1000)
            task.execution_time_ms = execution_time
            self._update_metrics(success=True, execution_time=execution_time)
            
            self._logger.info(
                "Task completed successfully", 
                task_id=task.id, 
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            # Handle execution errors
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now()
            
            execution_time = int((time.time() - start_time) * 1000)
            task.execution_time_ms = execution_time
            self._update_metrics(success=False, execution_time=execution_time)
            
            self._logger.error(
                "Task execution failed", 
                task_id=task.id, 
                error=str(e),
                execution_time_ms=execution_time
            )
            
        finally:
            # Cleanup
            if task.id in self._current_tasks:
                self._current_tasks.remove(task.id)
        
        return task
    
    async def _execute_work(self, task: Task) -> str:
        """Override this method in specific agent implementations"""
        if self._ai_provider:
            prompt = self._build_prompt(task)
            return await self._ai_provider.execute(prompt, task.context)
        else:
            # Fallback simulation
            await asyncio.sleep(0.5)  # Simulate work
            return f"✅ {self.name} completed: {task.description}"
    
    def _build_prompt(self, task: Task) -> str:
        """Build AI prompt based on agent capabilities and task"""
        capabilities_str = ", ".join([cap.value for cap in self._capabilities])
        
        prompt = f"""You are {self.name}, an AI agent with the following capabilities: {capabilities_str}.

Task Type: {task.type}
Description: {task.description}
Priority: {task.priority}/10

Please execute this task according to your capabilities and provide a detailed response.
"""
        
        # Add context if available
        if task.context:
            prompt += f"\nAdditional Context: {task.context}"
        
        return prompt
    
    def _update_metrics(self, success: bool, execution_time: int):
        """Update agent performance metrics"""
        if success:
            self._metrics.tasks_completed += 1
        else:
            self._metrics.tasks_failed += 1
        
        total_tasks = self._metrics.tasks_completed + self._metrics.tasks_failed
        self._metrics.success_rate = self._metrics.tasks_completed / total_tasks if total_tasks > 0 else 0
        
        # Update average execution time
        if self._metrics.tasks_completed > 0:
            current_avg = self._metrics.average_execution_time_ms
            self._metrics.average_execution_time_ms = (
                (current_avg * (self._metrics.tasks_completed - 1) + execution_time) 
                / self._metrics.tasks_completed
            )
        
        self._metrics.last_active = datetime.now()
    
    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics"""
        return AgentMetrics(
            tasks_completed=self._metrics.tasks_completed,
            tasks_failed=self._metrics.tasks_failed,
            average_execution_time_ms=self._metrics.average_execution_time_ms,
            success_rate=self._metrics.success_rate,
            last_active=self._metrics.last_active
        )
    
    async def health_check(self) -> bool:
        """Check agent health"""
        try:
            # Basic health checks
            if not self._is_healthy:
                return False
            
            # Check if AI provider is available (if configured)
            if self._ai_provider:
                test_result = await self._ai_provider.execute("Health check", {})
                return bool(test_result)
            
            return True
            
        except Exception as e:
            self._logger.error("Health check failed", error=str(e))
            return False
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} name={self.name}>"