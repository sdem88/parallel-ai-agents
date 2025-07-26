"""
Core interfaces for the Agent System
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentCapability(Enum):
    """Agent capabilities"""
    RESEARCH = "research"
    CODE_GENERATION = "code_generation"
    TESTING = "testing"
    ANALYSIS = "analysis"
    DESIGN = "design"
    ORCHESTRATION = "orchestration"


@dataclass
class Task:
    """Task definition with enhanced metadata"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    priority: int = 5  # 1-10, higher is more important
    agent_requirements: List[AgentCapability] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution metadata
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: Optional[int] = None


@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_execution_time_ms: float = 0.0
    success_rate: float = 0.0
    last_active: Optional[datetime] = None


class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def execute(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Execute a prompt with the AI provider"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name"""
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get maximum tokens supported"""
        pass


class AgentInterface(ABC):
    """Abstract base class for all agents"""
    
    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Unique agent identifier"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human readable agent name"""
        pass
    
    @property
    @abstractmethod
    def capabilities(self) -> List[AgentCapability]:
        """List of agent capabilities"""
        pass
    
    @abstractmethod
    async def execute_task(self, task: Task) -> Task:
        """Execute a task and return updated task with results"""
        pass
    
    @abstractmethod
    def can_handle_task(self, task: Task) -> bool:
        """Check if agent can handle the given task"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> AgentMetrics:
        """Get agent performance metrics"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if agent is healthy and ready"""
        pass


class TaskManagerInterface(ABC):
    """Abstract base class for task management"""
    
    @abstractmethod
    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        pass
    
    @abstractmethod
    async def get_task_status(self, task_id: str) -> TaskStatus:
        """Get task execution status"""
        pass
    
    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        pass
    
    @abstractmethod
    async def get_task_result(self, task_id: str) -> Optional[Any]:
        """Get task execution result"""
        pass


class EventBusInterface(ABC):
    """Abstract base class for event communication"""
    
    @abstractmethod
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event"""
        pass
    
    @abstractmethod
    async def subscribe(self, event_type: str, callback) -> None:
        """Subscribe to an event type"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, event_type: str, callback) -> None:
        """Unsubscribe from an event type"""
        pass