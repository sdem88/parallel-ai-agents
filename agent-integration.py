#!/usr/bin/env python3
"""
Система параллельных агентов с интеграцией Claude Opus 4 и OpenAI O3
"""

import asyncio
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import anthropic
import openai
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    RESEARCH = "research" 
    CODE_GENERATOR = "code_generator"
    ANALYZER = "analyzer"
    TESTER = "tester"

@dataclass
class Task:
    id: str
    type: str
    description: str
    dependencies: List[str] = None
    priority: int = 5
    status: str = "pending"
    result: Any = None
    error: Optional[str] = None

class Agent:
    def __init__(self, agent_type: AgentType, config: Dict):
        self.type = agent_type
        self.config = config
        self.name = config["name"]
        self.model = config["model"]
        self.capabilities = config["capabilities"]
        
    async def execute_task(self, task: Task) -> Task:
        """Выполняет задачу в зависимости от типа агента"""
        logger.info(f"{self.name} начинает задачу: {task.id}")
        
        try:
            if self.model == "claude-opus-4":
                result = await self._execute_claude_task(task)
            elif self.model == "openai-o3":
                result = await self._execute_openai_task(task)
            else:
                raise ValueError(f"Неизвестная модель: {self.model}")
                
            task.status = "completed"
            task.result = result
            
        except Exception as e:
            logger.error(f"Ошибка в {self.name}: {str(e)}")
            task.status = "failed"
            task.error = str(e)
            
        return task
    
    async def _execute_claude_task(self, task: Task) -> Any:
        """Выполнение через Claude Opus 4 API"""
        client = anthropic.Anthropic()
        
        system_prompt = self._get_system_prompt()
        
        message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=4096,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": task.description}
            ]
        )
        
        return message.content[0].text
    
    async def _execute_openai_task(self, task: Task) -> Any:
        """Выполнение через OpenAI O3 API"""
        client = openai.OpenAI()
        
        response = client.chat.completions.create(
            model="o3",  # Предполагаемое название модели O3
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": task.description}
            ],
            temperature=0.7,
            max_tokens=4096
        )
        
        return response.choices[0].message.content
    
    def _get_system_prompt(self) -> str:
        """Генерирует системный промпт для агента"""
        prompts = {
            AgentType.ORCHESTRATOR: """Вы - главный координатор системы агентов. 
                Ваши задачи: декомпозиция сложных задач, распределение работы, 
                отслеживание прогресса и синтез результатов.""",
                
            AgentType.RESEARCH: """Вы - исследовательский агент. 
                Специализируетесь на поиске информации, анализе документации 
                и сборе данных из различных источников.""",
                
            AgentType.CODE_GENERATOR: """Вы - агент генерации кода. 
                Создаете высококачественный, оптимизированный код 
                на различных языках программирования.""",
                
            AgentType.ANALYZER: """Вы - аналитический агент. 
                Выполняете code review, анализ производительности 
                и поиск потенциальных проблем.""",
                
            AgentType.TESTER: """Вы - тестирующий агент. 
                Создаете тесты, выполняете тестирование 
                и анализируете покрытие кода."""
        }
        
        return prompts.get(self.type, "Вы - специализированный агент.")

class ParallelAgentSystem:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.agents = self._initialize_agents()
        self.task_queue = asyncio.Queue()
        self.results = {}
        self.executor = ThreadPoolExecutor(
            max_workers=self.config["execution"]["max_parallel_agents"]
        )
        
    def _initialize_agents(self) -> Dict[AgentType, Agent]:
        """Инициализирует всех агентов"""
        agents = {}
        for agent_type in AgentType:
            if agent_type.value in self.config["agents"]:
                agent_config = self.config["agents"][agent_type.value]
                agents[agent_type] = Agent(agent_type, agent_config)
        return agents
    
    async def process_request(self, request: str) -> Dict[str, Any]:
        """Обрабатывает запрос пользователя"""
        # Orchestrator декомпозирует задачу
        decomposition_task = Task(
            id="decompose_001",
            type="decomposition",
            description=f"Декомпозировать задачу: {request}"
        )
        
        orchestrator = self.agents[AgentType.ORCHESTRATOR]
        decomposed = await orchestrator.execute_task(decomposition_task)
        
        # Создаем подзадачи на основе декомпозиции
        subtasks = self._create_subtasks(decomposed.result)
        
        # Выполняем задачи параллельно
        results = await self._execute_parallel_tasks(subtasks)
        
        # Синтезируем результаты
        synthesis_task = Task(
            id="synthesis_001",
            type="synthesis",
            description=f"Синтезировать результаты: {json.dumps(results)}"
        )
        
        final_result = await orchestrator.execute_task(synthesis_task)
        
        return {
            "request": request,
            "subtasks": len(subtasks),
            "results": results,
            "synthesis": final_result.result
        }
    
    def _create_subtasks(self, decomposition: str) -> List[Task]:
        """Создает подзадачи на основе декомпозиции"""
        # Здесь должна быть логика парсинга декомпозиции
        # Для примера создаем фиксированные задачи
        return [
            Task(id="research_001", type="research", 
                 description="Исследовать лучшие практики"),
            Task(id="code_001", type="code_generation",
                 description="Создать базовую реализацию"),
            Task(id="analyze_001", type="analysis",
                 description="Проанализировать созданный код"),
            Task(id="test_001", type="testing",
                 description="Создать и выполнить тесты")
        ]
    
    async def _execute_parallel_tasks(self, tasks: List[Task]) -> Dict[str, Any]:
        """Выполняет задачи параллельно"""
        async def execute_with_agent(task: Task) -> Task:
            agent_type = self._get_agent_type_for_task(task)
            if agent_type in self.agents:
                return await self.agents[agent_type].execute_task(task)
            return task
        
        # Запускаем все задачи параллельно
        completed_tasks = await asyncio.gather(
            *[execute_with_agent(task) for task in tasks],
            return_exceptions=True
        )
        
        # Собираем результаты
        results = {}
        for task in completed_tasks:
            if isinstance(task, Task):
                results[task.id] = {
                    "status": task.status,
                    "result": task.result,
                    "error": task.error
                }
            else:
                logger.error(f"Task execution error: {task}")
                
        return results
    
    def _get_agent_type_for_task(self, task: Task) -> AgentType:
        """Определяет тип агента для задачи"""
        task_type_mapping = {
            "research": AgentType.RESEARCH,
            "code_generation": AgentType.CODE_GENERATOR,
            "analysis": AgentType.ANALYZER,
            "testing": AgentType.TESTER
        }
        return task_type_mapping.get(task.type, AgentType.ORCHESTRATOR)

# Пример использования
async def main():
    system = ParallelAgentSystem("agent-config.json")
    
    request = "Создать REST API для управления задачами с аутентификацией"
    result = await system.process_request(request)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())