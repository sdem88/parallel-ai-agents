from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime
import os

app = FastAPI(title="Agent System API")

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модели данных
class TaskRequest(BaseModel):
    agent: str
    task: str
    priority: Optional[str] = "medium"

class BatchTaskRequest(BaseModel):
    tasks: List[TaskRequest]
    execution_mode: Optional[str] = "parallel"  # parallel или sequential

class Agent:
    def __init__(self, agent_id: str, name: str, skills: List[str]):
        self.id = agent_id
        self.name = name
        self.skills = skills
        self.status = "idle"
        self.current_task = None
        
    async def execute(self, task: str) -> Dict[str, Any]:
        self.status = "working"
        self.current_task = task
        
        # Симуляция работы
        await asyncio.sleep(2)
        
        result = {
            "task_id": str(uuid.uuid4()),
            "agent": self.id,
            "task": task,
            "status": "completed",
            "result": f"✅ {self.name} завершил: {task}",
            "timestamp": datetime.now().isoformat()
        }
        
        self.status = "idle"
        self.current_task = None
        
        return result

# Система агентов
class AgentSystem:
    def __init__(self):
        self.agents = {
            "researcher": Agent("researcher", "Исследователь 🔍", ["поиск", "анализ", "документация"]),
            "coder": Agent("coder", "Программист 💻", ["код", "рефакторинг", "оптимизация"]),
            "tester": Agent("tester", "Тестировщик 🧪", ["тесты", "проверка", "отладка"]),
            "reviewer": Agent("reviewer", "Ревьюер 📋", ["код-ревью", "стандарты", "безопасность"]),
            "architect": Agent("architect", "Архитектор 🏗️", ["дизайн", "паттерны", "масштабирование"]),
            "devops": Agent("devops", "DevOps 🚀", ["деплой", "CI/CD", "мониторинг"])
        }
        self.task_history = []
        self.active_tasks = {}
        
    async def execute_task(self, agent_id: str, task: str) -> Dict[str, Any]:
        if agent_id not in self.agents:
            raise ValueError(f"Agent {agent_id} not found")
            
        agent = self.agents[agent_id]
        result = await agent.execute(task)
        self.task_history.append(result)
        
        return result
        
    async def execute_batch(self, tasks: List[TaskRequest], mode: str = "parallel") -> List[Dict[str, Any]]:
        if mode == "parallel":
            coroutines = [
                self.execute_task(task.agent, task.task) 
                for task in tasks
            ]
            results = await asyncio.gather(*coroutines)
        else:
            results = []
            for task in tasks:
                result = await self.execute_task(task.agent, task.task)
                results.append(result)
                
        return results

# Глобальный экземпляр системы
agent_system = AgentSystem()

# WebSocket для real-time обновлений
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Обработка команд через WebSocket
            command = json.loads(data)
            
            if command["type"] == "execute_task":
                result = await agent_system.execute_task(
                    command["agent"], 
                    command["task"]
                )
                
                # Отправляем результат всем подключенным клиентам
                for connection in active_connections:
                    await connection.send_json(result)
                    
    except Exception as e:
        active_connections.remove(websocket)

# REST API endpoints
@app.get("/")
async def root():
    return {
        "message": "Agent System API",
        "version": "1.0",
        "endpoints": {
            "/agents": "Get all agents",
            "/execute": "Execute single task",
            "/batch": "Execute batch tasks",
            "/history": "Get task history",
            "/status": "System status"
        }
    }

@app.get("/agents")
async def get_agents():
    agents_info = []
    for agent_id, agent in agent_system.agents.items():
        agents_info.append({
            "id": agent.id,
            "name": agent.name,
            "skills": agent.skills,
            "status": agent.status,
            "current_task": agent.current_task
        })
    return {"agents": agents_info}

@app.post("/execute")
async def execute_task(task_request: TaskRequest):
    try:
        result = await agent_system.execute_task(
            task_request.agent,
            task_request.task
        )
        return result
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.post("/batch")
async def execute_batch(batch_request: BatchTaskRequest):
    results = await agent_system.execute_batch(
        batch_request.tasks,
        batch_request.execution_mode
    )
    return {
        "execution_mode": batch_request.execution_mode,
        "total_tasks": len(batch_request.tasks),
        "results": results
    }

@app.get("/history")
async def get_history(limit: int = 50):
    history = agent_system.task_history[-limit:]
    return {
        "total": len(agent_system.task_history),
        "showing": len(history),
        "history": history
    }

@app.get("/status")
async def get_status():
    active_agents = sum(
        1 for agent in agent_system.agents.values() 
        if agent.status == "working"
    )
    
    return {
        "total_agents": len(agent_system.agents),
        "active_agents": active_agents,
        "total_tasks_completed": len(agent_system.task_history),
        "websocket_connections": len(active_connections)
    }

@app.get("/templates")
async def get_templates():
    """Готовые шаблоны задач"""
    return {
        "templates": [
            {
                "name": "Создать REST API",
                "tasks": [
                    {"agent": "architect", "task": "Спроектировать структуру API"},
                    {"agent": "coder", "task": "Реализовать endpoints"},
                    {"agent": "tester", "task": "Написать тесты API"},
                    {"agent": "reviewer", "task": "Проверить безопасность"}
                ]
            },
            {
                "name": "Исправить баг",
                "tasks": [
                    {"agent": "researcher", "task": "Исследовать причину бага"},
                    {"agent": "coder", "task": "Исправить проблему"},
                    {"agent": "tester", "task": "Проверить исправление"},
                    {"agent": "reviewer", "task": "Валидировать решение"}
                ]
            },
            {
                "name": "Новая функция",
                "tasks": [
                    {"agent": "researcher", "task": "Изучить требования"},
                    {"agent": "architect", "task": "Спроектировать решение"},
                    {"agent": "coder", "task": "Реализовать функционал"},
                    {"agent": "tester", "task": "Покрыть тестами"},
                    {"agent": "devops", "task": "Подготовить к деплою"}
                ]
            }
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)