from fastapi import FastAPI, WebSocket, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import uuid
import sqlite3
import os
from datetime import datetime
import anthropic
import openai
from pathlib import Path

app = FastAPI(title="Universal AI Agent System", version="2.0")

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# База данных SQLite для персистентности
DB_PATH = "agents_system.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Таблица задач
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            task_description TEXT,
            status TEXT,
            result TEXT,
            error TEXT,
            created_at TIMESTAMP,
            completed_at TIMESTAMP,
            priority INTEGER,
            user_id TEXT,
            workflow_id TEXT
        )
    ''')
    
    # Таблица воркфлоу
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            config TEXT,
            created_at TIMESTAMP,
            user_id TEXT
        )
    ''')
    
    # Таблица агентов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS custom_agents (
            id TEXT PRIMARY KEY,
            name TEXT,
            description TEXT,
            prompt TEXT,
            skills TEXT,
            ai_provider TEXT,
            created_at TIMESTAMP,
            user_id TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Модели данных
class TaskRequest(BaseModel):
    agent: str
    task: str
    priority: Optional[int] = 5
    user_id: Optional[str] = "default"
    ai_provider: Optional[str] = "simulation"  # simulation, openai, anthropic

class WorkflowRequest(BaseModel):
    name: str
    tasks: List[TaskRequest]
    execution_mode: Optional[str] = "parallel"
    user_id: Optional[str] = "default"

class AgentConfig(BaseModel):
    name: str
    description: str
    prompt: str
    skills: List[str]
    ai_provider: str = "simulation"
    user_id: Optional[str] = "default"

class UniversalAgent:
    def __init__(self, agent_id: str, name: str, skills: List[str], 
                 prompt: str = "", ai_provider: str = "simulation"):
        self.id = agent_id
        self.name = name
        self.skills = skills
        self.prompt = prompt
        self.ai_provider = ai_provider
        self.status = "idle"
        self.current_task = None
        
    async def execute(self, task: str, ai_provider: str = None) -> Dict[str, Any]:
        provider = ai_provider or self.ai_provider
        self.status = "working"
        self.current_task = task
        
        try:
            if provider == "simulation":
                result = await self._simulate_work(task)
            elif provider == "openai":
                result = await self._execute_openai(task)
            elif provider == "anthropic":
                result = await self._execute_anthropic(task)
            else:
                result = f"✅ {self.name}: Обработал задачу - {task}"
                
            task_result = {
                "task_id": str(uuid.uuid4()),
                "agent": self.id,
                "task": task,
                "status": "completed",
                "result": result,
                "timestamp": datetime.now().isoformat(),
                "ai_provider": provider
            }
            
        except Exception as e:
            task_result = {
                "task_id": str(uuid.uuid4()),
                "agent": self.id,
                "task": task,
                "status": "failed",
                "result": None,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "ai_provider": provider
            }
        
        self.status = "idle"
        self.current_task = None
        return task_result
    
    async def _simulate_work(self, task: str) -> str:
        # Симуляция работы с разным временем в зависимости от сложности
        complexity = min(len(task) // 20, 5)
        await asyncio.sleep(1 + complexity * 0.5)
        return f"✅ {self.name} завершил: {task}"
    
    async def _execute_openai(self, task: str) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "⚠️ OpenAI API ключ не настроен"
            
        client = openai.OpenAI(api_key=api_key)
        
        system_prompt = f"""Ты {self.name}. {self.prompt}
        Твои навыки: {', '.join(self.skills)}
        
        Выполни задачу качественно и подробно."""
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    async def _execute_anthropic(self, task: str) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return "⚠️ Anthropic API ключ не настроен"
            
        client = anthropic.Anthropic(api_key=api_key)
        
        system_prompt = f"""Ты {self.name}. {self.prompt}
        Твои навыки: {', '.join(self.skills)}
        
        Выполни задачу качественно и подробно."""
        
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1000,
            temperature=0.7,
            system=system_prompt,
            messages=[
                {"role": "user", "content": task}
            ]
        )
        
        return message.content[0].text

# Универсальная система агентов
class UniversalAgentSystem:
    def __init__(self):
        self.agents = self._load_default_agents()
        self.load_custom_agents()
        self.active_workflows = {}
        
    def _load_default_agents(self) -> Dict[str, UniversalAgent]:
        return {
            "researcher": UniversalAgent(
                "researcher", 
                "Исследователь 🔍", 
                ["поиск", "анализ", "документация"],
                "Ты опытный исследователь. Находишь информацию, анализируешь данные и составляешь отчеты."
            ),
            "coder": UniversalAgent(
                "coder", 
                "Программист 💻", 
                ["код", "рефакторинг", "оптимизация"],
                "Ты профессиональный программист. Пишешь чистый, эффективный код и решаешь технические задачи."
            ),
            "tester": UniversalAgent(
                "tester", 
                "Тестировщик 🧪", 
                ["тесты", "проверка", "отладка"],
                "Ты специалист по тестированию. Находишь баги и создаешь тесты для обеспечения качества."
            ),
            "reviewer": UniversalAgent(
                "reviewer", 
                "Ревьюер 📋", 
                ["код-ревью", "стандарты", "безопасность"],
                "Ты эксперт по качеству кода. Проверяешь соблюдение стандартов и безопасность."
            ),
            "architect": UniversalAgent(
                "architect", 
                "Архитектор 🏗️", 
                ["дизайн", "паттерны", "масштабирование"],
                "Ты системный архитектор. Проектируешь надежные и масштабируемые решения."
            ),
            "devops": UniversalAgent(
                "devops", 
                "DevOps 🚀", 
                ["деплой", "CI/CD", "мониторинг"],
                "Ты DevOps инженер. Автоматизируешь процессы и обеспечиваешь стабильность."
            ),
            "analyst": UniversalAgent(
                "analyst", 
                "Аналитик 📊", 
                ["данные", "метрики", "отчеты"],
                "Ты бизнес-аналитик. Анализируешь данные и делаешь выводы для бизнеса."
            ),
            "designer": UniversalAgent(
                "designer", 
                "Дизайнер 🎨", 
                ["UI/UX", "прототипы", "визуализация"],
                "Ты UX/UI дизайнер. Создаешь удобные и красивые интерфейсы."
            )
        }
    
    def load_custom_agents(self):
        """Загружает пользовательских агентов из базы"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM custom_agents")
        for row in cursor.fetchall():
            agent_id, name, description, prompt, skills_str, ai_provider, created_at, user_id = row
            skills = json.loads(skills_str) if skills_str else []
            
            self.agents[agent_id] = UniversalAgent(
                agent_id, name, skills, prompt, ai_provider
            )
        
        conn.close()
    
    async def execute_task(self, task_request: TaskRequest) -> Dict[str, Any]:
        if task_request.agent not in self.agents:
            raise ValueError(f"Agent {task_request.agent} not found")
            
        agent = self.agents[task_request.agent]
        result = await agent.execute(task_request.task, task_request.ai_provider)
        
        # Сохранить в базу
        self._save_task_to_db(task_request, result)
        
        return result
    
    async def execute_workflow(self, workflow_request: WorkflowRequest) -> Dict[str, Any]:
        workflow_id = str(uuid.uuid4())
        self.active_workflows[workflow_id] = {
            "name": workflow_request.name,
            "status": "running",
            "tasks": [],
            "results": []
        }
        
        # Сохранить воркфлоу в базу
        self._save_workflow_to_db(workflow_request, workflow_id)
        
        if workflow_request.execution_mode == "parallel":
            tasks = [
                self.execute_task(task) for task in workflow_request.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for task in workflow_request.tasks:
                result = await self.execute_task(task)
                results.append(result)
        
        self.active_workflows[workflow_id]["status"] = "completed"
        self.active_workflows[workflow_id]["results"] = results
        
        return {
            "workflow_id": workflow_id,
            "name": workflow_request.name,
            "execution_mode": workflow_request.execution_mode,
            "total_tasks": len(workflow_request.tasks),
            "results": results
        }
    
    def create_custom_agent(self, config: AgentConfig) -> str:
        agent_id = f"custom_{uuid.uuid4().hex[:8]}"
        
        # Создать агента
        self.agents[agent_id] = UniversalAgent(
            agent_id, config.name, config.skills, 
            config.prompt, config.ai_provider
        )
        
        # Сохранить в базу
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO custom_agents 
            (id, name, description, prompt, skills, ai_provider, created_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            agent_id, config.name, config.description, config.prompt,
            json.dumps(config.skills), config.ai_provider,
            datetime.now().isoformat(), config.user_id
        ))
        
        conn.commit()
        conn.close()
        
        return agent_id
    
    def _save_task_to_db(self, task_request: TaskRequest, result: Dict[str, Any]):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO tasks 
            (id, agent_id, task_description, status, result, error, created_at, priority, user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result["task_id"], task_request.agent, task_request.task,
            result["status"], result.get("result"), result.get("error"),
            result["timestamp"], task_request.priority, task_request.user_id
        ))
        
        conn.commit()
        conn.close()
    
    def _save_workflow_to_db(self, workflow_request: WorkflowRequest, workflow_id: str):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        config = {
            "tasks": [task.dict() for task in workflow_request.tasks],
            "execution_mode": workflow_request.execution_mode
        }
        
        cursor.execute('''
            INSERT INTO workflows 
            (id, name, description, config, created_at, user_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            workflow_id, workflow_request.name, "",
            json.dumps(config), datetime.now().isoformat(), workflow_request.user_id
        ))
        
        conn.commit()
        conn.close()

# Глобальная система
agent_system = UniversalAgentSystem()

# WebSocket соединения
active_connections: List[WebSocket] = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command["type"] == "execute_task":
                task_request = TaskRequest(**command["data"])
                result = await agent_system.execute_task(task_request)
                
                # Отправить результат всем клиентам
                for connection in active_connections:
                    try:
                        await connection.send_json(result)
                    except:
                        pass
                        
    except Exception as e:
        if websocket in active_connections:
            active_connections.remove(websocket)

# API Endpoints
@app.get("/")
async def root():
    return FileResponse("/Users/sergeidemchuk/agents-system/frontend/enhanced_index.html")

@app.get("/api/agents")
async def get_agents():
    agents_info = []
    for agent_id, agent in agent_system.agents.items():
        agents_info.append({
            "id": agent.id,
            "name": agent.name,
            "skills": agent.skills,
            "status": agent.status,
            "current_task": agent.current_task,
            "ai_provider": agent.ai_provider,
            "is_custom": agent_id.startswith("custom_")
        })
    return {"agents": agents_info}

@app.post("/api/execute")
async def execute_task(task_request: TaskRequest):
    try:
        result = await agent_system.execute_task(task_request)
        return result
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

@app.post("/api/workflow")
async def execute_workflow(workflow_request: WorkflowRequest):
    result = await agent_system.execute_workflow(workflow_request)
    return result

@app.post("/api/agents/create")
async def create_agent(config: AgentConfig):
    agent_id = agent_system.create_custom_agent(config)
    return {"agent_id": agent_id, "message": "Агент создан успешно"}

@app.get("/api/history")
async def get_history(limit: int = 50, user_id: str = "default"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM tasks 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (user_id, limit))
    
    tasks = []
    for row in cursor.fetchall():
        task_id, agent_id, task_desc, status, result, error, created_at, completed_at, priority, user_id, workflow_id = row
        tasks.append({
            "task_id": task_id,
            "agent": agent_id,
            "task": task_desc,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": created_at,
            "priority": priority
        })
    
    conn.close()
    return {"history": tasks}

@app.get("/api/workflows")
async def get_workflows(user_id: str = "default"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM workflows 
        WHERE user_id = ? 
        ORDER BY created_at DESC
    ''', (user_id,))
    
    workflows = []
    for row in cursor.fetchall():
        workflow_id, name, description, config, created_at, user_id = row
        workflows.append({
            "id": workflow_id,
            "name": name,
            "description": description,
            "config": json.loads(config),
            "created_at": created_at
        })
    
    conn.close()
    return {"workflows": workflows}

@app.get("/api/templates")
async def get_templates():
    return {
        "templates": [
            {
                "name": "Создать веб-приложение",
                "description": "Полный цикл разработки веб-приложения",
                "tasks": [
                    {"agent": "analyst", "task": "Проанализировать требования и создать техническое задание"},
                    {"agent": "architect", "task": "Спроектировать архитектуру приложения"},
                    {"agent": "designer", "task": "Создать UI/UX дизайн и прототипы"},
                    {"agent": "coder", "task": "Реализовать frontend и backend"},
                    {"agent": "tester", "task": "Написать и выполнить тесты"},
                    {"agent": "reviewer", "task": "Проверить качество и безопасность кода"},
                    {"agent": "devops", "task": "Настроить CI/CD и деплой"}
                ]
            },
            {
                "name": "Исправить критический баг",
                "description": "Быстрое исправление критической ошибки",
                "tasks": [
                    {"agent": "analyst", "task": "Проанализировать логи и найти причину бага"},
                    {"agent": "coder", "task": "Исправить найденную проблему"},
                    {"agent": "tester", "task": "Протестировать исправление"},
                    {"agent": "reviewer", "task": "Проверить что исправление не сломало другой функционал"}
                ]
            },
            {
                "name": "Исследование рынка",
                "description": "Комплексное исследование рынка и конкурентов",
                "tasks": [
                    {"agent": "researcher", "task": "Собрать данные о рынке и трендах"},
                    {"agent": "analyst", "task": "Проанализировать конкурентов"},
                    {"agent": "researcher", "task": "Изучить потребности клиентов"},
                    {"agent": "analyst", "task": "Составить итоговый отчет с рекомендациями"}
                ]
            },
            {
                "name": "Оптимизация производительности",
                "description": "Анализ и улучшение производительности системы",
                "tasks": [
                    {"agent": "analyst", "task": "Провести анализ производительности"},
                    {"agent": "coder", "task": "Оптимизировать узкие места в коде"},
                    {"agent": "devops", "task": "Настроить мониторинг и кэширование"},
                    {"agent": "tester", "task": "Провести нагрузочное тестирование"}
                ]
            }
        ]
    }

@app.get("/api/status")
async def get_status():
    active_agents = sum(
        1 for agent in agent_system.agents.values() 
        if agent.status == "working"
    )
    
    # Статистика из базы
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM tasks")
    total_tasks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed'")
    completed_tasks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM workflows")
    total_workflows = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_agents": len(agent_system.agents),
        "active_agents": active_agents,
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "total_workflows": total_workflows,
        "websocket_connections": len(active_connections),
        "active_workflows": len(agent_system.active_workflows)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)