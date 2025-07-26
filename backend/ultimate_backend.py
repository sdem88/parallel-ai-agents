"""
🚀 Universal AI Agent System - Ultimate Backend
Топовая система с интеграцией всех лучших AI моделей
"""

import os
import asyncio
import sqlite3
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# AI Providers
import openai
import anthropic
import google.generativeai as genai
from openai import AsyncOpenAI

# Logging and monitoring
import logging
import structlog
from prometheus_client import Counter, Histogram, generate_latest

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.LoggerFactory(),
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
task_counter = Counter('agent_tasks_total', 'Total number of tasks', ['agent_type', 'status'])
task_duration = Histogram('agent_task_duration_seconds', 'Task execution time')

app = FastAPI(
    title="🤖 Universal AI Agent System",
    description="Топовая система ИИ агентов с интеграцией лучших моделей мира",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене ограничить конкретными доменами
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Database setup
DB_PATH = "ultimate_agents.db"

def init_database():
    """Инициализация базы данных с расширенной схемой"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Таблица пользователей
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            api_key TEXT UNIQUE,
            plan TEXT DEFAULT 'free',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP,
            settings TEXT DEFAULT '{}'
        )
    ''')
    
    # Таблица агентов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS agents (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            system_prompt TEXT,
            skills TEXT,
            ai_provider TEXT DEFAULT 'simulation',
            model_name TEXT,
            user_id TEXT,
            is_public BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            usage_count INTEGER DEFAULT 0,
            rating REAL DEFAULT 0.0
        )
    ''')
    
    # Таблица задач с расширенной информацией
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            agent_id TEXT,
            user_id TEXT,
            task_description TEXT,
            status TEXT DEFAULT 'pending',
            result TEXT,
            error TEXT,
            ai_provider TEXT,
            model_used TEXT,
            tokens_used INTEGER DEFAULT 0,
            execution_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            priority INTEGER DEFAULT 5,
            workflow_id TEXT,
            parent_task_id TEXT,
            metadata TEXT DEFAULT '{}'
        )
    ''')
    
    # Таблица воркфлоу
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS workflows (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            config TEXT,
            user_id TEXT,
            is_template BOOLEAN DEFAULT FALSE,
            is_public BOOLEAN DEFAULT FALSE,
            category TEXT,
            tags TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            usage_count INTEGER DEFAULT 0,
            rating REAL DEFAULT 0.0
        )
    ''')
    
    # Таблица API провайдеров
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ai_providers (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            provider_name TEXT,
            api_key_encrypted TEXT,
            model_preferences TEXT DEFAULT '{}',
            usage_stats TEXT DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used TIMESTAMP
        )
    ''')
    
    # Таблица шаблонов
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            category TEXT,
            difficulty TEXT DEFAULT 'beginner',
            steps TEXT,
            estimated_time INTEGER,
            tags TEXT,
            author_id TEXT,
            is_verified BOOLEAN DEFAULT FALSE,
            usage_count INTEGER DEFAULT 0,
            rating REAL DEFAULT 0.0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")

# Pydantic models
class AgentConfig(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: str = Field(..., min_length=10, max_length=2000)
    skills: List[str] = Field(default_factory=list)
    ai_provider: str = Field(default="simulation")
    model_name: Optional[str] = None
    is_public: bool = False

class TaskRequest(BaseModel):
    agent_id: str
    task_description: str = Field(..., min_length=1, max_length=1000)
    priority: int = Field(default=5, ge=1, le=10)
    ai_provider: Optional[str] = None
    model_override: Optional[str] = None
    workflow_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkflowRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    tasks: List[TaskRequest]
    execution_mode: str = Field(default="parallel", regex="^(parallel|sequential)$")
    is_template: bool = False
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class AIProviderConfig(BaseModel):
    provider_name: str
    api_key: str
    model_preferences: Dict[str, Any] = Field(default_factory=dict)

# AI Provider Manager
class AIProviderManager:
    def __init__(self):
        self.providers = {}
        self.models = {
            "openai": {
                "gpt-4o": {"context": 128000, "output": 4096, "cost_per_1k": 0.005},
                "gpt-4o-mini": {"context": 128000, "output": 16384, "cost_per_1k": 0.00015},
                "o3-mini": {"context": 128000, "output": 65536, "cost_per_1k": 0.002},  # Предполагаемые характеристики
                "gpt-4-turbo": {"context": 128000, "output": 4096, "cost_per_1k": 0.01}
            },
            "anthropic": {
                "claude-3-opus-20240229": {"context": 200000, "output": 4096, "cost_per_1k": 0.015},
                "claude-3-sonnet-20240229": {"context": 200000, "output": 4096, "cost_per_1k": 0.003},
                "claude-3-haiku-20240307": {"context": 200000, "output": 4096, "cost_per_1k": 0.00025}
            },
            "google": {
                "gemini-2.0-flash-exp": {"context": 1000000, "output": 8192, "cost_per_1k": 0.0015},
                "gemini-1.5-pro": {"context": 2000000, "output": 8192, "cost_per_1k": 0.00125},
                "gemini-1.5-flash": {"context": 1000000, "output": 8192, "cost_per_1k": 0.000075}
            },
            "openrouter": {
                "anthropic/claude-3-opus": {"context": 200000, "output": 4096, "cost_per_1k": 0.018},
                "openai/gpt-4-turbo": {"context": 128000, "output": 4096, "cost_per_1k": 0.012},
                "google/gemini-pro-1.5": {"context": 1000000, "output": 8192, "cost_per_1k": 0.0015}
            }
        }
        
    async def setup_provider(self, provider_name: str, api_key: str, user_id: str):
        """Настройка AI провайдера"""
        try:
            if provider_name == "openai":
                client = AsyncOpenAI(api_key=api_key)
                # Тест подключения
                await client.models.list()
                self.providers[f"{user_id}_{provider_name}"] = client
                
            elif provider_name == "anthropic":
                client = anthropic.AsyncAnthropic(api_key=api_key)
                # Тест подключения
                await client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                self.providers[f"{user_id}_{provider_name}"] = client
                
            elif provider_name == "google":
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-1.5-flash')
                # Тест подключения
                response = model.generate_content("Hi")
                self.providers[f"{user_id}_{provider_name}"] = model
                
            elif provider_name == "openrouter":
                client = AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                # Тест подключения
                await client.models.list()
                self.providers[f"{user_id}_{provider_name}"] = client
                
            # Сохранить в базу
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO ai_providers 
                (id, user_id, provider_name, api_key_encrypted, is_active, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                f"{user_id}_{provider_name}",
                user_id,
                provider_name,
                api_key,  # В продакшене должно быть зашифровано
                True,
                datetime.now().isoformat()
            ))
            conn.commit()
            conn.close()
            
            logger.info("AI provider configured", provider=provider_name, user_id=user_id)
            return True
            
        except Exception as e:
            logger.error("Failed to setup AI provider", 
                        provider=provider_name, user_id=user_id, error=str(e))
            raise HTTPException(status_code=400, detail=f"Failed to setup {provider_name}: {str(e)}")
    
    async def execute_with_ai(self, provider_name: str, model: str, system_prompt: str, 
                            user_prompt: str, user_id: str) -> Dict[str, Any]:
        """Выполнение запроса через AI провайдера"""
        provider_key = f"{user_id}_{provider_name}"
        
        if provider_key not in self.providers:
            raise HTTPException(status_code=400, detail=f"Provider {provider_name} not configured")
        
        start_time = time.time()
        
        try:
            if provider_name == "openai":
                client = self.providers[provider_key]
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                result = {
                    "content": response.choices[0].message.content,
                    "model": model,
                    "tokens_used": response.usage.total_tokens,
                    "cost": self._calculate_cost(provider_name, model, response.usage.total_tokens)
                }
                
            elif provider_name == "anthropic":
                client = self.providers[provider_key]
                response = await client.messages.create(
                    model=model,
                    max_tokens=4000,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                
                result = {
                    "content": response.content[0].text,
                    "model": model,
                    "tokens_used": response.usage.input_tokens + response.usage.output_tokens,
                    "cost": self._calculate_cost(provider_name, model, 
                                               response.usage.input_tokens + response.usage.output_tokens)
                }
                
            elif provider_name == "google":
                model_client = self.providers[provider_key]
                full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}"
                response = model_client.generate_content(full_prompt)
                
                result = {
                    "content": response.text,
                    "model": model,
                    "tokens_used": len(full_prompt.split()) * 1.3,  # Приблизительно
                    "cost": self._calculate_cost(provider_name, model, len(full_prompt.split()) * 1.3)
                }
                
            elif provider_name == "openrouter":
                client = self.providers[provider_key]
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                result = {
                    "content": response.choices[0].message.content,
                    "model": model,
                    "tokens_used": response.usage.total_tokens if response.usage else 0,
                    "cost": self._calculate_cost(provider_name, model, 
                                               response.usage.total_tokens if response.usage else 0)
                }
            
            execution_time = int((time.time() - start_time) * 1000)
            result["execution_time_ms"] = execution_time
            
            logger.info("AI task completed", 
                       provider=provider_name, model=model, 
                       tokens=result["tokens_used"], time_ms=execution_time)
            
            return result
            
        except Exception as e:
            logger.error("AI execution failed", 
                        provider=provider_name, model=model, error=str(e))
            raise HTTPException(status_code=500, detail=f"AI execution failed: {str(e)}")
    
    def _calculate_cost(self, provider: str, model: str, tokens: int) -> float:
        """Расчет стоимости запроса"""
        if provider in self.models and model in self.models[provider]:
            cost_per_1k = self.models[provider][model]["cost_per_1k"]
            return (tokens / 1000) * cost_per_1k
        return 0.0

# Global AI manager
ai_manager = AIProviderManager()

# Agent System
class UniversalAgent:
    def __init__(self, agent_data: Dict[str, Any]):
        self.id = agent_data["id"]
        self.name = agent_data["name"]
        self.description = agent_data.get("description", "")
        self.system_prompt = agent_data.get("system_prompt", "")
        self.skills = json.loads(agent_data.get("skills", "[]"))
        self.ai_provider = agent_data.get("ai_provider", "simulation")
        self.model_name = agent_data.get("model_name")
        self.user_id = agent_data.get("user_id")
        self.status = "idle"
        self.current_task = None
    
    async def execute_task(self, task: TaskRequest, user_id: str) -> Dict[str, Any]:
        """Выполнение задачи агентом"""
        self.status = "working"
        self.current_task = task.task_description
        
        start_time = time.time()
        task_id = str(uuid.uuid4())
        
        try:
            # Определение провайдера и модели
            provider = task.ai_provider or self.ai_provider
            model = task.model_override or self.model_name
            
            if provider == "simulation":
                # Симуляция работы
                await asyncio.sleep(2)
                result_content = f"✅ {self.name} выполнил задачу: {task.task_description}"
                tokens_used = 0
                cost = 0.0
            else:
                # Реальный AI запрос
                if not model:
                    # Выбор модели по умолчанию
                    default_models = {
                        "openai": "gpt-4o-mini",
                        "anthropic": "claude-3-haiku-20240307", 
                        "google": "gemini-1.5-flash",
                        "openrouter": "openai/gpt-4-turbo"
                    }
                    model = default_models.get(provider, "gpt-4o-mini")
                
                ai_result = await ai_manager.execute_with_ai(
                    provider, model, self.system_prompt, 
                    task.task_description, user_id
                )
                
                result_content = ai_result["content"]
                tokens_used = ai_result["tokens_used"]
                cost = ai_result["cost"]
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Сохранение результата в базу
            result = {
                "task_id": task_id,
                "agent_id": self.id,
                "agent_name": self.name,
                "task": task.task_description,
                "status": "completed",
                "result": result_content,
                "ai_provider": provider,
                "model_used": model,
                "tokens_used": tokens_used,
                "cost": cost,
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Сохранить в базу данных
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks 
                (id, agent_id, user_id, task_description, status, result, 
                 ai_provider, model_used, tokens_used, execution_time_ms, 
                 started_at, completed_at, priority, workflow_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id, self.id, user_id, task.task_description, "completed", result_content,
                provider, model, tokens_used, execution_time,
                start_time, time.time(), task.priority, task.workflow_id, 
                json.dumps(task.metadata)
            ))
            conn.commit()
            conn.close()
            
            # Метрики
            task_counter.labels(agent_type=self.id, status="completed").inc()
            task_duration.observe(execution_time / 1000)
            
            logger.info("Task completed successfully", 
                       task_id=task_id, agent_id=self.id, 
                       execution_time_ms=execution_time, tokens=tokens_used)
            
            return result
            
        except Exception as e:
            execution_time = int((time.time() - start_time) * 1000)
            
            error_result = {
                "task_id": task_id,
                "agent_id": self.id,
                "agent_name": self.name,
                "task": task.task_description,
                "status": "failed",
                "error": str(e),
                "execution_time_ms": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            # Сохранить ошибку в базу
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO tasks 
                (id, agent_id, user_id, task_description, status, error, 
                 execution_time_ms, started_at, priority, workflow_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task_id, self.id, user_id, task.task_description, "failed", str(e),
                execution_time, start_time, task.priority, task.workflow_id
            ))
            conn.commit()
            conn.close()
            
            task_counter.labels(agent_type=self.id, status="failed").inc()
            
            logger.error("Task failed", task_id=task_id, agent_id=self.id, error=str(e))
            
            return error_result
        
        finally:
            self.status = "idle"
            self.current_task = None

# Agent System Manager
class AgentSystemManager:
    def __init__(self):
        self.active_agents = {}
        self.load_agents()
    
    def load_agents(self):
        """Загрузка агентов из базы данных"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Загрузить предустановленных агентов если их нет
        cursor.execute("SELECT COUNT(*) FROM agents")
        if cursor.fetchone()[0] == 0:
            self._create_default_agents()
        
        # Загрузить всех агентов
        cursor.execute("SELECT * FROM agents")
        agents_data = cursor.fetchall()
        
        for agent_data in agents_data:
            agent_dict = {
                "id": agent_data[0],
                "name": agent_data[1], 
                "description": agent_data[2],
                "system_prompt": agent_data[3],
                "skills": agent_data[4],
                "ai_provider": agent_data[5],
                "model_name": agent_data[6],
                "user_id": agent_data[7]
            }
            self.active_agents[agent_data[0]] = UniversalAgent(agent_dict)
        
        conn.close()
        logger.info("Loaded agents", count=len(self.active_agents))
    
    def _create_default_agents(self):
        """Создание агентов по умолчанию"""
        default_agents = [
            {
                "id": "researcher",
                "name": "Исследователь 🔍",
                "description": "Поиск и анализ информации из различных источников",
                "system_prompt": "Ты опытный исследователь-аналитик. Твоя задача - находить релевантную информацию, анализировать данные и представлять структурированные выводы. Всегда указывай источники и степень достоверности информации.",
                "skills": '["поиск информации", "анализ данных", "составление отчетов", "fact-checking"]',
                "ai_provider": "openai",
                "model_name": "gpt-4o-mini"
            },
            {
                "id": "coder",
                "name": "Программист 💻", 
                "description": "Разработка и оптимизация программного кода",
                "system_prompt": "Ты профессиональный программист с экспертизой в современных языках и технологиях. Пишешь чистый, эффективный и хорошо документированный код. Следуешь лучшим практикам и принципам SOLID.",
                "skills": '["Python", "JavaScript", "TypeScript", "React", "FastAPI", "SQL", "Git"]',
                "ai_provider": "anthropic",
                "model_name": "claude-3-sonnet-20240229"
            },
            {
                "id": "designer",
                "name": "Дизайнер 🎨",
                "description": "UI/UX дизайн и создание пользовательского опыта",
                "system_prompt": "Ты креативный UX/UI дизайнер с глубоким пониманием принципов пользовательского опыта. Создаешь интуитивные, доступные и визуально привлекательные интерфейсы.",
                "skills": '["UI/UX дизайн", "прототипирование", "пользовательское тестирование", "визуальная иерархия"]',
                "ai_provider": "google",
                "model_name": "gemini-1.5-flash"
            },
            {
                "id": "analyst",
                "name": "Аналитик 📊",
                "description": "Бизнес-анализ и работа с данными",
                "system_prompt": "Ты бизнес-аналитик с экспертизой в анализе данных и бизнес-процессов. Умеешь выявлять паттерны, делать прогнозы и предлагать стратегические решения на основе данных.",
                "skills": '["анализ данных", "SQL", "Excel", "бизнес-моделирование", "KPI", "A/B тестирование"]',
                "ai_provider": "openai",
                "model_name": "gpt-4o"
            },
            {
                "id": "writer",
                "name": "Копирайтер ✍️",
                "description": "Создание текстового контента и маркетинговых материалов",
                "system_prompt": "Ты профессиональный копирайтер и контент-мейкер. Создаешь убедительные, вовлекающие тексты для различных аудиторий и платформ. Понимаешь принципы маркетинга и психологии покупателей.",
                "skills": '["копирайтинг", "контент-маркетинг", "SEO", "сторителлинг", "email-маркетинг"]',
                "ai_provider": "anthropic",
                "model_name": "claude-3-opus-20240229"
            },
            {
                "id": "tester",
                "name": "Тестировщик 🧪",
                "description": "Тестирование и обеспечение качества ПО",
                "system_prompt": "Ты опытный QA-инженер с экспертизой в различных видах тестирования. Находишь баги, создаешь тест-кейсы и обеспечиваешь высокое качество продуктов.",
                "skills": '["тестирование ПО", "автоматизация тестов", "регрессионное тестирование", "bug tracking"]',
                "ai_provider": "google",
                "model_name": "gemini-1.5-pro"
            }
        ]
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        for agent in default_agents:
            cursor.execute('''
                INSERT INTO agents 
                (id, name, description, system_prompt, skills, ai_provider, model_name, user_id, is_public)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                agent["id"], agent["name"], agent["description"], agent["system_prompt"],
                agent["skills"], agent["ai_provider"], agent["model_name"], "system", True
            ))
        
        conn.commit()
        conn.close()
        logger.info("Created default agents")

# Global agent manager
agent_manager = AgentSystemManager()

# WebSocket connections
active_connections: List[WebSocket] = []

# Initialize database
init_database()

# API Endpoints
@app.get("/")
async def root():
    """Главная страница"""
    return FileResponse("/Users/sergeidemchuk/agents-system/frontend/ultimate_ui.html")

@app.get("/api/health")
async def health_check():
    """Проверка состояния системы"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "agents_count": len(agent_manager.active_agents),
        "connections": len(active_connections)
    }

@app.get("/api/metrics")
async def get_metrics():
    """Prometheus метрики"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/api/agents")
async def get_agents(user_id: str = "demo"):
    """Получить список всех агентов"""
    agents_info = []
    for agent_id, agent in agent_manager.active_agents.items():
        agents_info.append({
            "id": agent.id,
            "name": agent.name,
            "description": agent.description,
            "skills": agent.skills,
            "ai_provider": agent.ai_provider,
            "model_name": agent.model_name,
            "status": agent.status,
            "current_task": agent.current_task,
            "is_custom": not agent.id in ["researcher", "coder", "designer", "analyst", "writer", "tester"]
        })
    
    return {"agents": agents_info}

@app.post("/api/agents")
async def create_agent(config: AgentConfig, user_id: str = "demo"):
    """Создать нового агента"""
    agent_id = f"agent_{uuid.uuid4().hex[:8]}"
    
    # Сохранить в базу данных
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO agents 
        (id, name, description, system_prompt, skills, ai_provider, model_name, user_id, is_public)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        agent_id, config.name, config.description, config.system_prompt,
        json.dumps(config.skills), config.ai_provider, config.model_name, 
        user_id, config.is_public
    ))
    conn.commit()
    conn.close()
    
    # Создать объект агента
    agent_data = {
        "id": agent_id,
        "name": config.name,
        "description": config.description,
        "system_prompt": config.system_prompt,
        "skills": json.dumps(config.skills),
        "ai_provider": config.ai_provider,
        "model_name": config.model_name,
        "user_id": user_id
    }
    
    agent_manager.active_agents[agent_id] = UniversalAgent(agent_data)
    
    logger.info("Agent created", agent_id=agent_id, name=config.name, user_id=user_id)
    
    return {"agent_id": agent_id, "message": "Agent created successfully"}

@app.post("/api/tasks")
async def execute_task(task: TaskRequest, user_id: str = "demo"):
    """Выполнить одиночную задачу"""
    if task.agent_id not in agent_manager.active_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent = agent_manager.active_agents[task.agent_id]
    result = await agent.execute_task(task, user_id)
    
    # Отправить обновление через WebSocket
    for connection in active_connections:
        try:
            await connection.send_json(result)
        except:
            pass
    
    return result

@app.post("/api/workflows")
async def execute_workflow(workflow: WorkflowRequest, user_id: str = "demo"):
    """Выполнить воркфлоу"""
    workflow_id = str(uuid.uuid4())
    
    # Сохранить воркфлоу в базу
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO workflows 
        (id, name, description, config, user_id, is_template, category, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        workflow_id, workflow.name, workflow.description,
        json.dumps(workflow.dict()), user_id, workflow.is_template,
        workflow.category, json.dumps(workflow.tags)
    ))
    conn.commit()
    conn.close()
    
    # Выполнить задачи
    results = []
    
    if workflow.execution_mode == "parallel":
        # Параллельное выполнение
        tasks = []
        for task in workflow.tasks:
            task.workflow_id = workflow_id
            if task.agent_id in agent_manager.active_agents:
                agent = agent_manager.active_agents[task.agent_id]
                tasks.append(agent.execute_task(task, user_id))
        
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
    else:
        # Последовательное выполнение
        for task in workflow.tasks:
            task.workflow_id = workflow_id
            if task.agent_id in agent_manager.active_agents:
                agent = agent_manager.active_agents[task.agent_id]
                result = await agent.execute_task(task, user_id)
                results.append(result)
                
                # Отправить промежуточный результат
                for connection in active_connections:
                    try:
                        await connection.send_json(result)
                    except:
                        pass
    
    logger.info("Workflow completed", workflow_id=workflow_id, 
                tasks_count=len(workflow.tasks), user_id=user_id)
    
    return {
        "workflow_id": workflow_id,
        "name": workflow.name,
        "execution_mode": workflow.execution_mode,
        "total_tasks": len(workflow.tasks),
        "results": results
    }

@app.post("/api/ai-providers")
async def setup_ai_provider(config: AIProviderConfig, user_id: str = "demo"):
    """Настроить AI провайдера"""
    success = await ai_manager.setup_provider(config.provider_name, config.api_key, user_id)
    
    if success:
        return {"message": f"{config.provider_name} configured successfully"}
    else:
        raise HTTPException(status_code=400, detail="Failed to configure provider")

@app.get("/api/templates")
async def get_templates():
    """Получить готовые шаблоны"""
    # Загрузить из базы данных
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM templates ORDER BY usage_count DESC, rating DESC")
    templates_data = cursor.fetchall()
    conn.close()
    
    if not templates_data:
        # Создать базовые шаблоны если их нет
        return {"templates": get_default_templates()}
    
    templates = []
    for template in templates_data:
        templates.append({
            "id": template[0],
            "name": template[1],
            "description": template[2],
            "category": template[3],
            "difficulty": template[4],
            "steps": json.loads(template[5]),
            "estimated_time": template[6],
            "tags": json.loads(template[7] or "[]"),
            "usage_count": template[9],
            "rating": template[10]
        })
    
    return {"templates": templates}

def get_default_templates():
    """Базовые шаблоны для начала работы"""
    return [
        {
            "id": "website-creation",
            "name": "Создание веб-сайта",
            "description": "Полный цикл создания современного веб-сайта",
            "category": "development",
            "difficulty": "intermediate",
            "estimated_time": 240,
            "steps": [
                {"agent_id": "analyst", "task": "Проанализировать требования к сайту и целевую аудиторию"},
                {"agent_id": "designer", "task": "Создать дизайн-макет и пользовательские сценарии"},
                {"agent_id": "coder", "task": "Разработать фронтенд с адаптивным дизайном"},
                {"agent_id": "coder", "task": "Создать бэкенд API и базу данных"},
                {"agent_id": "writer", "task": "Написать контент для всех страниц сайта"},
                {"agent_id": "tester", "task": "Протестировать функциональность и производительность"}
            ],
            "tags": ["веб-разработка", "фулл-стек", "дизайн"]
        },
        {
            "id": "market-analysis", 
            "name": "Анализ рынка",
            "description": "Комплексное исследование рынка и конкурентов",
            "category": "business",
            "difficulty": "beginner",
            "estimated_time": 120,
            "steps": [
                {"agent_id": "researcher", "task": "Собрать данные о размере рынка и трендах"},
                {"agent_id": "analyst", "task": "Проанализировать конкурентов и их стратегии"},
                {"agent_id": "researcher", "task": "Изучить потребности и боли целевой аудитории"},
                {"agent_id": "analyst", "task": "Составить SWOT-анализ и рекомендации"}
            ],
            "tags": ["исследование", "бизнес", "стратегия"]
        },
        {
            "id": "content-campaign",
            "name": "Контент-кампания",
            "description": "Создание комплексной контент-кампании",
            "category": "marketing", 
            "difficulty": "beginner",
            "estimated_time": 180,
            "steps": [
                {"agent_id": "analyst", "task": "Определить целевую аудиторию и KPI кампании"},
                {"agent_id": "writer", "task": "Создать контент-план на месяц"},
                {"agent_id": "writer", "task": "Написать посты для социальных сетей"},
                {"agent_id": "designer", "task": "Создать визуальные материалы"},
                {"agent_id": "analyst", "task": "Разработать метрики для оценки эффективности"}
            ],
            "tags": ["маркетинг", "контент", "SMM"]
        },
        {
            "id": "bug-fix",
            "name": "Исправление критического бага",
            "description": "Быстрое решение критических проблем в продукте",
            "category": "development",
            "difficulty": "advanced",
            "estimated_time": 60,
            "steps": [
                {"agent_id": "analyst", "task": "Проанализировать логи и воспроизвести проблему"},
                {"agent_id": "coder", "task": "Найти и исправить корневую причину бага"},
                {"agent_id": "tester", "task": "Проверить исправление и протестировать смежную функциональность"},
                {"agent_id": "analyst", "task": "Задокументировать решение и предложить профилактические меры"}
            ],
            "tags": ["баг", "отладка", "срочно"]
        }
    ]

@app.get("/api/history")
async def get_task_history(limit: int = 50, user_id: str = "demo"):
    """Получить историю задач"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT t.*, a.name as agent_name
        FROM tasks t
        LEFT JOIN agents a ON t.agent_id = a.id  
        WHERE t.user_id = ? OR t.user_id = 'system'
        ORDER BY t.created_at DESC
        LIMIT ?
    ''', (user_id, limit))
    
    tasks = []
    for row in cursor.fetchall():
        tasks.append({
            "task_id": row[0],
            "agent_id": row[1],
            "agent_name": row[-1] or "Unknown Agent",
            "task": row[3],
            "status": row[4],
            "result": row[5],
            "error": row[6],
            "ai_provider": row[7],
            "model_used": row[8],
            "tokens_used": row[9] or 0,
            "execution_time_ms": row[10] or 0,
            "created_at": row[11],
            "workflow_id": row[16]
        })
    
    conn.close()
    return {"history": tasks}

@app.get("/api/stats")
async def get_statistics(user_id: str = "demo"):
    """Получить статистику системы"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Общая статистика
    cursor.execute("SELECT COUNT(*) FROM tasks WHERE user_id = ? OR user_id = 'system'", (user_id,))
    total_tasks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM tasks WHERE status = 'completed' AND (user_id = ? OR user_id = 'system')", (user_id,))
    completed_tasks = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM workflows WHERE user_id = ?", (user_id,))
    total_workflows = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM agents WHERE user_id = ? OR user_id = 'system'", (user_id,))
    total_agents = cursor.fetchone()[0]
    
    # Статистика использования AI
    cursor.execute('''
        SELECT ai_provider, COUNT(*), SUM(tokens_used), AVG(execution_time_ms)
        FROM tasks 
        WHERE user_id = ? OR user_id = 'system'
        GROUP BY ai_provider
    ''', (user_id,))
    
    ai_stats = []
    for row in cursor.fetchall():
        ai_stats.append({
            "provider": row[0],
            "tasks_count": row[1],
            "total_tokens": row[2] or 0,
            "avg_time_ms": int(row[3]) if row[3] else 0
        })
    
    conn.close()
    
    success_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    return {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "total_workflows": total_workflows,
        "total_agents": total_agents,
        "success_rate": round(success_rate, 1),
        "ai_providers_stats": ai_stats,
        "active_connections": len(active_connections)
    }

# WebSocket endpoint
@app.websocket("/api/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Поддерживать соединение живым
            await asyncio.sleep(30)
            await websocket.send_json({"type": "ping", "timestamp": datetime.now().isoformat()})
    except Exception as e:
        logger.info("WebSocket connection closed", error=str(e))
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(
        "ultimate_backend:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=False,
        log_level="info"
    )