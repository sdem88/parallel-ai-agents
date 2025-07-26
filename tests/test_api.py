#!/usr/bin/env python3
"""
Тесты для API системы агентов
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'backend'))

from app import app


class TestAgentAPI:
    """Тесты для API агентов"""
    
    def setup_method(self):
        """Настройка для каждого теста"""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Тест корневого endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "Agent System API" in data["message"]
    
    def test_get_agents(self):
        """Тест получения списка агентов"""
        response = self.client.get("/agents")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert len(data["agents"]) > 0
        
        # Проверим структуру агента
        agent = data["agents"][0]
        required_fields = ["id", "name", "skills", "status"]
        for field in required_fields:
            assert field in agent
    
    def test_execute_task(self):
        """Тест выполнения задачи"""
        task_data = {
            "agent": "researcher",
            "task": "Test research task",
            "priority": "high"
        }
        
        response = self.client.post("/execute", json=task_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["agent"] == "researcher"
        assert data["task"] == "Test research task"
        assert data["status"] == "completed"
    
    def test_execute_invalid_agent(self):
        """Тест выполнения задачи с несуществующим агентом"""
        task_data = {
            "agent": "nonexistent",
            "task": "Test task"
        }
        
        response = self.client.post("/execute", json=task_data)
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
    
    def test_batch_execution(self):
        """Тест пакетного выполнения задач"""
        batch_data = {
            "tasks": [
                {"agent": "researcher", "task": "Research task 1"},
                {"agent": "coder", "task": "Coding task 1"}
            ],
            "execution_mode": "parallel"
        }
        
        response = self.client.post("/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        
        assert data["execution_mode"] == "parallel"
        assert data["total_tasks"] == 2
        assert len(data["results"]) == 2
    
    def test_get_status(self):
        """Тест получения статуса системы"""
        response = self.client.get("/status")
        assert response.status_code == 200
        data = response.json()
        
        required_fields = [
            "total_agents", "active_agents", 
            "total_tasks_completed", "websocket_connections"
        ]
        for field in required_fields:
            assert field in data
    
    def test_get_templates(self):
        """Тест получения шаблонов"""
        response = self.client.get("/templates")
        assert response.status_code == 200
        data = response.json()
        
        assert "templates" in data
        assert len(data["templates"]) > 0
        
        # Проверим структуру шаблона
        template = data["templates"][0]
        assert "name" in template
        assert "tasks" in template
        assert len(template["tasks"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])