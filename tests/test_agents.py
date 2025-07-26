#!/usr/bin/env python3
"""
Базовые тесты для системы агентов
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
import sys
import os

# Добавляем путь к модулям
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_agents import SimpleAgent, AgentTeam, create_task


class TestSimpleAgent:
    """Тесты для базового агента"""
    
    def test_agent_initialization(self):
        """Тест создания агента"""
        agent = SimpleAgent("TestAgent", ["skill1", "skill2"])
        assert agent.name == "TestAgent"
        assert agent.skills == ["skill1", "skill2"]
    
    @pytest.mark.asyncio
    async def test_agent_work_execution(self):
        """Тест выполнения задачи агентом"""
        agent = SimpleAgent("TestAgent", ["testing"])
        result = await agent.work("Test task")
        
        assert result["agent"] == "TestAgent"
        assert result["task"] == "Test task"
        assert result["status"] == "completed"
        assert "result" in result


class TestAgentTeam:
    """Тесты для команды агентов"""
    
    def test_team_initialization(self):
        """Тест создания команды"""
        team = AgentTeam()
        assert len(team.agents) == 4
        assert "researcher" in team.agents
        assert "coder" in team.agents
        assert "tester" in team.agents
        assert "reviewer" in team.agents
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Тест параллельного выполнения задач"""
        team = AgentTeam()
        tasks = [
            {"agent": "researcher", "task": "Research task"},
            {"agent": "coder", "task": "Coding task"}
        ]
        
        results = await team.execute_parallel(tasks)
        
        assert len(results) == 2
        assert all(result["status"] == "completed" for result in results)


class TestUtilityFunctions:
    """Тесты для вспомогательных функций"""
    
    def test_create_task(self):
        """Тест создания задачи"""
        task = create_task("researcher", "Test task")
        assert task["agent"] == "researcher"
        assert task["task"] == "Test task"
    
    @pytest.mark.asyncio
    async def test_run_tasks(self):
        """Тест запуска задач"""
        from simple_agents import run_tasks
        
        tasks = [
            create_task("researcher", "Test research"),
            create_task("coder", "Test coding")
        ]
        
        results = await run_tasks(tasks)
        assert len(results) == 2


if __name__ == "__main__":
    pytest.main([__file__])