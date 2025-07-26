#!/usr/bin/env python3
"""
Простая система параллельных агентов
"""

import asyncio
from typing import List, Dict, Any
import json

class SimpleAgent:
    """Базовый агент для выполнения задач"""
    
    def __init__(self, name: str, skills: List[str]):
        self.name = name
        self.skills = skills
        
    async def work(self, task: str) -> Dict[str, Any]:
        """Выполняет задачу"""
        print(f"🤖 {self.name}: Работаю над '{task}'")
        
        # Имитация работы
        await asyncio.sleep(1)
        
        result = {
            "agent": self.name,
            "task": task,
            "status": "completed",
            "result": f"Выполнено: {task}"
        }
        
        print(f"✅ {self.name}: Завершил '{task}'")
        return result

class AgentTeam:
    """Команда агентов для параллельной работы"""
    
    def __init__(self):
        self.agents = {
            "researcher": SimpleAgent("Исследователь 🔍", ["поиск", "анализ", "документация"]),
            "coder": SimpleAgent("Программист 💻", ["код", "рефакторинг", "оптимизация"]),
            "tester": SimpleAgent("Тестировщик 🧪", ["тесты", "проверка", "отладка"]),
            "reviewer": SimpleAgent("Ревьюер 📋", ["проверка кода", "стандарты", "безопасность"])
        }
        
    async def execute_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Выполняет задачи параллельно"""
        print("\n🚀 Запускаю параллельное выполнение задач...\n")
        
        # Создаем корутины для всех задач
        coroutines = []
        for task_info in tasks:
            agent_name = task_info["agent"]
            task = task_info["task"]
            
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                coroutines.append(agent.work(task))
            else:
                print(f"⚠️  Агент '{agent_name}' не найден")
                
        # Выполняем все задачи параллельно
        results = await asyncio.gather(*coroutines)
        
        print("\n✨ Все задачи выполнены!\n")
        return results
    
    def show_agents(self):
        """Показывает доступных агентов"""
        print("\n👥 Доступные агенты:")
        for name, agent in self.agents.items():
            print(f"  - {name}: {agent.name}")
            print(f"    Навыки: {', '.join(agent.skills)}")

# Простые функции для использования
async def run_tasks(tasks: List[Dict[str, str]]):
    """Запускает задачи"""
    team = AgentTeam()
    results = await team.execute_parallel(tasks)
    
    print("\n📊 Результаты:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['agent']}:")
        print(f"   Задача: {result['task']}")
        print(f"   Статус: {result['status']}")
        print(f"   Результат: {result['result']}")
    
    return results

def create_task(agent: str, task: str) -> Dict[str, str]:
    """Создает задачу для агента"""
    return {"agent": agent, "task": task}

# Примеры использования
if __name__ == "__main__":
    # Пример 1: Простые задачи
    print("=== ПРИМЕР 1: Простые задачи ===")
    
    tasks = [
        create_task("researcher", "Найти лучшие практики REST API"),
        create_task("coder", "Написать базовый CRUD контроллер"),
        create_task("tester", "Создать unit тесты"),
        create_task("reviewer", "Проверить безопасность кода")
    ]
    
    asyncio.run(run_tasks(tasks))
    
    # Пример 2: Показать агентов
    print("\n=== ПРИМЕР 2: Информация об агентах ===")
    team = AgentTeam()
    team.show_agents()