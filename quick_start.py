#!/usr/bin/env python3
"""
🚀 БЫСТРЫЙ СТАРТ - Система Агентов
"""

import asyncio
from simple_agents import run_tasks, create_task, AgentTeam

# 🎯 СУПЕР ПРОСТОЕ ИСПОЛЬЗОВАНИЕ

async def example_1():
    """Пример 1: Создание веб-приложения"""
    print("\n🌟 ЗАДАЧА: Создать веб-приложение\n")
    
    tasks = [
        create_task("researcher", "Изучить требования и выбрать технологии"),
        create_task("coder", "Создать базовую структуру приложения"),
        create_task("tester", "Настроить тестовое окружение"),
        create_task("reviewer", "Проверить архитектуру")
    ]
    
    await run_tasks(tasks)

async def example_2():
    """Пример 2: Исправление бага"""
    print("\n🐛 ЗАДАЧА: Исправить баг в системе\n")
    
    tasks = [
        create_task("researcher", "Найти причину бага в логах"),
        create_task("coder", "Исправить найденную проблему"),
        create_task("tester", "Проверить что баг исправлен"),
        create_task("reviewer", "Убедиться что fix не сломал другое")
    ]
    
    await run_tasks(tasks)

async def example_3():
    """Пример 3: Добавление новой функции"""
    print("\n✨ ЗАДАЧА: Добавить новую функцию\n")
    
    tasks = [
        create_task("researcher", "Изучить API документацию"),
        create_task("coder", "Реализовать интеграцию"),
        create_task("tester", "Написать интеграционные тесты"),
        create_task("reviewer", "Проверить обработку ошибок")
    ]
    
    await run_tasks(tasks)

# 🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ
async def interactive_mode():
    """Интерактивный режим работы с агентами"""
    team = AgentTeam()
    
    print("\n🎮 ИНТЕРАКТИВНЫЙ РЕЖИМ АГЕНТОВ")
    print("=" * 40)
    
    while True:
        print("\nВыберите действие:")
        print("1. Показать агентов")
        print("2. Запустить быстрые задачи")
        print("3. Создать свои задачи")
        print("4. Выход")
        
        choice = input("\nВаш выбор (1-4): ")
        
        if choice == "1":
            team.show_agents()
            
        elif choice == "2":
            print("\nБыстрые сценарии:")
            print("a. Создать веб-приложение")
            print("b. Исправить баг")
            print("c. Добавить функцию")
            
            scenario = input("\nВыберите сценарий (a-c): ")
            
            if scenario == "a":
                await example_1()
            elif scenario == "b":
                await example_2()
            elif scenario == "c":
                await example_3()
                
        elif choice == "3":
            tasks = []
            print("\nСоздание задач (введите 'готово' для запуска)")
            
            while True:
                agent = input("\nАгент (researcher/coder/tester/reviewer): ")
                if agent == "готово":
                    break
                    
                task = input("Задача: ")
                tasks.append(create_task(agent, task))
                
            if tasks:
                await run_tasks(tasks)
                
        elif choice == "4":
            print("\n👋 До свидания!")
            break

# 🏃 ЗАПУСК
if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════╗
║     🤖 СИСТЕМА ПАРАЛЛЕЛЬНЫХ АГЕНТОВ   ║
╚═══════════════════════════════════════╝

Простая и эффективная система для параллельного
выполнения задач с помощью специализированных агентов.

ИСПОЛЬЗОВАНИЕ:
1. python quick_start.py - интерактивный режим
2. Импортируйте в свой код:
   from simple_agents import run_tasks, create_task
""")
    
    # Запускаем интерактивный режим
    asyncio.run(interactive_mode())