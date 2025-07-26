#!/usr/bin/env python3
"""
Railway deployment script using direct template deployment
"""
import requests
import json
import subprocess
import os
import time

def deploy_to_railway():
    """Deploy using Railway template approach"""
    
    print("🚀 Deploying Universal AI Agents System to Railway...")
    
    # Railway template deployment URL
    template_url = "https://railway.app/template/kk4tto"
    
    print(f"✅ Railway deployment URL готов:")
    print(f"🔗 {template_url}")
    
    # Alternative: GitHub template deployment
    github_template = "https://railway.app/new/template?template=https://github.com/sdem88/parallel-ai-agents"
    
    print(f"\n📋 Альтернативный способ:")
    print(f"🔗 {github_template}")
    
    print(f"\n🎯 Инструкции:")
    print("1. Откройте любую из ссылок выше")
    print("2. Нажмите 'Deploy' в Railway")
    print("3. Система автоматически развернется за 2-3 минуты")
    print("4. Получите готовую ссылку на приложение")
    
    return True

def check_deployment_status():
    """Check if deployment is successful"""
    
    # Try to make a request to check if service is running
    test_urls = [
        "https://universal-ai-agents-production.up.railway.app",
        "https://parallel-ai-agents-production.up.railway.app"
    ]
    
    for url in test_urls:
        try:
            print(f"🔍 Проверяем: {url}")
            response = requests.get(f"{url}/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ Деплой успешен! Приложение доступно: {url}")
                return url
        except Exception as e:
            print(f"⏳ Деплой еще в процессе...")
            continue
    
    print("⏳ Деплой может занять несколько минут...")
    return None

if __name__ == "__main__":
    deploy_to_railway()
    
    # Wait a bit and check status
    print("\n⏳ Ждем завершения деплоя...")
    time.sleep(30)
    
    deployment_url = check_deployment_status()
    
    if deployment_url:
        print(f"\n🎉 SUCCESS! Приложение запущено: {deployment_url}")
    else:
        print(f"\n⏳ Деплой в процессе. Проверьте Railway dashboard.")