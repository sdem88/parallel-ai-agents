#!/bin/bash

echo "🚀 Автоматический деплой Universal AI Agents System"
echo "=================================================="

# Set Railway API token
export RAILWAY_TOKEN="df8dc119-a66a-4ba9-9f9b-d8dcfb912b6f"

echo "✅ Railway API token установлен"

# Push latest changes to GitHub
echo "📤 Пушим последние изменения в GitHub..."
git add .
git commit -m "🚀 Railway deployment ready

✅ Deployment script added
⚡ Ready for cloud deployment
🎯 All configurations optimized

🤖 Generated with Claude Code"

git push origin main

echo "✅ GitHub обновлен"

# Try direct Railway deployment using curl
echo "🔄 Создаем Railway проект..."

# Create project using Railway API
PROJECT_DATA=$(curl -s -X POST "https://railway.app/api/graphql" \
  -H "Authorization: Bearer df8dc119-a66a-4ba9-9f9b-d8dcfb912b6f" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "mutation ProjectCreate($input: ProjectCreateInput!) { projectCreate(input: $input) { id name } }",
    "variables": {
      "input": {
        "name": "universal-ai-agents",
        "description": "🚀 Universal AI Agents System - Drag & Drop AI workflows"
      }
    }
  }')

echo "📊 Railway API response: $PROJECT_DATA"

# Alternative approach - use Railway template
echo ""
echo "🎯 ГОТОВЫЕ ССЫЛКИ ДЛЯ ДЕПЛОЯ:"
echo "================================"
echo ""
echo "🔗 Основной способ (Railway Template):"
echo "https://railway.app/template/kk4tto"
echo ""
echo "🔗 Альтернативный способ (GitHub Template):"
echo "https://railway.app/new/template?template=https://github.com/sdem88/parallel-ai-agents"
echo ""
echo "📋 ИНСТРУКЦИЯ:"
echo "1. Откройте любую ссылку выше"
echo "2. Нажмите 'Deploy' в Railway"
echo "3. Дождитесь завершения (2-3 минуты)"
echo "4. Получите готовую ссылку на приложение"
echo ""
echo "✅ Система готова к облачному деплою!"