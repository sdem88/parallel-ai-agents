# 🚀 Деплой в облако

## Railway (Рекомендуется)

1. **Подготовка:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Деплой:**
   - Зайти на [railway.app](https://railway.app)
   - Подключить GitHub репозиторий
   - Railway автоматически развернет проект

3. **Настройки в Railway:**
   - Start Command: `cd backend && python static_server.py`
   - Environment: Python
   - Port: 8000

## Replit

1. **Создать Repl:**
   - Зайти на [replit.com](https://replit.com)
   - Create → Import from GitHub
   - Вставить ссылку на ваш репозиторий

2. **Запуск:**
   - Replit автоматически установит зависимости
   - Нажать "Run"

## GitHub Pages + Railway (Гибридный)

1. **Фронтенд на GitHub Pages:**
   ```bash
   # Создать отдельный репозиторий для фронтенда
   # Включить GitHub Pages в настройках
   ```

2. **Бэкенд на Railway:**
   - Развернуть только backend/
   - Изменить apiUrl в index.html на URL Railway

## Heroku (Альтернатива)

1. **Подготовка:**
   ```bash
   heroku create your-agent-system
   git push heroku main
   ```

2. **Настройки:**
   - Добавить Python buildpack
   - Установить PORT переменную

## Vercel (Только фронтенд)

1. **Настройка:**
   - Подключить GitHub репозиторий
   - Root Directory: `frontend`
   - Build Command: не требуется (статический HTML)

## 🔧 Локальный запуск

```bash
# Установить зависимости
pip install -r requirements.txt

# Запустить полное приложение
cd backend
python static_server.py

# Или только API
python app.py
```

## 🌍 URLs после деплоя

- **Railway:** `https://your-app.railway.app`
- **Replit:** `https://your-repl.your-username.repl.co`  
- **Heroku:** `https://your-app.herokuapp.com`

## ⚡ Быстрый деплой (1 клик)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

1. Нажать кнопку выше
2. Подключить GitHub
3. Выбрать этот репозиторий
4. Через 2-3 минуты приложение будет доступно!