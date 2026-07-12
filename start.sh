#!/usr/bin/env bash
set -e

# ════════════════════════════════════════
# Super Trading System — Quick Start
# ════════════════════════════════════════
# Запускает всё: установку зависимостей,
# проверку конфига и старт трейдера.
# ════════════════════════════════════════

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo "🏭 Super Trading System — Quick Start"
echo ""

# 1. Проверка Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 не найден. Установите: sudo apt install python3 python3-pip"
    exit 1
fi

# 2. Виртуальное окружение
if [ ! -d "venv" ]; then
    echo "📦 Создаю виртуальное окружение..."
    python3 -m venv venv
fi
source venv/bin/activate

# 3. Зависимости
echo "📥 Устанавливаю зависимости..."
pip install -q -r requirements.txt

# 4. Конфиг
if [ ! -f "config.json" ]; then
    if [ -f "config.json.example" ]; then
        echo "⚠️  config.json не найден. Создаю из шаблона..."
        cp config.json.example config.json
        echo "⚠️  ОТРЕДАКТИРУЙТЕ config.json — вставьте свои API-ключи!"
        echo "   nano config.json"
        exit 1
    else
        echo "❌ config.json.example не найден!"
        exit 1
    fi
fi

if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo "⚠️  .env не найден. Создаю из шаблона..."
        cp .env.example .env
        echo "⚠️  ОТРЕДАКТИРУЙТЕ .env — вставьте свои API-ключи!"
    fi
fi

# 5. Запуск
echo "🚀 Запускаю трейдер..."
exec python3 -B -u orchestrator.py