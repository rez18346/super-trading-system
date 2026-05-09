#!/usr/bin/env bash
# ============================================================
# start.sh — Запуск Super Trading System
# ============================================================
# Использование:
#   chmod +x start.sh
#   ./start.sh
#
# Работает на: Linux, macOS, WSL2 (Windows)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "🏭 Super Trading System — запуск..."
echo ""

# 1. Проверка Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.10+"
    exit 1
fi

PY_VER=$(python3 --version 2>&1)
echo "✅ Python: $PY_VER"

# 2. Проверка config.py
if [ ! -f "config.py" ]; then
    echo "⚠️  config.py не найден!"
    echo "   Создайте из шаблона: cp config.example.py config.py"
    echo "   И укажите API-ключи Bybit"
    exit 1
fi

# 3. Проверка зависимостей
echo "📦 Проверка зависимостей..."
pip install -q -r requirements.txt 2>/dev/null || {
    echo "⚠️  Устанавливаю зависимости..."
    pip install -r requirements.txt
}
echo "✅ Зависимости установлены"

# 4. Проверка, не запущен ли уже
if [ -f "data/trader.pid" ]; then
    OLD_PID=$(cat data/trader.pid 2>/dev/null)
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "⚠️  Трейдер уже запущен (PID: $OLD_PID)"
        echo "   Перезапуск: ./start.sh --restart"
        exit 0
    else
        echo "🧹 Очистка старого PID-файла..."
        rm -f data/trader.pid
    fi
fi

# 5. Запуск
echo "🚀 Запуск трейдера..."
nohup python3 main.py > /tmp/trader_output.log 2>&1 &
PID=$!
echo $PID > data/trader.pid

echo "✅ Трейдер запущен, PID: $PID"
echo "📝 Логи: tail -f /tmp/system_v4.log"
echo "📊 Дашборд: http://localhost:8765"

# Для WSL: показать IP если нужно
if grep -qi microsoft /proc/version 2>/dev/null; then
    WSL_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "🌐 WSL2: дашборд доступен на http://${WSL_IP}:8765 (из Windows)"
fi
