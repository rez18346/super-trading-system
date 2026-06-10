#!/usr/bin/env bash
# traderctl.sh — Единое управление трейдером и терминалом
# Использование:
#   ./traderctl.sh start       — запустить трейдер + терминал
#   ./traderctl.sh stop        — остановить трейдер + терминал
#   ./traderctl.sh restart     — перезапустить
#   ./traderctl.sh status      — статус
#   ./traderctl.sh logs        — логи трейдера
#   ./traderctl.sh term-logs   — логи терминала

set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
TRADER_SCREEN="trader"
TERM_SCREEN="terminal"
TRADER_CMD="python3 -B -u main.py"
TERM_CMD="python3 -B -u terminal/terminal_server.py"
TRADER_LOG="$BASE_DIR/data/trader_stderr.log"
TERM_LOG="/tmp/terminal.log"

# Очистка кэша Python — обязательна при перезапусках
clear_cache() {
    find "$BASE_DIR" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
    find "$BASE_DIR" -name '*.pyc' -delete 2>/dev/null || true
}

# Убить все процессы трейдера (main.py, orchestrator.py, trader_entry.py)
kill_trader() {
    echo "🛑 Останавливаю трейдер..."
    
    # 1. screen-сессию
    if screen -list | grep -q "$TRADER_SCREEN"; then
        screen -S "$TRADER_SCREEN" -X stuff $'\003'  # Ctrl+C
        sleep 1
        screen -S "$TRADER_SCREEN" -X quit 2>/dev/null || true
    fi
    
    # 2. Все процессы по имени (на случай если screen не убил)
    for proc in "trader_entry.py" "orchestrator.py" "main.py"; do
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                kill -TERM "$pid" 2>/dev/null || true
            done
        fi
    done
    sleep 2
    
    # 3. Оставшиеся —  SIGKILL
    for proc in "trader_entry.py" "orchestrator.py" "main.py"; do
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            for pid in $pids; do
                kill -9 "$pid" 2>/dev/null || true
            done
        fi
    done
    
    # 4. Удалить PID-файлы
    rm -f "$BASE_DIR/.main_running" 2>/dev/null || true
    
    echo "✅ Трейдер остановлен"
}

kill_terminal() {
    # 1. screen-сессия
    if screen -list | grep -q "$TERM_SCREEN"; then
        screen -S "$TERM_SCREEN" -X stuff $'\003'
        sleep 0.5
        screen -S "$TERM_SCREEN" -X quit 2>/dev/null || true
    fi
    
    # 2. Процесс
    pids=$(pgrep -f "terminal_server.py" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        for pid in $pids; do
            kill -TERM "$pid" 2>/dev/null || true
        done
    fi
    sleep 1
    
    echo "✅ Терминал остановлен"
}

start_trader() {
    echo "🚀 Запускаю трейдер..."
    clear_cache
    
    # Проверить не запущен ли уже
    if pgrep -f "trader_entry.py" > /dev/null 2>&1; then
        echo "⚠️  Трейдер уже запущен (PID: $(pgrep -f trader_entry.py))"
        return 0
    fi
    
    cd "$BASE_DIR"
    screen -dmS "$TRADER_SCREEN" bash -c "
        cd '$BASE_DIR'
        python3 -B -u main.py \"$@\" >> '$TRADER_LOG' 2>&1
    "
    sleep 2
    
    if pgrep -f "trader_entry.py" > /dev/null 2>&1; then
        echo "✅ Трейдер запущен (PID: $(pgrep -f trader_entry.py | head -1))"
    else
        echo "❌ Трейдер не запустился!"
        echo "   Логи: $TRADER_LOG"
        tail -5 "$TRADER_LOG" 2>/dev/null || true
        return 1
    fi
}

start_terminal() {
    echo "🚀 Запускаю терминал..."
    
    # Проверить не запущен ли уже
    if pgrep -f "terminal_server.py" > /dev/null 2>&1; then
        echo "⚠️  Терминал уже запущен (PID: $(pgrep -f terminal_server.py | head -1))"
        return 0
    fi
    
    cd "$BASE_DIR"
    screen -dmS "$TERM_SCREEN" bash -c "
        cd '$BASE_DIR'
        $TERM_CMD
    "
    sleep 2
    
    # Проверка
    if curl -s -o /dev/null -w '%{http_code}' http://localhost:8765/ 2>/dev/null | grep -q 200; then
        echo "✅ Терминал запущен (порт 8765)"
    else
        echo "❌ Терминал не отвечает на порту 8765!"
        tail -5 "$TERM_LOG" 2>/dev/null || true
        return 1
    fi
}

case "${1:-status}" in
    start)
        kill_trader 2>/dev/null
        sleep 1
        start_trader
        start_terminal
        echo ""
        echo "📊 Дашборд: http://localhost:8765"
        ;;
    stop)
        kill_trader
        kill_terminal
        ;;
    restart)
        echo "🔄 Перезапуск..."
        "$0" stop
        sleep 2
        "$0" start
        ;;
    status)
        echo "=== СТАТУС СИСТЕМЫ ==="
        echo ""
        
        # Трейдер
        trader_pid=$(pgrep -f "trader_entry.py" 2>/dev/null || true)
        if [ -n "$trader_pid" ]; then
            echo "✅ Трейдер: PID $trader_pid"
            echo "   Время работы: $(ps -o etime= -p $(echo "$trader_pid" | head -1) | xargs)"
        else
            echo "❌ Трейдер: НЕ ЗАПУЩЕН"
        fi
        
        # Терминал
        term_pid=$(pgrep -f "terminal_server.py" 2>/dev/null || true)
        if [ -n "$term_pid" ] && curl -s -o /dev/null -w '' http://localhost:8765/ 2>/dev/null; then
            echo "✅ Терминал: PID $term_pid (порт 8765)"
        else
            echo "❌ Терминал: НЕ ОТВЕЧАЕТ"
        fi
        
        # screen-сессии
        echo ""
        echo "Screen сессии:"
        screen -list 2>/dev/null || echo "   (нет screen-сессий)"
        
        # Баланс
        echo ""
        if [ -n "$trader_pid" ]; then
            echo "💰 Капитал: $(grep -oP 'Капитал:.*?\$[\d.]+' /home/ksysha/.openclaw/industrial_super_system/data/trader_stderr.log 2>/dev/null | tail -1 || echo 'неизвестно')"
        fi
        ;;
    logs)
        tail -50 "$TRADER_LOG"
        ;;
    term-logs)
        cat "$TERM_LOG"
        ;;
    *)
        echo "Использование: $0 {start|stop|restart|status|logs|term-logs}"
        exit 1
        ;;
esac
