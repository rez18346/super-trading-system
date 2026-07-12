#!/usr/bin/env bash
# traderctl.sh — Управление супер-трейдером v2
# Использование:
#   ./traderctl.sh start       — запустить FastAPI + трейдер
#   ./traderctl.sh stop        — остановить всё
#   ./traderctl.sh restart     — перезапустить
#   ./traderctl.sh status      — статус

set -euo pipefail
BASE_DIR="/home/ksysha/new_trader"

start() {
    echo "🚀 Запуск супер-трейдера..."
    
    # FastAPI
    screen -dmS fastapi bash -c "cd $BASE_DIR && python3 -B -u main.py"
    sleep 2
    
    # Трейдер
    screen -dmS trader bash -c "cd $BASE_DIR && python3 -B -u super_trader.py"
    sleep 3
    
    curl -s --max-time 3 http://127.0.0.1:8765/api/status > /dev/null 2>&1 && \
        echo "✅ FastAPI: http://localhost:8765" || \
        echo "❌ FastAPI не отвечает"
    
    pgrep -f super_trader.py > /dev/null && \
        echo "✅ Трейдер: PID $(pgrep -f super_trader.py | head -1)" || \
        echo "❌ Трейдер не запущен"
}

stop() {
    echo "🛑 Остановка..."
    for s in fastapi trader; do
        screen -S $s -X stuff $'\003' 2>/dev/null
        sleep 1
        screen -S $s -X quit 2>/dev/null
    done
    sleep 1
    echo "✅ Остановлено"
}

restart() {
    stop
    sleep 2
    start
}

status() {
    echo "=== СУПЕР-ТРЕЙДЕР v2 ==="
    echo ""
    
    # FastAPI
    if curl -s --max-time 3 http://127.0.0.1:8765/ > /dev/null 2>&1; then
        echo "✅ FastAPI: порт 8765"
        curl -s --max-time 3 http://127.0.0.1:8765/api/status 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(f'   Баланс: \${d.get(\"balance\",{}).get(\"total\",0):.2f}')
print(f'   Позиции: {d.get(\"positions_count\",0)}')
print(f'   Закрытых сделок: {d.get(\"closed_count\",0)}')
print(f'   BTC: {d.get(\"btc\",{}).get(\"regime\",\"?\")}')
" 2>/dev/null || echo "   (нет данных)"
    else
        echo "❌ FastAPI: не отвечает"
    fi
    
    # Трейдер
    trader_pid=$(pgrep -f super_trader.py 2>/dev/null || true)
    if [ -n "$trader_pid" ]; then
        echo "✅ Трейдер: PID $trader_pid"
    else
        echo "❌ Трейдер: не запущен"
    fi
    
    echo ""
    screen -ls 2>/dev/null || echo "(нет screen-сессий)"
}

case "${1:-status}" in
    start) start ;;
    stop) stop ;;
    restart) restart ;;
    status) status ;;
    logs) 
        if [ -f "$BASE_DIR/data/trader.log" ]; then
            tail -30 "$BASE_DIR/data/trader.log"
        else
            echo "Нет лога"
        fi
        ;;
    *)
        echo "Использование: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac
