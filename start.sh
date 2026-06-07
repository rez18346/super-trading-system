#!/bin/bash
# start.sh — Единый запуск торговой системы
# Использование: ./start.sh [start|stop|restart|status]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$SCRIPT_DIR/data/trader.pid"
LOGFILE="/tmp/system_v5.log"
VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
MAIN="$SCRIPT_DIR/main.py"
PORT=8765

start() {
    echo "▶️ Запуск торговой системы..."

    # 1. Убедиться что PostgreSQL работает
    if ! pg_isready -h /tmp -q 2>/dev/null; then
        echo "   ⚠️ PostgreSQL не запущен, запускаю..."
        /home/ksysha/.local/pgsql/bin/pg_ctl -D /home/ksysha/pgdata -l /home/ksysha/pgdata/logfile -o "-k /tmp" start
        sleep 3
        if ! pg_isready -h /tmp -q 2>/dev/null; then
            echo "   ❌ PostgreSQL не запустился! Лог:"
            tail -5 /home/ksysha/pgdata/logfile
            exit 1
        fi
    fi
    echo "   ✅ PostgreSQL: $(pg_isready -h /tmp -q && echo 'OK')"

    # 2. Убить старые процессы на порту трейдера
    local old_pids
    old_pids=$(lsof -ti :$PORT 2>/dev/null)
    if [ -n "$old_pids" ]; then
        echo "   ⚠️ Освобождаю порт $PORT (PID: $old_pids)"
        kill -9 $old_pids 2>/dev/null
        sleep 1
    fi

    # 3. Удалить старый PID-файл
    [ -f "$PIDFILE" ] && rm -f "$PIDFILE" && echo "   🧹 Удалён старый PID-файл"

    # 4. Удалить старый real_balance
    [ -f /tmp/real_balance.json ] && rm -f /tmp/real_balance.json

    # 5. Запустить трейдер
    nohup "$VENV_PYTHON" "$MAIN" >> "$LOGFILE" 2>&1 &
    local pid=$!
    echo "   🚀 PID: $pid"

    # 6. Подождать и проверить
    sleep 5
    if kill -0 $pid 2>/dev/null; then
        echo "   ✅ Трейдер запущен (PID $pid)"
        echo "   📊 Дашборд: http://localhost:$PORT"
        echo "   📝 Лог: $LOGFILE"
    else
        echo "   ❌ Трейдер упал! Лог:"
        tail -5 "$LOGFILE"
        exit 1
    fi
}

stop() {
    echo "⏹️ Остановка..."

    local pids
    pids=$(pgrep -f "$MAIN" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "   Убиваю PID: $pids"
        kill $pids 2>/dev/null
        sleep 3
        # force kill если не умерли
        pids=$(pgrep -f "$MAIN" 2>/dev/null)
        [ -n "$pids" ] && kill -9 $pids 2>/dev/null && echo "   force kill"
    fi

    # освободить порт
    local port_pids
    port_pids=$(lsof -ti :$PORT 2>/dev/null)
    [ -n "$port_pids" ] && kill -9 $port_pids 2>/dev/null

    [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
    echo "   ✅ Остановлено"
}

status() {
    local pids
    pids=$(pgrep -f "$MAIN" 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "✅ Трейдер работает: PID=$pids"
        if command -v curl &>/dev/null; then
            curl -s --connect-timeout 3 http://localhost:$PORT/ 2>/dev/null | grep -oP '(stCapital|stPositions|stBtcRegime)[^>]*>\K[^<]+' | paste -sd ',' || echo "⚠️ Дашборд не отвечает"
        fi
    else
        echo "❌ Трейдер не работает"
    fi
}

case "${1:-status}" in
    start)   start ;;
    stop)    stop ;;
    restart) stop; sleep 2; start ;;
    status)  status ;;
    *)
        echo "Использование: $0 {start|stop|restart|status}"
        exit 1
        ;;
esac
