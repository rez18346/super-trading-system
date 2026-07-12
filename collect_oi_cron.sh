#!/usr/bin/env bash
# collect_oi_cron.sh — Сбор OI данных каждые 15 минут
# Запускается из системного cron, не через OpenClaw (экономия токенов)
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$DIR/logs/oi_collector.log"
DATA_DIR="$DIR/data"

# Загружаем окружение из .env (Bybit API ключи)
if [ -f "$DIR/.env" ]; then
    set -a
    source "$DIR/.env"
    set +a
fi

# Создаём директории если нет
mkdir -p "$DATA_DIR" "$DIR/logs"

cd "$DIR"

# Пишем в лог
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Запуск OI collector..." >> "$LOG_FILE"

# Запускаем сборщик
python3 -c "
import ccxt, os, json, sys
sys.path.insert(0, '$DIR')
from collect_oi import OICollector, OI_PATH

ex = ccxt.bybit({
    'apiKey': os.environ.get('BYBIT_API_KEY', ''),
    'secret': os.environ.get('BYBIT_SECRET', ''),
    'options': {'defaultType': 'spot'},
})
oc = OICollector()
results = oc.collect(ex)
oc.save()
print(f'OK {len(results)} coins collected')
" >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] OI collector завершён" >> "$LOG_FILE"
