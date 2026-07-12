#!/usr/bin/env python3
"""
super_trader.py — Запуск трейдера в отдельном screen-процессе.
Использует ту же БД, что и FastAPI.
"""

import os, sys, time, logging, json
from logging.handlers import RotatingFileHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Создаём корневой логгер с двумя хендлерами:
# 1. stderr (screen) — для живого просмотра
# 2. /tmp/system_v4.log — для vote_parser и истории сигналов
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# stderr
console = logging.StreamHandler()
console.setFormatter(formatter)
root_logger.addHandler(console)

# Файловый лог (100MB, 5 файлов ротации)
file_handler = RotatingFileHandler(
    '/tmp/system_v4.log', maxBytes=100*1024*1024, backupCount=5
)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

log = logging.getLogger("super_trader")

from industrial_trader import IndustrialTrader

def main():
    # ── Защита от дублей ──
    # Ищем процессы, где ядро (binary) == python и в cmdline есть super_trader.py
    import subprocess
    result = subprocess.run(['ps', 'ax', '-o', 'pid=,comm=,args='], capture_output=True, text=True)
    pids = []
    my_pid = str(os.getpid())
    for line in result.stdout.strip().split('\n'):
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, comm, args = parts
        if pid == my_pid:
            continue
        if 'python' in comm and 'super_trader.py' in args:
            pids.append(pid)
    if len(pids) > 0:
        log.warning(f"⛔ Другой процесс super_trader.py уже запущен: {pids}. Выхожу.")
        sys.exit(0)
    config_path = os.path.join(BASE_DIR, "config.json")
    log.info(f"📈 ЗАПУСК ТРЕЙДЕРА с конфигом: {config_path}")
    
    trader = IndustrialTrader(config_path)
    
    # Не используем signal.signal из трейдера (мы в отдельном процессе)
    # Просто запускаем цикл
    trader.running = True
    try:
        trader.trading_cycle()
    except KeyboardInterrupt:
        log.info("🛑 Получен Ctrl+C, останавливаю...")
    except Exception as e:
        log.error(f"❌ Ошибка: {e}")
    finally:
        trader.running = False
        log.info("✅ Трейдер остановлен")

if __name__ == "__main__":
    main()
