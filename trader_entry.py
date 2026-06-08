#!/usr/bin/env python3
"""
trader_entry.py — Entry point для запуска IndustrialTrader как subprocess.

Запускает IndustrialTrader.start() и ждёт SIGTERM.
При SIGTERM — IndustrialTrader.stop() (graceful, сохраняет позиции).

Параметры:
  --heartbeat-fd <fd> — файловый дескриптор для heartbeat (pipe от оркестратора)
  config_path — путь к конфигу
"""

import sys
import os
import signal
import time
import json
import logging
from datetime import datetime, timezone

# Настройка логирования (без дублирования с industrial_trader)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
log = logging.getLogger('trader_entry')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

_shutdown_requested = False


def signal_handler(signum, frame):
    global _shutdown_requested
    if signum == signal.SIGTERM:
        log.info("🛑 Получен SIGTERM — завершение после текущего цикла")
        _shutdown_requested = True
    elif signum == signal.SIGINT:
        log.info("⌨️ Получен SIGINT — завершение")
        _shutdown_requested = True


def _send_heartbeat(hb_file, pid: int):
    try:
        hb = json.dumps({
            'type': 'heartbeat',
            'pid': pid,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })
        hb_file.write(hb + '\n')
        hb_file.flush()
    except (BrokenPipeError, OSError):
        log.warning("⚠️ Heartbeat pipe разорван")
    except Exception:
        pass


def main():
    global _shutdown_requested

    # Парсинг аргументов
    heartbeat_fd = None
    config_path = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--heartbeat-fd' and i + 1 < len(args):
            heartbeat_fd = int(args[i + 1])
            i += 2
        else:
            config_path = args[i]
            i += 1

    if not config_path:
        config_path = os.path.join(BASE_DIR, "config", "api_config_final.json")
        log.warning(f"⚠️ Конфиг не указан, использую: {config_path}")

    # Инициализация heartbeat pipe
    hb_file = None
    if heartbeat_fd is not None:
        try:
            hb_file = os.fdopen(heartbeat_fd, 'w')
            log.info(f"💓 Heartbeat pipe: FD {heartbeat_fd}")
        except Exception as e:
            log.warning(f"⚠️ Heartbeat FD {heartbeat_fd} не открылся: {e}")

    # Установка обработчиков сигналов
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    log.info("🚀 Запуск IndustrialTrader...")

    from industrial_trader import IndustrialTrader

    trader = IndustrialTrader(config_path)

    # Запускаем трейдер (start устанавливает self.running=True и запускает потоки)
    trader.start()

    # Первый heartbeat сразу
    if hb_file:
        _send_heartbeat(hb_file, os.getpid())

    # Ожидаем сигнал завершения
    # trader.trading_cycle() сам бесконечный в своём потоке, мы только ждём
    while not _shutdown_requested:
        try:
            time.sleep(15)
            if hb_file:
                _send_heartbeat(hb_file, os.getpid())
        except Exception as e:
            log.error(f"❌ Ошибка в heartbeat: {e}")

    # Graceful shutdown: IndustrialTrader.stop() → self.running = False + save
    log.info("🛑 Graceful shutdown трейдера...")
    trader.stop()
    if hb_file:
        try:
            hb_file.close()
        except Exception:
            pass
    log.info("👋 Трейдер остановлен")


if __name__ == "__main__":
    main()
