#!/usr/bin/env python3
"""
lifecycle.py — Lifecycle Manager для единого процесса трейдера.

Управляет состоянием процесса:
  STARTING → RUNNING → SHUTTING_DOWN → STOPPED

Обрабатывает SIGTERM/SIGINT — graceful shutdown.
Сохраняет состояние в БД перед выходом.
"""

import os
import sys
import signal
import logging
import json
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

log = logging.getLogger("lifecycle")


class State(Enum):
    STARTING = "starting"
    RUNNING = "running"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


class LifecycleManager:
    """
    Единый менеджер жизненного цикла процесса.
    
    Использование:
        lm = LifecycleManager("trader")
        lm.start()
        # ... работа ...
        lm.wait_for_shutdown()  # блокируется до SIGTERM/SIGINT
        lm.stop()
    """
    
    def __init__(self, name: str = "trader"):
        self.name = name
        self.state = State.STARTING
        self._shutdown_requested = False
        self._shutdown_callbacks: list[Callable] = []
        self._original_sigint = None
        self._original_sigterm = None
        self._start_time = datetime.now(timezone.utc)
        
    def start(self):
        """Переводит в RUNNING, устанавливает обработчики сигналов."""
        log.info(f"🚀 {self.name} запускается...")
        
        # Установка обработчиков сигналов
        self._original_sigint = signal.getsignal(signal.SIGINT)
        self._original_sigterm = signal.getsignal(signal.SIGTERM)
        
        def _handle_shutdown(signum, frame):
            if self._shutdown_requested:
                log.warning("⚠️  Повторный сигнал — принудительный выход")
                sys.exit(1)
            self._shutdown_requested = True
            self.state = State.SHUTTING_DOWN
            sig_name = signal.Signals(signum).name
            log.info(f"🛑 Получен сигнал {sig_name}, начинаю graceful shutdown...")
            self._do_shutdown()
            
        signal.signal(signal.SIGINT, _handle_shutdown)
        signal.signal(signal.SIGTERM, _handle_shutdown)
        
        self.state = State.RUNNING
        log.info(f"✅ {self.name} запущен (PID: {os.getpid()})")
        
    def on_shutdown(self, callback: Callable):
        """Добавляет callback, который вызовется при shutdown."""
        self._shutdown_callbacks.append(callback)
        
    def _do_shutdown(self):
        """Выполняет shutdown: вызывает callback'и, ждёт завершения."""
        log.info(f"⏳ Выполняется shutdown: {len(self._shutdown_callbacks)} callback(ов)")
        
        for i, cb in enumerate(self._shutdown_callbacks):
            try:
                cb()
                log.info(f"  ✅ Callback {i+1}/{len(self._shutdown_callbacks)} выполнен")
            except Exception as e:
                log.error(f"  ❌ Callback {i+1} упал: {e}")
                
        self.state = State.STOPPED
        uptime = datetime.now(timezone.utc) - self._start_time
        log.info(f"✅ {self.name} остановлен (работал {uptime})")
        
        # Восстанавливаем обработчики
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)
        
    def wait_for_shutdown(self):
        """Блокируется, пока не придёт сигнал на shutdown."""
        while not self._shutdown_requested:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                pass
                
    def is_running(self) -> bool:
        return self.state == State.RUNNING
        
    def is_shutting_down(self) -> bool:
        return self.state == State.SHUTTING_DOWN
    
    def uptime(self) -> str:
        delta = datetime.now(timezone.utc) - self._start_time
        hours, remainder = divmod(int(delta.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def get_state(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "pid": os.getpid(),
            "uptime": self.uptime(),
            "started_at": self._start_time.isoformat(),
        }
