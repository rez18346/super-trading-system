#!/usr/bin/env python3
"""
main.py — Entry point.
1. Проверяет что orchestrator.py не запущен (pgrep)
2. Запускает orchestrator.py как subprocess
3. Ждёт его завершения (не exit, а wait)
4. Возвращает exit code subprocess'а

Это позволяет:
- systemd отслеживать процесс (мы не exit(0) при конкуренции)
- OpenClaw запускать main.py сколько угодно
- Если процесс уже работает — тихий exit(0)
- Если процесс мёртв — запуск
"""

import sys
import os
import subprocess
import time
import signal

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _is_orchestrator_running() -> bool:
    """Проверяет жив ли orchistrator.py (исключая себя)."""
    my_pid = os.getpid()
    try:
        out = subprocess.check_output(
            ['pgrep', '-f', 'orchestrator.py'],
            timeout=3, stderr=subprocess.DEVNULL
        ).decode().strip()
        if not out:
            return False
        for pid_str in out.split('\n'):
            pid = pid_str.strip()
            if not pid:
                continue
            try:
                pid_i = int(pid)
                if pid_i == my_pid:
                    continue
                cmdline = open(f'/proc/{pid_i}/cmdline').read().replace('\0', ' ')
                if 'orchestrator.py' in cmdline:
                    return True
            except (ValueError, FileNotFoundError, IOError):
                continue
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return False


if __name__ == "__main__":
    # ⚡ Флаг-файл: systemd рестартит нас каждые 10с, блокируем дубли
    _lock = os.path.join(BASE_DIR, '.main_running')
    if os.path.exists(_lock):
        # Старше 5 минут? Тогда можно перезапустить
        try:
            age = time.time() - os.path.getmtime(_lock)
            if age < 300:
                sys.exit(0)
        except:
            pass
    # Создаём/обновляем флаг
    try:
        open(_lock, 'w').write(str(os.getpid()))
    except:
        pass

    if _is_orchestrator_running():
        sys.exit(0)

    orch = os.path.join(BASE_DIR, "orchestrator.py")

    # ⚡ Fork + detach: запускаем orchestrator в фоне, а main.py завершается
    # Это нужно чтобы systemd не убивал трейдер через 10с
    pid = os.fork()
    if pid == 0:
        # Дочерний процесс — запускаем оркестратор
        os.setsid()
        proc = subprocess.Popen(
            [sys.executable, '-B', '-u', orch] + sys.argv[1:],
            cwd=BASE_DIR,
            stdin=subprocess.DEVNULL,
            stdout=open('/dev/null', 'w'),
            stderr=open('/dev/null', 'w'),
            close_fds=True,
        )
        proc.wait()
        # Удаляем флаг при завершении
        try:
            os.remove(_lock)
        except:
            pass
        sys.exit(proc.returncode or 0)
    else:
        # Родительский процесс — выходим сразу (systemd видит success)
        sys.exit(0)
