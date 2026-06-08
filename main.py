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
    if _is_orchestrator_running():
        sys.exit(0)

    orch = os.path.join(BASE_DIR, "orchestrator.py")
    proc = subprocess.Popen(
        [sys.executable, '-B', '-u', orch] + sys.argv[1:],
        cwd=BASE_DIR,
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGTERM)
        proc.wait()
    sys.exit(proc.returncode or 0)
