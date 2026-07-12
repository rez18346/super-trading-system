#!/usr/bin/env python3
"""
orchestrator.py — Оркестратор процессов трейдера.

Запускает:
  1. super_trader.py  — торговый движок
  2. control_api.py   — дашборд / REST API

Автоматически перезапускает упавший процесс.
Останавливает всё по Ctrl+C.
"""

import os, sys, time, signal, logging, subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log = logging.getLogger("orchestrator")

PROCESSES = [
    {
        'name': 'trader',
        'cmd': [sys.executable, '-B', '-u', os.path.join(BASE_DIR, 'super_trader.py')],
        'restart_delay': 5,
    },
    {
        'name': 'dashboard',
        'cmd': [sys.executable, '-B', '-u', os.path.join(BASE_DIR, 'control_api.py')],
        'restart_delay': 3,
    },
]

PIDFILE = '/tmp/openclaw-orchestrator.pid'

class Orchestrator:
    def __init__(self):
        self.running = True
        self.processes = {}  # name -> subprocess.Popen
        self.setup_logging()
        # PID-файл для защиты от дублей
        if self._check_pidfile():
            sys.exit(0)
        self._write_pidfile()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
            ]
        )

    def signal_handler(self, signum, frame):
        log.info("🛑 Получен сигнал остановки, гашу процессы...")
        self.running = False
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                log.info(f"⛔ Останавливаю {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning(f"⚠️ {name} не ответил на SIGTERM, киллю...")
                    proc.kill()
        log.info("✅ Все процессы остановлены")

    def start_process(self, name, cmd):
        # Для dashboard проверяем, не занят ли порт
        if name == 'dashboard':
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(('0.0.0.0', 8765))
                s.close()
            except OSError:
                log.info(f"⏭️ Порт 8765 уже занят — dashboard уже работает, не запускаю")
                # Возвращаем «заглушку», чтобы мониторинг не дёргался
                fake = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(99999)'],
                                        stdout=sys.stdout, stderr=sys.stderr)
                return fake
        log.info(f"🚀 Запуск {name}: {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            cwd=BASE_DIR
        )

    def _ensure_pg(self):
        """Проверяет что PostgreSQL работает, если нет — запускает."""
        import socket
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.connect('/tmp/.s.PGSQL.5432')
            sock.close()
            log.info("✅ PostgreSQL уже запущен")
            return True
        except (ConnectionRefusedError, FileNotFoundError):
            log.warning("🔄 PostgreSQL не запущен, запускаю...")
            pg_ctl = os.path.expanduser('~/.local/pgsql/bin/pg_ctl')
            pg_data = os.path.expanduser('~/.local/pgsql/data')
            
            if not os.path.exists(pg_data):
                log.warning("📁 Data-директория PG не найдена, инициализирую...")
                initdb = os.path.expanduser('~/.local/pgsql/bin/initdb')
                subprocess.run([initdb, '-D', pg_data], capture_output=True)
                # Настраиваем
                with open(os.path.join(pg_data, 'postgresql.conf'), 'a') as f:
                    f.write("\nunix_socket_directories = '/tmp'\n")
                    f.write("port = 5432\n")
                    f.write("listen_addresses = ''\n")
            
            result = subprocess.run(
                [pg_ctl, 'start', '-D', pg_data, '-l', '/tmp/pg.log'],
                capture_output=True, timeout=10
            )
            if result.returncode == 0:
                log.info("✅ PostgreSQL запущен")
                return True
            else:
                log.error(f"❌ Не удалось запустить PG: {result.stderr.decode()}")
                return False

    def run(self):
        # Проверяем PostgreSQL
        self._ensure_pg()
        
        # Стартуем всё
        for proc in PROCESSES:
            p = self.start_process(proc['name'], proc['cmd'])
            self.processes[proc['name']] = p

        log.info("✅ Оркестратор запущен, мониторю процессы...")

        # Мониторим
        while self.running:
            time.sleep(2)
            for proc in PROCESSES:
                name = proc['name']
                p = self.processes.get(name)
                if p and p.poll() is not None:
                    ret = p.returncode
                    log.warning(f"💀 {name} упал с кодом {ret}, перезапуск через {proc['restart_delay']}с...")
                    time.sleep(proc['restart_delay'])
                    self.processes[name] = self.start_process(name, proc['cmd'])

    def _check_pidfile(self):
        """Проверяет PID-файл. Если процесс жив — выходим, если мёртв — чистим."""
        if os.path.exists(PIDFILE):
            try:
                with open(PIDFILE) as f:
                    old_pid = int(f.read().strip())
                # Проверяем, жив ли процесс
                if os.path.exists(f'/proc/{old_pid}'):
                    try:
                        with open(f'/proc/{old_pid}/cmdline') as cf:
                            cmd = cf.read()
                        if 'orchestrator.py' in cmd:
                            log.warning(f"⛔ Оркестратор уже запущен (PID {old_pid})")
                            return True
                    except:
                        pass
            except (ValueError, OSError):
                pass
            os.remove(PIDFILE)
        return False

    def _write_pidfile(self):
        with open(PIDFILE, 'w') as f:
            f.write(str(os.getpid()))

    def _clean_pidfile(self):
        try:
            if os.path.exists(PIDFILE):
                with open(PIDFILE) as f:
                    if f.read().strip() == str(os.getpid()):
                        os.remove(PIDFILE)
        except:
            pass

    def signal_handler(self, signum, frame):
        log.info("🛑 Получен сигнал остановки, гашу процессы...")
        self.running = False
        self._clean_pidfile()
        for name, proc in self.processes.items():
            if proc and proc.poll() is None:
                log.info(f"⛔ Останавливаю {name} (PID {proc.pid})...")
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning(f"⚠️ {name} не ответил на SIGTERM, киллю...")
                    proc.kill()
        log.info("✅ Все процессы остановлены")

if __name__ == '__main__':
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch._clean_pidfile()
