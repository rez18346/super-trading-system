#!/usr/bin/env python3
"""
orchestrator.py — Единый оркестратор супер-системы.

Архитектура:
  ┌────────────────────────────────────────┐
  │         ORCHESTRATOR (main.py)         │
  │  PID-файл · systemd · логи · решения   │
  ├────────────────────────────────────────┤
  │  │── trader (subprocess)               │
  │  │── dashboard (поток)                 │
  │  │── healthcheck (сердцебиение)        │
  └────────────────────────────────────────┘

Принципы:
  - Только один процесс оркестратора (PID-файл → дубли блокируются)
  - Все дочерние процессы — subprocess с PID (изоляция, управление)
  - graceful stop через SIGTERM, timeout → SIGKILL
  - state.json — снапшот для восстановления при падении оркестратора
  - Healthcheck каждые 30 сек: жив ли трейдер, жива ли система
"""

import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

import json
import time
import signal
import logging
import subprocess
import threading
from datetime import datetime, timezone
from typing import Optional, Dict

# ─── Настройка логгера ──────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/system_v4.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger('orch')

# ─── PID-файл ────────────────────────────────────────────────────────────────
PID_FILE = os.path.join(BASE_DIR, "data", "orchestrator.pid")
STATE_FILE = os.path.join(BASE_DIR, "data", "orchestrator_state.json")

# ─── Конфигурация компонентов ────────────────────────────────────────────────
COMPONENTS = {
    'trader': {
        'cmd': [sys.executable, '-B', '-u', os.path.join(BASE_DIR, 'trader_entry.py')],
        'graceful_shutdown_sec': 10,
        'max_silent_sec': 180,  # 3 минуты без heartbeat = мёртв
        'restart_delay_sec': 5,
        'max_restarts': 5,  # в течение часа
        'restart_window_sec': 3600,
    },
}


def _check_pid_file():
    """
    Единственный PID-страж. Вызывается один раз при старте.

    Логика:
      1. Если PID-файл существует и процесс в нём жив и это main.py —
         выходим (дубль не нужен).
      2. Если PID мёртв или это не main.py — чистим трейдеров-сирот
         и занимаем место.
      3. Записываем свой PID.
    """
    current_pid = os.getpid()
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)

    # Читаем старый PID
    old_pid = None
    try:
        with open(PID_FILE) as f:
            raw = f.read().strip()
            if raw:
                old_pid = int(raw)
    except (FileNotFoundError, ValueError):
        pass

    # Если PID-файл уже указывает на нас — всё ок
    if old_pid == current_pid:
        return

    # Проверка: жив ли старый процесс и это main.py?
    if old_pid:
        try:
            os.kill(old_pid, 0)
            # Жив. Кто это?
            cmdline = ''
            try:
                with open(f'/proc/{old_pid}/cmdline', 'rb') as f:
                    cmdline = f.read().replace(b'\x00', b' ').decode('utf-8', errors='replace')
            except Exception:
                pass

            if 'orchestrator.py' in cmdline or 'main.py' in cmdline:
                log.warning(f"🔄 Оркестратор PID={old_pid} уже работает — выхожу (exit 1 для systemd)")
                sys.exit(1)
            else:
                # Чужак (старый control_api, старый трейдер) — убиваем
                log.warning(f"⚰️ Чужой процесс PID={old_pid} ({cmdline[:80]}), убиваю...")
                os.kill(old_pid, 9)
                time.sleep(0.5)
        except ProcessLookupError:
            log.info(f"📝 Старый PID={old_pid} мёртв — перезаписываю")

    # Чистим сирот трейдеров (могут висеть от упавшего оркестратора)
    _cleanup_orphans()

    # Записываем свой PID
    _write_pid(current_pid)


def _write_pid(pid: int):
    import tempfile
    tmp = PID_FILE + '.tmp'
    with open(tmp, 'w') as f:
        f.write(str(pid))
    os.rename(tmp, PID_FILE)
    log.info(f"📝 PID-файл оркестратора: {pid}")


def _cleanup_orphans():
    """Найти и убить процессы-сироты:
       - industrial_trader.py (без живого родителя-оркестратора)
       - control_api.py (без живого main.py/orchestrator.py)
    """
    import psutil  # тяжёлая зависимость, используем только на старте
    try:
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'ppid']):
            try:
                cmdline = ' '.join(proc.info.get('cmdline') or [])
                pid = proc.info['pid']
                ppid = proc.info['ppid']

                # Проверяем только наши процессы
                if pid == current_pid:
                    continue
                if 'main.py' not in cmdline and 'orchestrator.py' not in cmdline \
                   and 'industrial_trader.py' not in cmdline and 'control_api.py' not in cmdline:
                    continue

                # Если родитель мёртв (PPid=1) — сирота
                if ppid == 1 or ppid == 0:
                    log.warning(f"🧹 Сирота PID={pid} ({cmdline[:60]}), убиваю")
                    proc.kill()
                    time.sleep(0.2)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        log.warning("⚠️ psutil не установлен, очистка сирот пропущена")
    except Exception as e:
        log.debug(f"cleanup_orphans: {e}")


# ─── State Manager ────────────────────────────────────────────────────────────

class StateManager:
    """
    Управляет состоянием оркестратора.
    Сохраняется в state.json для восстановления после падения.
    """

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.lock = threading.Lock()
        self.state: Dict = {
            'orchestrator_pid': os.getpid(),
            'started_at': datetime.now(timezone.utc).isoformat(),
            'components': {},  # {name: {pid, status, started_at, last_heartbeat, restart_count}}
            'last_regime': None,
            'btc_regime': None,
        }
        self._load()

    def _load(self):
        try:
            with open(self.state_file) as f:
                saved = json.load(f)
                # Берём только то что не перезатрёт новое состояние
                if 'last_regime' in saved:
                    self.state['last_regime'] = saved['last_regime']
                if 'btc_regime' in saved:
                    self.state['btc_regime'] = saved['btc_regime']
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save(self):
        with self.lock:
            self.state['saved_at'] = datetime.now(timezone.utc).isoformat()
            tmp = self.state_file + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(self.state, f, indent=2, default=str)
            os.rename(tmp, self.state_file)

    def component_started(self, name: str, pid: int):
        with self.lock:
            self.state['components'][name] = {
                'pid': pid,
                'status': 'starting',
                'started_at': datetime.now(timezone.utc).isoformat(),
                'last_heartbeat': datetime.now(timezone.utc).isoformat(),
                'restart_count': self.state['components'].get(name, {}).get('restart_count', 0),
            }
        self.save()

    def component_heartbeat(self, name: str):
        with self.lock:
            if name in self.state['components']:
                self.state['components'][name]['last_heartbeat'] = datetime.now(timezone.utc).isoformat()
                self.state['components'][name]['status'] = 'running'
        # Не сохраняем каждый heartbeat — слишком часто
        # Сохраняем каждые 5 хартбитов
        if int(time.time()) % 5 == 0:
            self.save()

    def component_stopped(self, name: str):
        with self.lock:
            if name in self.state['components']:
                self.state['components'][name]['status'] = 'stopped'
        self.save()

    def component_failed(self, name: str):
        with self.lock:
            if name in self.state['components']:
                rc = self.state['components'][name].get('restart_count', 0)
                self.state['components'][name]['restart_count'] = rc + 1
                self.state['components'][name]['status'] = 'failed'
                self.state['components'][name]['last_failure'] = datetime.now(timezone.utc).isoformat()
        self.save()

    def get_component(self, name: str) -> Optional[Dict]:
        with self.lock:
            return self.state['components'].get(name)

    def set_regime(self, regime: str, btc_regime: str):
        with self.lock:
            self.state['last_regime'] = regime
            self.state['btc_regime'] = btc_regime
        self.save()

    def set_pid(self, pid: int):
        with self.lock:
            self.state['orchestrator_pid'] = pid
        self.save()

    def to_dict(self) -> Dict:
        with self.lock:
            return dict(self.state)


# ─── Trader Manager ───────────────────────────────────────────────────────────

class TraderManager:
    """
    Управляет процессом трейдера (trader_entry.py).
    Запускает как subprocess с pipe для heartbeat, мониторит, перезапускает.
    """

    def __init__(self, config: Dict, state: StateManager):
        self.config = config
        self.state = state
        self.process: Optional[subprocess.Popen] = None
        self.running = False
        self._lock = threading.Lock()
        self._last_heartbeat = 0.0
        self._heartbeat_thread: Optional[threading.Thread] = None

    def start(self):
        with self._lock:
            if self.process and self.process.poll() is None:
                log.warning("⚠️ Трейдер уже запущен, пропускаю")
                return

            self.running = True
            log.info("🚀 Запуск трейдера...")
            try:
                # Создаём pipe для heartbeat
                r_fd, w_fd = os.pipe()

                cmd = self.config['cmd'] + ['--heartbeat-fd', str(w_fd),
                                            os.path.join(BASE_DIR, "config", "api_config_final.json")]
                self.process = subprocess.Popen(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    pass_fds=(w_fd,),
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                )

                # Закрываем write-конец в родителе (он у дочки)
                os.close(w_fd)

                # Запускаем чтение heartbeat в потоке
                self._start_heartbeat_reader(r_fd)

                self.state.component_started('trader', self.process.pid)
                log.info(f"✅ Трейдер запущен: PID {self.process.pid}")
            except Exception as e:
                log.error(f"❌ Ошибка запуска трейдера: {e}")
                self.state.component_failed('trader')

    def _start_heartbeat_reader(self, r_fd: int):
        """Читает heartbeat из pipe в отдельном потоке."""
        def _reader():
            try:
                with os.fdopen(r_fd, 'r') as f:
                    while self.running:
                        line = f.readline()
                        if not line:
                            break  # EOF — дочерний процесс умер
                        try:
                            hb = json.loads(line.strip())
                            if hb.get('type') == 'heartbeat':
                                self._last_heartbeat = time.time()
                                self.state.component_heartbeat('trader')
                                log.debug(f"💓 Heartbeat от трейдера PID {hb.get('pid')}")
                        except (json.JSONDecodeError, Exception):
                            pass
            except (BrokenPipeError, OSError):
                pass
            except Exception as e:
                log.debug(f"heartbeat reader error: {e}")
            log.debug("💓 Heartbeat reader завершён")

        self._heartbeat_thread = threading.Thread(target=_reader, name='hb-reader', daemon=True)
        self._heartbeat_thread.start()

    def stop(self, timeout: int = 10):
        with self._lock:
            if not self.process or self.process.poll() is not None:
                log.info("ℹ️ Трейдер не запущен, нечего останавливать")
                self.running = False
                return

            pid = self.process.pid
            log.info(f"🛑 Остановка трейдера PID {pid}...")

            # SIGTERM — graceful shutdown
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass

            # Ждём graceful shutdown
            try:
                self.process.wait(timeout=timeout)
                log.info(f"✅ Трейдер PID {pid} завершён gracefully")
            except subprocess.TimeoutExpired:
                # SIGKILL — принудительно
                log.warning(f"⚠️ Трейдер PID {pid} не завершился за {timeout}с, шлём SIGKILL")
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self.process.kill()
                self.process.wait()
                log.warning(f"💀 Трейдер PID {pid} убит принудительно")

            self.state.component_stopped('trader')
            self.process = None
            self.running = False

    def restart(self):
        log.info("🔄 Перезапуск трейдера...")
        self.stop(timeout=5)
        time.sleep(self.config['restart_delay_sec'])
        self.start()

    def is_alive(self) -> bool:
        with self._lock:
            if self.process and self.process.poll() is None:
                return True
            return False

    def get_uptime_since_last_heartbeat(self) -> Optional[float]:
        """Сколько секунд прошло с последнего heartbeat."""
        if self._last_heartbeat > 0:
            return time.time() - self._last_heartbeat
        return None

    def get_pid(self) -> Optional[int]:
        with self._lock:
            return self.process.pid if self.process else None


# ─── Dashboard Manager ───────────────────────────────────────────────────────

class DashboardManager:
    """
    Управляет дашбордом (control_api.py).
    Запускается как subprocess (отдельный процесс, не блокирует оркестратор).
    """

    def __init__(self, state: StateManager, port: int = 8765):
        self.state = state
        self.port = port
        self._process: Optional[subprocess.Popen] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            if self._process and self._process.poll() is None:
                log.warning("⚠️ Дашборд уже запущен, пропускаю")
                return

            self._running = True
            log.info(f"📊 Запуск дашборда на порту {self.port}...")
            try:
                cmd = [sys.executable, '-B', '-u',
                       os.path.join(BASE_DIR, 'control_api.py'),
                       '--port', str(self.port), '--standalone']
                self._process = subprocess.Popen(
                    cmd,
                    cwd=BASE_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    preexec_fn=os.setsid if hasattr(os, 'setsid') else None,
                )
                log.info(f"✅ Дашборд запущен: PID {self._process.pid}")
            except Exception as e:
                log.error(f"❌ Ошибка запуска дашборда: {e}")

    def stop(self, timeout: int = 5):
        with self._lock:
            if not self._process or self._process.poll() is not None:
                self._running = False
                return

            pid = self._process.pid
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                self._process.wait(timeout=timeout)
            except (ProcessLookupError, PermissionError):
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                self._process.kill()
                self._process.wait()

            self._process = None
            self._running = False

    def is_alive(self) -> bool:
        with self._lock:
            return self._process is not None and self._process.poll() is None

    def get_pid(self) -> Optional[int]:
        with self._lock:
            return self._process.pid if self._process else None


# ─── Healthcheck ──────────────────────────────────────────────────────────────

class Healthcheck:
    """
    Проверяет здоровье системы:
      - Трейдер жив и отвечает?
      - Нет ли зависших циклов?
      - Не превышен ли лимит рестартов?
    
    Запускается как фоновый поток.
    """

    def __init__(self, trader: TraderManager, state: StateManager, interval_sec: int = 30):
        self.trader = trader
        self.state = state
        self.interval = interval_sec
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Счётчики рестартов за окно
        self._restart_times: Dict[str, list] = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, name='health', daemon=True)
        self._thread.start()
        log.info(f"❤️ Healthcheck запущен (интервал {self.interval}с)")

    def _run(self):
        while self._running:
            try:
                self._check()
            except Exception as e:
                log.error(f"❤️‍🩹 Healthcheck error: {e}")
            time.sleep(self.interval)

    def _check(self):
        now = time.time()
        trader_comp = self.state.get_component('trader')

        # Проверка: трейдер жив?
        if trader_comp and trader_comp.get('status') in ('running', 'starting'):
            if not self.trader.is_alive():
                # Трейдер помечен как running, но процесс мёртв
                log.warning("❤️‍🩹 Трейдер мёртв (помечен как running, но процесс не отвечает)")
                self.state.component_failed('trader')
                self._maybe_restart('trader', now)

        # Проверка: не молчит ли трейдер слишком долго?
        if trader_comp and trader_comp.get('status') == 'running':
            max_silent = self.trader.config.get('max_silent_sec', 180)
            # Оценка: смотрим heartbeat через state или uptime
            silent_sec = self.trader.get_uptime_since_last_heartbeat()
            if silent_sec is not None and silent_sec > max_silent:
                log.warning(f"❤️‍🩹 Трейдер молчит {silent_sec:.0f}с — перезапуск")
                self.trader.restart()
            elif silent_sec is None and self.trader.is_alive():
                # Heartbeat пока не было, но процесс жив — ждём
                started = self.trader.get_pid()
                log.debug(f"💓 Heartbeat ещё не получен от PID {started}")

        # Периодическое сохранение состояния
        if int(now) % 60 < self.interval:  # примерно раз в минуту
            self.state.save()

    def _maybe_restart(self, name: str, now: float):
        config = COMPONENTS.get(name)
        if not config:
            return

        max_restarts = config['max_restarts']
        window = config['restart_window_sec']
        restart_delay = config['restart_delay_sec']

        # Очистка старых рестартов
        self._restart_times.setdefault(name, [])
        self._restart_times[name] = [t for t in self._restart_times[name] if now - t < window]

        if len(self._restart_times[name]) >= max_restarts:
            log.critical(f"🚨 {name}: превышен лимит рестартов ({max_restarts} за {window//60}мин). Пропускаю.")
            return

        self._restart_times[name].append(now)
        log.info(f"🔄 {name}: рестарт #{len(self._restart_times[name])} (через {restart_delay}с)")
        time.sleep(restart_delay)

        if name == 'trader':
            self.trader.restart()

    def stop(self):
        self._running = False


# ─── Signal Handler ───────────────────────────────────────────────────────────

class SignalHandler:
    """
    Обрабатывает сигналы от systemd:
      SIGTERM — graceful shutdown всего
      SIGHUP — перезагрузка конфига (заглушка)
      SIGUSR1 — статус в лог
    """

    def __init__(self, orchestrator: 'Orchestrator'):
        self.orch = orchestrator

    def setup(self):
        signal.signal(signal.SIGTERM, self._handle_sigterm)
        signal.signal(signal.SIGHUP, self._handle_sighup)
        signal.signal(signal.SIGUSR1, self._handle_sigusr1)
        log.info("🔔 Обработчики сигналов установлены")

    def _handle_sigterm(self, signum, frame):
        log.info("🛑 Получен SIGTERM — завершение работы...")
        self.orch.shutdown()

    def _handle_sighup(self, signum, frame):
        log.info("🔄 Получен SIGHUP — перезагрузка конфига (пока заглушка)")
        # TODO: перечитать конфиг, перезапустить компоненты

    def _handle_sigusr1(self, signum, frame):
        log.info(f"📋 Статус: {json.dumps(self.orch.state.to_dict(), indent=2, default=str)}")


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class Orchestrator:
    """
    Главный оркестратор. Точка входа всей супер-системы.
    """

    def __init__(self):
        log.info("=" * 50)
        log.info("🏭 ОРКЕСТРАТОР V1")
        log.info(f"   PID: {os.getpid()}")
        log.info(f"   БД: trading (PostgreSQL)")
        log.info(f"   PID-файл: {PID_FILE}")
        log.info("=" * 50)

        # Состояние
        self.state = StateManager(STATE_FILE)
        self.state.state['orchestrator_pid'] = os.getpid()
        self.state.save()

        # Менеджеры компонентов
        self.trader = TraderManager(COMPONENTS['trader'], self.state)
        self.dashboard = DashboardManager(self.state)
        self.healthcheck = Healthcheck(self.trader, self.state)

        # Сигналы
        self.signal_handler = SignalHandler(self)
        self.signal_handler.setup()

        # Флаг работы
        self.running = True

    def start(self):
        log.info("🚀 Запуск компонентов...")

        # 1. Трейдер (subprocess) — дашборд временно отключён
        self.trader.start()

        # 3. Healthcheck (поток)
        self.healthcheck.start()

        log.info("✅ Все компоненты запущены")
        self.state.save()

    def run(self):
        """Основной цикл оркестратора."""
        self.start()

        cycle_count = 0
        while self.running:
            try:
                time.sleep(15)
                cycle_count += 1

                # Проверка здоровья трейдера
                if self.trader.is_alive():
                    self.state.component_heartbeat('trader')
                else:
                    trader_state = self.state.get_component('trader')
                    if trader_state and trader_state.get('status') == 'running':
                        log.warning("⚠️ Трейдер упал, запускаю...")
                        self.trader.start()

                # Проверка дашборда — временно отключён

                # Статус раз в 4 цикла (минута)
                if cycle_count % 4 == 0:
                    self._log_status()

            except Exception as e:
                log.error(f"[ORCH] Ошибка цикла: {e}")

        log.info("🏁 Оркестратор завершён")

    def _log_status(self):
        """Логирование статуса."""
        try:
            trader_comp = self.state.get_component('trader')
            trader_status = trader_comp.get('status', 'unknown') if trader_comp else 'unknown'
            trader_pid = trader_comp.get('pid') if trader_comp else None
            trader_restarts = trader_comp.get('restart_count', 0) if trader_comp else 0

            log.info(f"📊 СТАТУС | Трейдер: {trader_status} (PID {trader_pid}) | Рестартов: {trader_restarts}")
        except Exception as e:
            log.debug(f"[STATUS] ошибка: {e}")

    def shutdown(self):
        """Graceful shutdown всех компонентов."""
        log.info("🛑 Остановка компонентов...")
        self.running = False
        self.healthcheck.stop()
        self.trader.stop()
        self.state.component_stopped('trader')
        self.state.save()
        log.info("👋 Все компоненты остановлены. Пока!")
        sys.exit(0)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _check_pid_file()
    orch = Orchestrator()
    try:
        orch.run()
    except KeyboardInterrupt:
        log.info("⌨️ Прервано пользователем")
        orch.shutdown()
    except Exception as e:
        log.critical(f"❌ Фатальная ошибка оркестратора: {e}")
        orch.shutdown()
