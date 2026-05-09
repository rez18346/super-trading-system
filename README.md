# 🏭 Super Trading System

<div align="center">

**Профессиональная автоматическая торговая система для Bybit Spot**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

</div>

---

## 📋 О системе

Мульти-агентная торговая система для криптовалютного спот-рынка Bybit. Анализирует рынок через **6 независимых голосов** (ML, MTF, ликвидность, VWAP и другие), принимает взвешенные решения и управляет рисками.

**Ключевые особенности:**
- 🧠 **6 голосов ансамбля:** ML-Pro, ML-Advisor, Multi-Timeframe, Liquidity Clusters, RSI/Vol/BTC, Volume/VWAP
- 📊 **Мультитаймфреймовый анализ** (5м, 1ч, 4ч) + HMM для определения режимов рынка
- 🛡️ **Автоматические стоп-лоссы и трейлинг-стопы**
- 🔄 **Профессиональное управление рисками** (размер позиции от Score, корреляция, Black Swan)
- 🚀 **WebSocket реального времени**
- 💾 **SQLite БД** — все данные в одном месте

---

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/rez18346/super-trading-system.git
cd super-trading-system
```

### 2. Установите зависимости

```bash
pip install -r requirements.txt
```

> Рекомендуется использовать виртуальное окружение:
> ```bash
> python3 -m venv venv
> source venv/bin/activate  # Linux/Mac
> # или
> venv\Scripts\activate     # Windows
> pip install -r requirements.txt
> ```

### 3. Настройте API-ключи Bybit

1. Создайте API-ключи на [Bybit](https://www.bybit.com/app/user/api-management) (права: Trade + Read)
2. Скопируйте шаблон конфига:
   ```bash
   cp config.example.py config.py
   ```
3. Откройте `config.py` и вставьте свои ключи:
   ```python
   BYBIT_API_KEY = "ваш_ключ"
   BYBIT_API_SECRET = "ваш_секрет"
   ```

### 4. Запустите

```bash
./start.sh
```

Или напрямую:
```bash
python3 main.py
```

Система запустится, подключится к Bybit через WebSocket и начнёт анализ и торговлю.

### 5. Мониторинг

```bash
# Логи в реальном времени
tail -f /tmp/system_v4.log

# Веб-интерфейс (если запущен)
# Откройте http://localhost:8765 в браузере
```

---

## 💻 Особенности платформ

### 🟢 Linux / VPS (рекомендуется)
Всё из коробки. Для production-запуска:
```bash
./start.sh
```
Для автостарта используйте [systemd](#-production-запуск-linux).

### 🟡 macOS
Всё работает. Для автостарта настройте launchctl.

### 🔵 Windows (через WSL2)
Если OpenClaw или Python уже работает через WSL2 (Ubuntu) — система запустится без проблем.

**Установка WSL2 (один раз):**
```powershell
# В PowerShell (администратор):
wsl --install -d Ubuntu
```

**После установки WSL:**
```bash
# Внутри WSL (Ubuntu):
sudo apt update && sudo apt install python3 python3-pip git -y
git clone https://github.com/rez18346/super-trading-system.git
cd super-trading-system
cp config.example.py config.py
# Отредактируйте config.py (nano config.py)
pip install -r requirements.txt
./start.sh
```

**Важно для WSL:**
- Ноутбук не должен уходить в сон — отключите спящий режим в Windows
- Дашборд доступен из Windows по адресу http://<IP-WSL>:8765 (IP покажет start.sh)
- После перезагрузки запускайте снова: `./start.sh`

---

## 🏗️ Архитектура

```
                    ┌─────────────────────┐
                    │   Decision Engine   │  ← Ансамбль из 6 голосов
                    │  (decision_engine)  │
                    └──────┬──────┬───────┘
                           │      │
              ┌────────────┘      └────────────┐
              ↓                                ↓
    ┌──────────────────┐          ┌──────────────────┐
    │   Industrial     │          │   Stop Loss      │
    │   Trader         │◄────────►│   Monitor        │
    │ (регистрация     │          │ (SL, трейлинг)   │
    │  ордеров)        │          │                  │
    └────────┬─────────┘          └──────────────────┘
             │
             ↓
    ┌──────────────────┐
    │   CCXT / Bybit   │  ← REST + WebSocket
    │   API Layer      │
    └──────────────────┘
```

### 6 голосов ансамбля

| Голос | Вес | Что делает |
|-------|-----|-----------|
| **ML-Pro** | 20% | XGBoost + LightGBM — обученная модель на 25 признаках |
| **ML-Advisor** | 10% | Легковесный советник (если обучен) |
| **MTF** | 25% | Multi-Timeframe — тренды на 5м, 1ч, 4ч + HMM |
| **Liquidity** | 25% | POC, VAH/VAL, FVG, объёмные профили |
| **RSI/Vol/BTC** | Отключён | Контекст BTC + RSI перекупленности |
| **Volume/VWAP** | 20% | Развороты по VWAP + объёмные всплески |

---

## ⚙️ Конфигурация

Основные настройки в `config.py`:

| Параметр | По умолч. | Описание |
|----------|-----------|----------|
| `MAX_BUY_ORDER_USD` | 90 | Максимальный размер ордера |
| `MAX_OPEN_POSITIONS` | 10 | Максимум открытых позиций |
| `TEST_MODE` | False | True = логгирование без реальных ордеров |
| `BYBIT_TESTNET` | False | Использовать testnet Bybit |
| `RECV_WINDOW` | 5000 | Окно синхронизации времени (мс) |
| `TRADING_PAIRS` | 19 пар | Список монет для торговли |

---

## 📊 Мониторинг

### Control API (порт 8765)

| Эндпоинт | Описание |
|----------|----------|
| `/status` | Статус системы, позиции, PnL |
| `/positions` | Все открытые позиции |
| `/metrics` | Метрики ML-моделей |
| `/trades` | История сделок |

### Логи

```bash
tail -f /tmp/system_v4.log
```

---

## 🔄 Production-запуск (Linux)

**Через systemd (рекомендуется):**

```ini
[Unit]
Description=Super Trading System
After=network.target

[Service]
Type=simple
User=ваш_пользователь
WorkingDirectory=/path/to/super-trading-system
ExecStart=/usr/bin/python3 /path/to/super-trading-system/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now super-trading-system
sudo systemctl status super-trading-system
```

---

## ⚠️ Риски

**Торговля криптовалютами — высокий риск.** Используйте только средства, которые готовы потерять полностью.

- 🟢 Система прошла 30+ дней реальной торговли
- 🟢 Ансамблевый подход снижает количество ложных сигналов
- 🔴 Ни одна система не гарантирует прибыль
- 🔴 Прошлые результаты не гарантируют будущих

---

## 📜 Лицензия

MIT — используйте, модифицируйте, делитесь.

---

## 🤖 Для AI-агентов

Если вы — AI-агент, которому поручили развернуть эту систему, выполните:

```
1. Клонировать: git clone https://github.com/rez18346/super-trading-system.git
2. Перейти: cd super-trading-system
3. Установить зависимости: pip install -r requirements.txt
4. Создать config.py из config.example.py
5. Запросить у пользователя API-ключи Bybit (Trade + Read)
6. Записать ключи в config.py
7. Запустить: python3 main.py
8. Создать systemd unit для автостарта (см. README → Production-запуск)
9. Сообщить пользователю, что система запущена и работает
```

**После запуска проверять:**
- Процесс жив: `ps aux | grep main.py`
- Логи: `tail -f /tmp/system_v4.log`
- Статус: `curl http://localhost:8765/status`

---

## 💬 Комьюнити и контрибьюции

Присоединяйтесь к развитию системы! Всё лучшее стекается в один репозиторий.

| Канал | Назначение |
|---|---|
| [🐛 Issues](https://github.com/rez18346/super-trading-system/issues) | Сообщить о баге |
| [💡 Discussions](https://github.com/rez18346/super-trading-system/discussions) | Идеи, фичи, вопросы |
| [🔀 Pull Requests](https://github.com/rez18346/super-trading-system/pulls) | Предложить улучшение |
| [📦 Releases](https://github.com/rez18346/super-trading-system/releases) | Следить за обновлениями |

**Схема работы:**
1. Нашёл баг → создаёшь **Issue**
2. Придумал фичу → создаёшь **Discussion**
3. Написал код → **Pull Request** → мы ревьюим → вмерживаем
4. Вышел апдейт → **Releases** → `git pull`

Все улучшения публикуются, все получают — экосистема растёт сама. Сливки собираем вместе 🤝

---

## 🤝 Контакты

По вопросам и предложениям — Issues на GitHub или Telegram.
