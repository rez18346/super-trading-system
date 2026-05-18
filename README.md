# 🏭 Super Trading System

<div align="center">

**Профессиональная автоматическая торговая система для Bybit Spot**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Code style](https://img.shields.io/badge/code%20style-black-black)](https://github.com/psf/black)

</div>

---

## 📋 О системе

Мульти-агентная торговая система для криптовалютного спот-рынка Bybit. Анализирует рынок через **ансамбль голосов** (ML, MTF, ликвидность, OI-уровни ликвидаций, VWAP, VSA и другие), принимает взвешенные решения и управляет рисками. Вдохновлена профессиональными терминалами — дашборд показывает OI-зоны, POC, VAH/VAL, Order Flow, CVD.

**Ключевые особенности:**
- 🧠 **Ансамбль голосов:** ML-Pro (XGBoost), ML-Advisor, Multi-Timeframe (HMM), Liquidity Clusters (POC/VAH/VAL/FVG), RSI/Vol/BTC, Volume/VWAP, OI-бонус за уровни ликвидаций, VSA (Volume Spread Analysis), ML-Pro Reverter
- 📊 **Мультитаймфреймовый анализ** (5м, 1ч, 4ч) + HMM для определения режимов рынка
- 🔥 **OI Liquidation Levels** — расчёт зон ликвидаций (100x-5x плечи) на основе Open Interest дельты с отображением на дашборде
- 📈 **CVD (Cumulative Volume Delta)** — сбор тиковых сделок BTC в реальном времени через WebSocket, 9 признаков для модели направления BTC
- 🔮 **BTC Direction Predictor** — ML-модель (LightGBM) для предсказания направления BTC (4h), 82+ признака, accuracy ~70%, с самовалидацией (PredTrack)
- 🛡️ **Автоматические стоп-лоссы и трейлинг-стопы**
- 🔄 **Профессиональное управление рисками** (размер позиции от Score, корреляция, Black Swan)
- 🚀 **WebSocket реального времени** (тики BTC, свечи 19 монет)
- 🖥️ **Веб-дашборд** — порт 8765: статус, позиции, голоса, OI зоны, BTC Order Flow, VSA, EARN сигнал
- 💾 **SQLite БД** — вся история сделок и капитала

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
                     ┌─────────────────────────────────────┐
                     │           Decision Engine           │  ← Ансамбль голосов
                     │         (decision_engine.py)         │
                     └──────┬──────┬──────┬──────┬─────────┘
                            │      │      │      │
              ┌─────────────┘      │      └─────────────┐
              ↓                    │                    ↓
    ┌──────────────────┐          │          ┌──────────────────┐
    │   Industrial     │          │          │   BTC Direction  │
    │   Trader         │          │          │ (btc_direction)  │
    │ (order exec.)    │          │          │ ML predictor 4h  │
    └────────┬─────────┘          │          └────────┬─────────┘
             │                    │                   │
             ↓                    ↓                   ↓
    ┌──────────────────┐  ┌──────────────┐  ┌────────────────────┐
    │   CCXT / Bybit   │  │   collect_   │  │   CVD Collector    │
    │   API Layer      │  │   oi.py       │  │  (ws_client.py)    │
    │ REST + WebSocket │  │ OI Liq Levels│  │  BTC trade ticks   │
    └──────────────────┘  └──────────────┘  └────────────────────┘
             │
             ↓
    ┌──────────────────┐
    │   Control API    │  ← Веб-дашборд (порт 8765)
    │  (control_api)   │
    └──────────────────┘
```

### Ансамбль голосов

| Голос | Вес | Что делает |
|-------|-----|-----------|
| **ML-Pro** | 20% | XGBoost + LightGBM — обученная модель на 25+ признаках |
| **ML-Advisor** | 10% | Легковесный советник (если обучен) |
| **MTF** | 25% | Multi-Timeframe — тренды на 5м, 1ч, 4ч + HMM |
| **Liquidity** | 25% | POC, VAH/VAL, FVG, Order Blocks, объёмные профили |
| **VSA (Volume Spread)** | — | Анализ дисбаланса объёмов, bearish/bullish сигналы |
| **OI Liq Bonus** | — | +8..+35 баллов к Liq голосу при высоких OI дельтах (heat 1-3) |
| **ML-Pro Reverter** | — | Определение перепроданности/перекупленности, сигнал реверсии |
| **VV (VWAP+Volume)** | 20% | Развороты по VWAP + объёмные всплески |

### Новые модули

| Модуль | Файл | Назначение |
|--------|------|-----------|
| **OI Collector** | `collect_oi.py` | Сбор Open Interest для 19 монет, расчёт heat (1-3), уровни ликвидаций для 100x-5x плеч |
| **BTC Direction** | `btc_direction.py` | ML-модель (LightGBM) для предсказания направления BTC (4h), 82+ признака (цена, RSI, Liq, CVD), PredTrack-валидация |
| **CVD Collector** | `ws_client.py` | Сбор тиковых трейдов BTC через WebSocket, агрегация в 1-минутные корзины buy/sell объёмов |
| **Control API** | `control_api.py` | Веб-дашборд: позиции, голоса, OI зоны, BTC Order Flow, EARN сигнал, VSA |
| **Regime Tracker** | `btc_regime_tracker.py` | HMM-кластеризация режимов рынка BTC (pump/dump/accumulation/recovery)

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

### Веб-дашборд (порт 8765)

В браузере: `http://localhost:8765` — показывает:
- 🎫 Статус системы (PID, uptime, память)
- 💰 Капитал, свободные средства, количество позиций
- 📊 BTC Order Flow (POC, VAH, VAL, FVG, Order Blocks)
- 🔥 **OI Liquidation Zones** — монеты с аномальным OI, уровни ликвидаций (heat, цены) с подсветкой при входе в зону
- 🏦 **EARN сигнал** — рекомендация держать BTC (>50) или выйти (на основе BTC Direction)
- 📈 График капитала (последние 500 точек)
- 📋 Таблица позиций с текущими голосами
- 📋 Таблица сделок (последние 50) с PnL
- 📋 История VETO (причины блокировки сигналов)

### API эндпоинты

| Эндпоинт | Описание |
|----------|----------|
| `/api/status` | Полный статус: позиции, голоса, OI зоны, PnL, BTC Direction |
| `/api/trade-history` | История сделок (последние 500) |
| `/api/capital-history` | График капитала (последние 500 точек) |
| `/api/vote-history` | Все голоса (последние 200) |
| `/api/veto-history` | Причины VETO-блокировок |
| `/api/last-trade` | Последняя сделка (для дашборда) |

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

## 📁 Структура данных

| Файл | Формат | Назначение |
|------|--------|-----------|
| `data/trading.db` | SQLite | История сделок, капитал, обучение ML-Pro |
| `data/oi_data.json` | JSON | OI история (до 500 записей на монету) |
| `data/btc_cvd_1m.csv` | CSV | CVD-корзины BTC (1 мин), автосохраняется каждые 60 с |
| `data/btc_features.json` | JSON | Нормализованные фичи BTC для модели Direction |
| `data/btc_direction.pkl` | Pickle | Обученная модель BTC Direction (LightGBM) |
| `data/btc_scaler.pkl` | Pickle | Scaler для фич BTC Direction |
| `data/vote_history.json` | JSON | Все голоса ансамбля (до 10000) |
| `data/positions.json` | JSON | Текущие позиции |
| `/tmp/system_v4.log` | Лог | Основной лог системы

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
- Дашборд: открыть `http://localhost:8765` в браузере
- Веб-сервис: `curl http://localhost:8765/api/status`

**При проблемах:**
- Проверьте `data/oi_data.json` — не пустой ли (первый сбор OI занимает ~15 мин)
- CVD накапливается ~24 часа до использования в модели
- Перезапуск: `pkill -f main.py && python3 main.py`

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
