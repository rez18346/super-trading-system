# 🏭 Super Trading System

<div align="center">

**Профессиональная автоматическая торговая система для Bybit Spot**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📋 О системе

Мульти-агентная торговая система для криптовалютного спот-рынка Bybit. Анализирует рынок через **ансамбль модулей** (ML, ликвидность, объёмы, структура), принимает взвешенные решения и управляет рисками.

**Ключевые особенности:**
- 🧠 **Ансамбль голосов** — ML-Pro (XGBoost/LightGBM), Multi-Timeframe (HMM), Liquidity Clusters (POC/VAH/VAL/FVG), VSA, CVD, RP, SMC-фильтр
- 📊 **Мультитаймфреймовый анализ** — 5м, 1ч, 4ч + HMM для определения режимов рынка
- 🔥 **BTC Regime Tracker** — определяет режим BTC и блокирует входы в опасных режимах
- 🔮 **BTC Direction Predictor** — ML-модель для предсказания направления BTC (4h)
- 🛡️ **Автоматические стоп-лоссы**, трейлинг-стопы, безубыток, Impulse Exit
- 🔄 **WebSocket реального времени** — свечи 19 монет, тики BTC
- 🖥️ **Веб-дашборд** — порт 8765: статус, позиции, PnL, баланс
- 💾 **PostgreSQL** — вся история сделок и капитала

---

## 🔄 Алгоритм работы

Система работает как **непрерывный цикл сканирования**:

```
1. СБОР ДАННЫХ (каждые 5 минут)
   ├── WebSocket → свечи (1м, 5м, 1ч, 4ч) для 19 монет
   ├── REST API → тики BTC, Open Interest, Funding Rate
   └── PostgreSQL → история сделок, баланс
   
2. АНАЛИЗ РЫНКА
   ├── BTC Regime Tracker → режим BTC (bullish/bearish/sell_only/range)
   ├── BTC Direction Predictor → ML-прогноз на 4ч вперёд
   ├── HMM Regime → кластеризация режимов для каждой монеты
   └── Range-блок → блокировка входов в low-volume chop
   
3. СКАНИРОВАНИЕ МОНЕТ (19 пар)
   Для каждой монеты:
   ├── Liquidity Clusters → POC, VAH/VAL, FVG, Order Blocks (базовый score)
   ├── ML-Pro (XGBoost) → 27 признаков, бинарная классификация
   ├── ML-Advisor (XGBoost v3) → альтернативная оценка
   ├── VSA → Volume Spread Analysis, дивергенции
   ├── Volume/VWAP → развороты по VWAP, объёмные всплески
   ├── CVD → Cumulative Volume Delta, поток агрессивных сделок
   ├── RP (Recovery/Potential) → качество монеты по просадке
   └── SMC Filter (BTC контекст) → CHoCH, FVG, Order Blocks
   
4. ПРИНЯТИЕ РЕШЕНИЯ (Decision Engine)
   ├── Взвешенное голосование (Liquidity — единственный голосующий модуль)
   ├── base_score = raw Liquidity score (без раздувания бонусами)
   ├── final_score = base_score + бонусы (цена, разворот, BTC)
   ├── Порог входа проверяется по base_score
   ├── SMC veto → может заблокировать любой вход
   └── BTC context → может заблокировать вход в опасных режимах
   
5. УПРАВЛЕНИЕ ПОЗИЦИЯМИ (если сделка открыта)
   ├── Стоп-лосс (SL) → жёсткая защита
   ├── SL→BE (безубыток при ≥1.5%) → ни одна сделка не уходит в минус
   ├── Основной трейлинг 2% → при росте ≥2% от входа
   ├── Тейк-профит (TP) → целевая +10%
   ├── Profit Protection → фиксация при откате от пика
   ├── Break-even → выход в ноль через ≥4ч без движения
   ├── Impulse Exit → выход при истощении импульса (можно отключить)
   └── Decision Engine exit → SL/TP/таймаут с ensemble hold (68%)
   
6. ЛОГГИРОВАНИЕ И ОТЧЁТЫ
   ├── /tmp/system_v4.log — подробный лог
   ├── PostgreSQL — история сделок
   ├── /tmp/real_balance.json — реальные трейды с биржи
   └── Веб-дашборд на порту 8765
```

---

## 🏗️ Архитектура

```
                     ┌─────────────────────────────────────┐
                     │           Decision Engine           │
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

## 🔧 Навигация по модулям

| Файл | Назначение |
|------|------------|
| `orchestrator.py` | 🚀 **Точка входа** — запускает всё |
| `industrial_trader.py` | ⚙️ **Ядро** — цикл сканирования, исполнение |
| `decision_engine.py` | 🧠 **Мозг** — ансамбль голосов (145 KB) |
| `control_api.py` | 📊 **Веб-дашборд** (FastAPI, порт 8765) |
| `btc_regime_tracker.py` | 🔥 **Режим BTC** — блокировка в range/sell_only |
| `btc_direction.py` | 🔮 **ML-прогноз BTC** (LightGBM, 82+ признака) |
| `btc_smc_filter.py` | 🏛️ **SMC фильтр** — CHoCH, FVG, Order Blocks |
| `liquidity_cluster.py` | 💧 **Кластеры ликвидности** — база голосования |
| `ml_professional_v2.py` | 🤖 **ML-Pro v2** — LGBM модель (95% точность) |
| `ml_advisor.py` | 🧑‍💼 **ML-Advisor** — XGBoost v3 (99% точность) |
| `hmm_regime.py` | 📈 **HMM** — кластеризация режимов рынка |
| `vsa_analyzer.py` | 📊 **VSA** — Volume Spread Analysis |
| `volume_vwap.py` | 📐 **VWAP** — объёмные сигналы |
| `cvd_analyzer.py` | 🔄 **CVD** — Cumulative Volume Delta |
| `rp_analyzer.py` | 📉 **RP** — Recovery & Potential |
| `impulse_exit.py` | 🎯 **Impulse Exit** — выход при истощении |
| `position_manager.py` | 💼 **Управление позициями** |
| `order_manager.py` | 📋 **Управление ордерами** |
| `risk_manager.py` | 🛡️ **Риск-менеджмент** |
| `ws_client.py` | 🔌 **WebSocket** — свечи, тики |
| `collect_oi.py` | 📊 **Open Interest** — сбор для 19 монет |
| `db_pg.py` | 🗄️ **PostgreSQL** |
| `bybit_live.py` | 🔗 **Bybit интеграция** для дашборда |
| `fetch_real_balance.py` | 💰 **Реальный баланс** с биржи |

---

## 🚀 Быстрый старт для AI-агента

### Шаг 1. Клонирование

```bash
git clone https://github.com/rez18346/super-trading-system.git
cd super-trading-system
```

### Шаг 2. Установка зависимостей

```bash
# Рекомендуется виртуальное окружение
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Или одной командой:

```bash
./start.sh  # создаст venv и установит всё
```

### Шаг 3. Настройка API-ключей

Создайте API-ключи на [Bybit](https://www.bybit.com/app/user/api-management) с правами **Trade + Read**.

Скопируйте шаблон конфига:

```bash
cp config.json.example config.json
cp .env.example .env
```

Отредактируйте `config.json` — вставьте свои ключи Bybit:

```json
{
  "bybit": {
    "api_key": "***",
    "secret": "***"
  }
}
```

**⚠️ Файлы `config.json` и `.env` добавлены в `.gitignore` — они никогда не попадут в репозиторий.**

### Шаг 4. Запуск

```bash
# Через start.sh (создаёт venv, проверяет конфиг, запускает)
./start.sh

# Или напрямую:
python3 -B -u orchestrator.py
```

### Шаг 5. Мониторинг

```bash
# Логи в реальном времени
tail -f /tmp/system_v4.log

# Веб-дашборд
# Откройте http://localhost:8765 в браузере
```

---

## ⚙️ Конфигурация

Основные настройки в `config.json`:

| Параметр | По умолч. | Описание |
|----------|-----------|----------|
| `bybit.api_key` | `—` | API ключ Bybit (обязательно) |
| `bybit.secret` | `—` | API секрет Bybit (обязательно) |
| `trading.max_order_usd` | 90 | Максимальный размер ордера |
| `trading.max_open_positions` | 10 | Максимум открытых позиций |
| `system.test_mode` | false | True = логгирование без реальных ордеров |
| `risk_management.default_sl_pct` | 2.5 | Стоп-лосс % |
| `risk_management.tp_pct` | 2.5 | Тейк-профит % |

Дополнительные переменные окружения (`.env`):

| Переменная | Описание |
|------------|----------|
| `BYBIT_API_KEY` | Переопределяет ключ из config.json |
| `BYBIT_SECRET` | Переопределяет секрет из config.json |
| `BYBIT_PASSWORD` | Пароль для вывода средств |

---

## 🖥️ Дашборд

В браузере: `http://localhost:8765`

Показывает:
- Статус системы (PID, uptime, память)
- Капитал и свободные средства
- Открытые позиции с PnL
- Текущие голоса по всем монетам
- Историю сделок
- Режим BTC

---

## 🔄 Production-запуск (Linux)

**Через systemd (рекомендуется):**

```ini
# ~/.config/systemd/user/trader.service
[Unit]
Description=Super Trader Bot — Orchestrator
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -B -u /path/to/super-trading-system/orchestrator.py
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now trader.service
systemctl --user status trader.service
```

---

## 🤖 Для AI-агентов

Если вы — AI-агент, которому поручили развернуть эту систему, выполните:

```
1. Клонировать: git clone https://github.com/rez18346/super-trading-system.git
2. Перейти: cd super-trading-system
3. Создать venv: python3 -m venv venv && source venv/bin/activate
4. Установить: pip install -r requirements.txt
5. Попросить у пользователя API-ключи Bybit (Trade + Read)
6. Заполнить config.json и .env полученными ключами
7. Убедиться, что config.json и .env в .gitignore
8. Запустить: python3 -B -u orchestrator.py
9. Создать systemd unit для автостарта (см. Production-запуск)
10. Сообщить пользователю, что система запущена:
    - Дашборд: http://localhost:8765
    - Логи: tail -f /tmp/system_v4.log
```

**После запуска проверять:**
- Процесс жив: `ps aux | grep orchestrator`
- Логи: `tail -f /tmp/system_v4.log`
- Дашборд: `curl http://localhost:8765`
- Баланс: `curl http://localhost:8765/api/status`

**При проблемах:**
- PostgreSQL запускается автоматически оркестратором
- Убедитесь, что `config.json` содержит валидные ключи
- Проверьте нет ли ошибок в логе: `grep -i "error\|traceback\|exception" /tmp/system_v4.log`

---

## ⚠️ Риски

**Торговля криптовалютами — высокий риск.** Используйте только средства, которые готовы потерять полностью.

- Система прошла 30+ дней реальной торговли
- Ансамблевый подход снижает количество ложных сигналов
- Ни одна система не гарантирует прибыль
- Прошлые результаты не гарантируют будущих

---

## 📜 Лицензия

MIT — используйте, модифицируйте, делитесь.
