# 🏭 Супер-Система — РАБОЧИЙ ТРЕЙДЕР

**Папка:** ~/new_trader/
**Дата:** 2026-06-13

## Как запускается
- Трейдер: `super_trader.py` → `industrial_trader.py` (в screen `trader`)
- Дашборд: `control_api.py` (в screen `terminal`, порт 8765)

## Запуск вручную
```bash
cd ~/new_trader

# Трейдер
screen -dmS trader bash -c "while true; do python3 -B -u super_trader.py 2>&1 | tee -a /tmp/super_trader.log; sleep 5; done"

# Дашборд
screen -dmS terminal python3 -B -u control_api.py
```

## Структура
- `super_trader.py` — точка входа трейдера
- `industrial_trader.py` — ядро торговой системы
- `control_api.py` — дашборд (порт 8765)
- `decision_engine.py` — модуль принятия решений
- `ml_professional_v2.py` — ML Pro v2 с 27f моделью
- Остальные модули: liquidity, VSA, CVD, ML-Advisor, RP и т.д.

## Бэкапы старых версий
Стеняты в ~/backup_traders/
