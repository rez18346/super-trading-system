# 📋 ПЛАН ВНЕДРЕНИЯ ML-PRO — ПОНЕДЕЛЬНИК 27 АПРЕЛЯ

## 🎯 ЦЕЛЬ
Замена текущего ML-советника (RandomForest, 9 признаков, 62.8%) на ML-PRO 
(XGBoost, 26 признаков, 3 модуля, time-series CV)

## 🔧 ЧТО НАДО СДЕЛАТЬ

### Шаг 1: Подготовка (утро, 08:00-08:30)
1. ✅ ML-PRO модуль написан: `/home/ksysha/.openclaw/industrial_super_system/ml_professional.py`
2. 📌 Проверить что XGBoost импортируется и работает
3. 📌 Дообучить модель на исторических данных (9426 сделок)
4. 📌 Сохранить модель в `data/ml_pro_model.pkl`

### Шаг 2: Интеграция в industrial_trader.py (08:30-09:00)
1. 📌 Заменить `from ml_advisor import ml_evaluate` → `from ml_professional import ml_pro_evaluate`
2. 📌 Заменить все вызовы `ml_evaluate(...)` → `ml_pro_evaluate(...)`
3. 📌 Заменить `ml_add_result(...)` → `ml_pro_add_result(...)`
4. 📌 Заменить `ml_train()` → `ml_pro_train()`
5. 📌 Добавить новую логику: размер позиции от ML-PRO

### Шаг 3: Замена текущей модели (09:00-09:15)
```python
# Старая модель (ml_advisor.py) — оставляем как резерв
cp ml_advisor.py ml_advisor_backup.py

# Активируем ml_professional.py — в trading_cycle заменить:
from ml_professional import ml_pro_evaluate as ml_evaluate

# ИЛИ сделать плавный переход: если ml_pro не обучен → ml_advisor
```

### Шаг 4: Тестовый запуск (09:15-10:00)
1. 📌 Запустить систему с ML-PRO
2. 📌 Проверить первые 5-10 циклов: логируются ли все 3 модуля?
3. 📌 Проверить что regime detection совместим с сигналами
4. 📌 Проверить нет ли ошибок в _cached_fetch

### Шаг 5: Наблюдение (весь день)
1. 📌 Сравнить процент SKIP с прежним
2. 📌 Посмотреть feature importance после первого обучения
3. 📌 Если ML-PRO показывает <60% — fallback на старую модель

## 🛡️ PLAN B (если что-то пошло не так)
```bash
cd /home/ksysha/.openclaw/industrial_super_system
# Вернуть старую модель:
git checkout industrial_trader.py  # если был git
# Или вручную: заменить импорт
# from ml_professional import ml_pro_evaluate as ml_evaluate
# обратно на:
# from ml_advisor import ml_evaluate
```

## 📊 ОЖИДАЕМЫЕ УЛУЧШЕНИЯ
- Фильтр слабых сигналов: +5-10% win rate (62% → 67-70%)
- Защита от покупки на хае: через 15M тренд + distance to 24h max
- Адаптация под режим рынка: в bear не входит, в sideways меньше
- Умный размер позиции: больше в сильных сигналах, меньше в слабых

## ⚡ КРИТИЧЕСКИЕ МОМЕНТЫ
1. TimeSeriesSplit вместо shuffle — не перемешивать время!
2. _cached_fetch с TTL 60 сек — не забивать API лимиты Bybit
3. Early stopping — не переобучить на 9426 сделках
4. Feature names — строгий порядок, не менять после обучения
