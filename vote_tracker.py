#!/usr/bin/env python3
"""
vote_tracker.py — Взвешенное голосование модулей DecisionEngine.

Отслеживает точность каждого модуля после каждой закрытой сделки
и динамически корректирует веса при принятии решений.

Архитектура:
  1. При ВХОДЕ в сделку — сохраняем голоса всех модулей
  2. При ЗАКРЫТИИ сделки — сравниваем голоса с результатом (PnL)
  3. Каждый модуль получает accuracy = верных прогнозов / всего прогнозов
  4. Вес модуля = max(0.5, accuracy * 2) — нормируется чтоб сумма = 1.0

Файл данных: data/vote_tracker.json
"""

import json
import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger('vote_tracker')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'vote_tracker.json')

# Пороги для определения "голос ЗА вход" по каждому модулю
# (какой score считается, что модуль поддерживает вход)
THRESHOLDS = {
    'ml_v2': 50,       # BUY/WEAK_BUY
    'advisor': 60,     # GOOD/WEAK
    'liquidity': 70,   # высокая ликвидность
    'volume_vwap': 55, # объёмный сигнал
    'vsa': 65,         # bullish дивергенция
    'cvd': 55,         # положительная дельта
    'mtf': 50,         # информационно
    'rsi_vol_btc': 50,
}

# Начальные веса (как в decision_engine.py BASE_WEIGHTS)
DEFAULT_WEIGHTS = {
    'ml_v2': 0.10,
    'advisor': 0.25,
    'mtf': 0.00,
    'rsi_vol_btc': 0.00,
    'liquidity': 0.25,
    'volume_vwap': 0.10,
    'vsa': 0.20,
    'cvd': 0.10,
}

# Модули, которые участвуют в голосовании (имеют вес > 0)
ACTIVE_MODULES = ['ml_v2', 'advisor', 'liquidity', 'volume_vwap', 'vsa', 'cvd']


class VoteTracker:
    """Отслеживает точность модулей DecisionEngine и корректирует веса."""

    def __init__(self):
        self.accuracy_stats: Dict[str, dict] = {}
        self._pending_entries: Dict[str, dict] = {}  # symbol → entry votes snapshot
        self._initialized = False
        self._load()

    # ═══════════ ЗАГРУЗКА / СОХРАНЕНИЕ ═══════════════════════════════

    def _load(self):
        """Загружает статистику из файла."""
        try:
            if os.path.exists(DATA_FILE):
                with open(DATA_FILE, 'r') as f:
                    data = json.load(f)
                self.accuracy_stats = data.get('stats', {})
                logger.info(f"[VoteTracker] загружена статистика: {len(self.accuracy_stats)} модулей")
            else:
                self._init_defaults()
            self._initialized = True
        except Exception as e:
            logger.warning(f"[VoteTracker] ошибка загрузки: {e}")
            self._init_defaults()
            self._initialized = True

    def _init_defaults(self):
        """Инициализирует пустую статистику."""
        self.accuracy_stats = {}
        for mod in ACTIVE_MODULES:
            self.accuracy_stats[mod] = {
                'total': 0, 'correct': 0, 'accuracy': 0.50,
                'weight': DEFAULT_WEIGHTS.get(mod, 0.10)
            }

    def _save(self):
        """Сохраняет статистику в файл."""
        try:
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
            tmp = DATA_FILE + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'stats': self.accuracy_stats}, f, indent=2, default=str)
            os.replace(tmp, DATA_FILE)
        except Exception as e:
            logger.debug(f"[VoteTracker] ошибка сохранения: {e}")

    # ═══════════ ЗАПИСЬ ПРИ ВХОДЕ ═══════════════════════════════════

    def record_entry_votes(self, symbol: str, votes: dict, entry_price: float,
                           entry_ts: str):
        """Сохраняет голоса модулей при входе в сделку.

        Args:
            symbol: тикер (напр. "MNT/USDT")
            votes: словарь голосов от decision_engine (поля module→{score, detail})
            entry_price: цена входа
            entry_ts: время входа (ISO)
        """
        snap = {
            'symbol': symbol,
            'price': entry_price,
            'ts': entry_ts,
            'votes': {},
        }
        for mod in ACTIVE_MODULES:
            if mod in votes:
                v = votes[mod]
                snap['votes'][mod] = {
                    'score': v.get('score', 50),
                    'detail': v.get('detail', ''),
                    'voted_for': v.get('score', 50) >= THRESHOLDS.get(mod, 50),
                }
            else:
                snap['votes'][mod] = {
                    'score': 50, 'detail': 'N/A', 'voted_for': False,
                }
        self._pending_entries[symbol] = snap
        logger.debug(f"[VoteTracker] сохранены голоса для {symbol} @ ${entry_price}")

    # ═══════════ ЗАПИСЬ РЕЗУЛЬТАТА ══════════════════════════════════

    def record_outcome(self, symbol: str, pnl_pct: float):
        """Записывает результат сделки и обновляет точность модулей.

        Args:
            symbol: тикер (напр. "MNT/USDT")
            pnl_pct: PnL в процентах (положительный = профит)
        """
        if symbol not in self._pending_entries:
            logger.debug(f"[VoteTracker] {symbol}: нет сохранённых голосов для входа")
            return

        snap = self._pending_entries.pop(symbol)
        success = pnl_pct > 0  # сделка профитная?

        updated = []
        for mod, vote_data in snap['votes'].items():
            if mod not in self.accuracy_stats:
                continue
            stats = self.accuracy_stats[mod]
            voted_for = vote_data.get('voted_for', False)

            # Модуль угадал, если:
            #   - голосовал ЗА вход И сделка профитная, ИЛИ
            #   - голосовал ПРОТИВ входа И сделка убыточная
            correct = (voted_for and success) or (not voted_for and not success)

            stats['total'] = stats.get('total', 0) + 1
            if correct:
                stats['correct'] = stats.get('correct', 0) + 1

            updated.append(f"{mod}:{'✅' if correct else '❌'} (voted={'FOR' if voted_for else 'AGAINST'}, result={'WIN' if success else 'LOSS'})")

        # Пересчитываем accuracy и веса
        self._recalculate_weights()
        self._save()

        logger.info(
            f"[VoteTracker] {symbol}: PnL={pnl_pct:+.2f}% ({'✅WIN' if success else '❌LOSS'}) | "
            + ' | '.join(updated)
        )

    # ═══════════ ПЕРЕСЧЁТ ВЕСОВ ═════════════════════════════════════

    def _recalculate_weights(self):
        """Пересчитывает accuracy и веса для всех модулей."""
        # 1. Считаем accuracy
        for mod in ACTIVE_MODULES:
            stats = self.accuracy_stats.get(mod, {'total': 0, 'correct': 0})
            total = stats.get('total', 0)
            correct = stats.get('correct', 0)
            if total > 0:
                accuracy = correct / total
            else:
                accuracy = 0.50  # нейтральное начальное значение
            stats['accuracy'] = round(accuracy, 4)
            stats['total'] = total
            stats['correct'] = correct
            self.accuracy_stats[mod] = stats

        # 2. Считаем веса: weight = max(0.1, accuracy * 2)
        # Для модулей без данных оставляем дефолтный вес
        raw_weights = {}
        for mod in ACTIVE_MODULES:
            stats = self.accuracy_stats.get(mod, {})
            total = stats.get('total', 0)
            acc = stats.get('accuracy', 0.50)
            if total >= 5:  # нужно минимум 5 примеров для изменения веса
                raw_weights[mod] = max(0.10, min(0.50, acc * 2))
            else:
                raw_weights[mod] = DEFAULT_WEIGHTS.get(mod, 0.10)

        # 3. Нормируем, чтобы сумма = исходной сумме активных весов
        #    (0.10+0.25+0.10+0.20+0.10 = 0.75)
        target_sum = sum(DEFAULT_WEIGHTS.get(mod, 0) for mod in ACTIVE_MODULES)
        raw_sum = sum(raw_weights.values())
        if raw_sum > 0:
            for mod in ACTIVE_MODULES:
                norm_weight = (raw_weights[mod] / raw_sum) * target_sum
                self.accuracy_stats[mod]['weight'] = round(norm_weight, 4)

    def get_weights(self) -> Dict[str, float]:
        """Возвращает текущие скорректированные веса."""
        weights = {}
        for mod in ACTIVE_MODULES:
            stats = self.accuracy_stats.get(mod, {})
            weights[mod] = stats.get('weight', DEFAULT_WEIGHTS.get(mod, 0.10))
        # Добавляем модули с нулевым весом
        for mod in ['mtf', 'rsi_vol_btc']:
            weights[mod] = DEFAULT_WEIGHTS.get(mod, 0.0)
        return weights

    def get_summary(self) -> Dict[str, dict]:
        """Возвращает читаемую сводку по модулям."""
        summary = {}
        for mod in ACTIVE_MODULES:
            stats = self.accuracy_stats.get(mod, {})
            total = stats.get('total', 0)
            correct = stats.get('correct', 0)
            acc = stats.get('accuracy', 0.50)
            w = stats.get('weight', DEFAULT_WEIGHTS.get(mod, 0.10))
            summary[mod] = {
                'total': total,
                'correct': correct,
                'accuracy': round(acc * 100, 1),
                'weight': round(w, 3),
                'default_weight': DEFAULT_WEIGHTS.get(mod, 0.10),
            }
        return summary

    def get_pending_count(self) -> int:
        """Сколько сделок ожидают закрытия для учёта."""
        return len(self._pending_entries)


# ═══════════ ГЛОБАЛЬНЫЙ СИНГЛТОН ═══════════════════════════════════

_vote_tracker: Optional[VoteTracker] = None


def get_tracker() -> VoteTracker:
    """Возвращает глобальный экземпляр VoteTracker."""
    global _vote_tracker
    if _vote_tracker is None:
        _vote_tracker = VoteTracker()
    return _vote_tracker
