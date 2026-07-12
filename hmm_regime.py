#!/usr/bin/env python3
"""
hmm_regime.py — HMM Market Regime Detector for ML-PRO v2
=========================================================
Определяет 3 режима рынка через HMM на данных BTC:
  0 = CALM    (низкая волатильность, трендовый)
  1 = NORMAL  (умеренная волатильность)
  2 = VOLATILE (высокая волатильность, хаотичный)

Использует hmmlearn (GaussianHMM, 3 компонента).
Признаки: % return + 14-часовая скользящая волатильность.

Подключается к ML-PRO v2 двумя способами:
  1. Как дополнительный признак (regime_score) в build_features_27f()
  2. Как динамический порог (BUY_THRESHOLD/SKIP_THRESHOLD)
"""

import numpy as np
import os
import json
import logging
from hmmlearn import hmm

log = logging.getLogger("HMM_REGIME")

STATE_NAMES = {0: 'CALM', 1: 'NORMAL', 2: 'VOLATILE'}
N_STATES = 3

# Пороги BUY/SKIP по режимам (для сигнала, не для обучения модели)
THRESHOLDS = {
    0: {'buy': 0.50, 'skip': 0.40},   # CALM — можно мягче
    1: {'buy': 0.55, 'skip': 0.45},   # NORMAL — стандартные
    2: {'buy': 0.65, 'skip': 0.55},   # VOLATILE — только уверенные
}


class RegimeDetector:
    """Обёртка над hmmlearn.GaussianHMM для режимов рынка."""

    def __init__(self, data_path='data/btc_usdt_1h.json', lookback=5000):
        self.model = None
        self.state_map = None  # волатильность → CALM/NORMAL/VOLATILE
        self.current_state = 1  # default NORMAL
        self.trained = False
        self.lookback = lookback

        # Пробуем загрузить или обучить
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'hmm_regime.pkl')
        if os.path.exists(model_path) and False:  # всегда переобучаем (пока нет сохранения)
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                self.model = data['model']
                self.state_map = data['state_map']
                self.trained = True
                log.info(f"[HMM] Загружена обученная модель")
            except Exception as e:
                log.warning(f"[HMM] Не загрузилась: {e}")

        if not self.trained:
            self.train(data_path)

        if self.trained:
            self.update_current_state()

    def _load_btc_data(self, data_path):
        """Загрузить BTC 1H данные."""
        full_path = os.path.join(os.path.dirname(__file__), data_path)
        if not os.path.exists(full_path):
            log.warning(f"[HMM] Нет данных BTC: {full_path}")
            return None

        with open(full_path) as f:
            data = json.load(f)

        if len(data) < 500:
            return None

        cl = np.array([c['c'] for c in data], float)
        returns = np.diff(cl) / cl[:-1] * 100

        # Волатильность (14 часов)
        vol = np.zeros(len(returns))
        for i in range(14, len(returns)):
            vol[i] = np.std(returns[i - 14:i])

        X = np.column_stack([returns.reshape(-1, 1)[14:], vol[14:].reshape(-1, 1)])

        # Берём последние lookback
        if len(X) > self.lookback:
            X = X[-self.lookback:]

        return X

    def train(self, data_path='data/btc_usdt_1h.json'):
        """Обучить HMM на BTC."""
        X = self._load_btc_data(data_path)
        if X is None or len(X) < 500:
            log.warning(f"[HMM] Недостаточно данных для обучения ({len(X) if X is not None else 0})")
            return

        try:
            self.model = hmm.GaussianHMM(
                n_components=N_STATES,
                covariance_type='diag',
                n_iter=100,
                tol=1e-4,
                random_state=42
            )
            self.model.fit(X)

            # Определяем порядок состояний по волатильности
            states = self.model.predict(X)
            vol_by_state = [X[states == s][:, 1].mean() for s in range(N_STATES)]
            order = np.argsort(vol_by_state)
            self.state_map = {order[0]: 0, order[1]: 1, order[2]: 2}  # low→CALM, mid→NORMAL, high→VOLATILE

            self.trained = True
            log.info(f"[HMM] ✅ Обучена на {len(X)} образцах BTC")
            log.info(f"[HMM]    CALM: vol≤{vol_by_state[order[0]]:.3f}%, NORMAL: vol≤{vol_by_state[order[1]]:.3f}%, VOLATILE: vol≤{vol_by_state[order[2]]:.3f}%")

        except Exception as e:
            log.error(f"[HMM] Ошибка обучения: {e}")

    def update_current_state(self):
        """Обновить текущий режим на основе последних данных."""
        if not self.trained or self.model is None:
            self.current_state = 1
            return 1

        # Берём последние 200 свечей для предсказания
        full_path = os.path.join(os.path.dirname(__file__), 'data/btc_usdt_1h.json')
        if not os.path.exists(full_path):
            return self.current_state

        try:
            with open(full_path) as f:
                data = json.load(f)

            last = data[-200:]
            cl = np.array([c['c'] for c in last], float)
            returns = np.diff(cl) / cl[:-1] * 100
            vol = np.zeros(len(returns))
            for i in range(14, len(returns)):
                vol[i] = np.std(returns[i - 14:i])
            X = np.column_stack([returns.reshape(-1, 1)[14:], vol[14:].reshape(-1, 1)])

            if len(X) < 10:
                return self.current_state

            states = self.model.predict(X)
            raw_state = states[-1]
            self.current_state = self.state_map.get(raw_state, 1)
            return self.current_state

        except Exception as e:
            log.warning(f"[HMM] update_current_state: {e}")
            return self.current_state

    def get_state_name(self):
        """Имя текущего режима."""
        return STATE_NAMES.get(self.current_state, 'UNKNOWN')

    def get_thresholds(self):
        """Пороги BUY/SKIP для текущего режима."""
        return THRESHOLDS.get(self.current_state, THRESHOLDS[1])

    def get_regime_score(self):
        """Численное значение: 0=CALM, 0.5=NORMAL, 1=VOLATILE (для ML)."""
        return self.current_state / (N_STATES - 1) if N_STATES > 1 else 0.5


# ─── Singleton ──────────────────────────────────────────────────────────
_instance = None


def get_regime():
    global _instance
    if _instance is None:
        _instance = RegimeDetector()
    return _instance


# ─── Self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    r = get_regime()
    print(f"Текущий режим: {r.get_state_name()} ({r.current_state})")
    print(f"Пороги: BUY≥{r.get_thresholds()['buy']}, SKIP<{r.get_thresholds()['skip']}")
    print(f"Regime score: {r.get_regime_score()}")
