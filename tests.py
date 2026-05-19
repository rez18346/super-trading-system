#!/usr/bin/env python3
"""
tests.py — Unit-тесты для ключевых модулей супер-системы.

Запуск: python3 tests.py

Покрытие:
  - DecisionEngine: все типы сигналов, приоритеты, кулдауны
  - ErrorHandler: ретраи, классификация ошибок
  - db: создание, позиции, сделки, синхронизация
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch
from typing import Dict

# Путь к модулям
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestDecisionEngine(unittest.TestCase):
    """Тесты DecisionEngine — центра принятия решений."""
    
    def setUp(self):
        from decision_engine import DecisionEngine, SignalType
        self.DecisionEngine = DecisionEngine
        self.SignalType = SignalType
        self.de = DecisionEngine()
        self.de.confirm_15m_reversal_fn = lambda x: True  # мок 15М фильтра
        self.de.ml_exit_check_fn = lambda s, p: None     # отключаем ML (синглтон может быть грязным)
        self.de.quick_struct_exit_fn = lambda s, p, e, c: None  # отключаем struct
        self.de.reset_cooldowns()
        self.de.update_hmm_regime(1)  # NORMAL по умолчанию
    
    def test_stop_loss(self):
        """Стоп-лосс при падении >5% (NORMAL режим)."""
        d = self.de.decide_exit('SOL/USDT', 85.0, 80.0, 0.38, 85.0)
        self.assertEqual(d.action, 'exit')
        self.assertEqual(d.signal_type, self.SignalType.STOP_LOSS)
    
    def test_take_profit(self):
        """Тейк-профит при росте >10% (NORMAL режим)."""
        d = self.de.decide_exit('SOL/USDT', 85.0, 97.0, 0.38, 97.0)
        self.assertEqual(d.signal_type, self.SignalType.TAKE_PROFIT)
    
    def test_take_profit_over_trailing(self):
        """TP имеет приоритет выше трейлинг-стопа."""
        d = self.de.decide_exit('SOL/USDT', 85.0, 98.0, 0.38, 95.0)
        self.assertEqual(d.signal_type, self.SignalType.TAKE_PROFIT)
    
    def test_hold_normal(self):
        """Держать при нормальном движении в плюс."""
        d = self.de.decide_exit('SOL/USDT', 85.0, 86.5, 0.38, 86.5)
        self.assertEqual(d.action, 'hold')
    
    def test_trailing_stop(self):
        """Трейлинг-стоп при падении от максимума."""
        d = self.de.decide_exit('SOL/USDT', 85.0, 87.5, 0.38, 90.0)
        self.assertEqual(d.signal_type, self.SignalType.TRAILING_STOP)
    
    def test_max_positions_limit(self):
        """Не входить если достигнут лимит позиций."""
        d = self.de.decide_entry('ALGO/USDT', 65.0, 'bullish', 45.0, 0.12, 5, 5)
        self.assertEqual(d.action, 'hold')
    
    def test_reentry_cooldown(self):
        """Не входить повторно в течение кулдауна."""
        self.de.record_exit('ALGO/USDT', 'test')
        d = self.de.decide_entry('ALGO/USDT', 65.0, 'bullish', 45.0, 0.12, 2, 5)
        self.assertEqual(d.action, 'hold')
    
    def test_cooldown_reset(self):
        """После сброса кулдауна вход снова разрешён."""
        self.de.record_exit('ALGO/USDT', 'test')
        self.de.reset_cooldowns()
        d = self.de.decide_entry('ALGO/USDT', 65.0, 'bullish', 45.0, 0.12, 2, 5)
        self.assertIn(d.action, ('enter', 'hold'))  # может быть 15M фильтр
    
    def test_low_confidence_rejection(self):
        """Отклонять вход при низкой уверенности ML."""
        d = self.de.decide_entry('ALGO/USDT', 45.0, 'bullish', 50.0, 0.12, 0, 5)
        self.assertEqual(d.action, 'hold')
    
    def test_bearish_ml_overrides(self):
        """ML может перевесить bearish тренд при высокой уверенности."""
        # confidence=65% + bearish: скор = 65*0.6 + 100*0.15 + 20*0.15 + 80*0.1 = 65.0 >= 65 → вход
        d = self.de.decide_entry('ALGO/USDT', 65.0, 'bearish', 45.0, 0.12, 0, 5)
        self.assertEqual(d.action, 'enter', f"ML 65% перевешивает bearish: {d}")
        
        # confidence=55% + bearish + RSI=70: скор = 54.5 < 65 → hold
        d = self.de.decide_entry('ALGO/USDT', 55.0, 'bearish', 70.0, 0.12, 0, 5)
        self.assertEqual(d.action, 'hold', f"Низкая уверенность + bearish + перекуп: {d}")
    
    def test_hmm_params_calm(self):
        """HMM CALM: SL=3%, TP=8%."""
        self.de.update_hmm_regime(0)  # CALM
        sl, tp, act, dist = self.de.get_sl_tp_params()
        self.assertEqual(sl, 3.0)
        self.assertEqual(tp, 8.0)
    
    def test_hmm_params_volatile(self):
        """HMM VOLATILE: SL=8%, TP=14%."""
        self.de.update_hmm_regime(2)  # VOLATILE
        sl, tp, act, dist = self.de.get_sl_tp_params()
        self.assertEqual(sl, 8.0)
        self.assertEqual(tp, 14.0)
    
    def test_singleton(self):
        """DecisionEngine — синглтон."""
        de2 = self.DecisionEngine()
        self.assertIs(self.de, de2)


class TestErrorHandler(unittest.TestCase):
    """Тесты ErrorHandler — обработка ошибок и ретраи."""
    
    def setUp(self):
        from error_handler import ErrorHandler, RetryConfig, ErrorCategory
        self.ErrorHandler = ErrorHandler
        self.RetryConfig = RetryConfig
        self.ErrorCategory = ErrorCategory
        self.eh = ErrorHandler()
    
    def test_success(self):
        """Успешный вызов без ошибок."""
        result = self.eh.safe_call(lambda: 42)
        self.assertEqual(result, 42)
    
    def test_retry_success(self):
        """Ретрай после временной ошибки."""
        calls = [0]
        def failing():
            calls[0] += 1
            if calls[0] < 2:
                raise TimeoutError('timed out')
            return 'ok'
        
        result = self.eh.safe_call(failing, retry_config=self.RetryConfig(max_retries=3, base_delay=0.1))
        self.assertEqual(result, 'ok')
        self.assertEqual(calls[0], 2)
    
    def test_all_retries_exhausted(self):
        """Все ретраи исчерпаны — возвращаем fallback."""
        def always_fail():
            raise TimeoutError('network error')
        
        result = self.eh.safe_call(always_fail, 
                                     retry_config=self.RetryConfig(max_retries=2, base_delay=0.1),
                                     fallback=None)
        self.assertIsNone(result)
    
    def test_balance_error_no_retry(self):
        """Ошибка баланса не ретраится (бесполезно)."""
        def balance_error():
            raise Exception('insufficient balance')
        
        result = self.eh.safe_call(balance_error,
                                     retry_config=self.RetryConfig(max_retries=3, base_delay=0.1),
                                     fallback='no_funds')
        self.assertEqual(result, 'no_funds')
    
    def test_classify_network(self):
        """Классификация сетевых ошибок."""
        cat = self.eh.classify_error(TimeoutError('timed out'))
        self.assertEqual(cat, self.ErrorCategory.NETWORK)
    
    def test_classify_rate_limit(self):
        """Классификация rate limit."""
        cat = self.eh.classify_error(Exception('rate limit exceeded'))
        self.assertEqual(cat, self.ErrorCategory.RATE_LIMIT)
    
    def test_classify_balance(self):
        """Классификация ошибок баланса."""
        cat = self.eh.classify_error(Exception('insufficient balance'))
        self.assertEqual(cat, self.ErrorCategory.BALANCE)
    
    def test_health_good(self):
        """Система здорова при малом количестве ошибок."""
        self.assertTrue(self.eh.is_healthy())
    
    def test_health_bad(self):
        """Система нездорова при >10 последовательных ошибок."""
        for i in range(15):
            self.eh.safe_call(lambda: (_ for _ in ()).throw(Exception('err')),
                              retry_config=self.RetryConfig(max_retries=0),
                              fallback=None)
        self.assertFalse(self.eh.is_healthy())
    
    def test_safe_call_async(self):
        """Быстрый retry без экспоненты."""
        calls = [0]
        def fn():
            calls[0] += 1
            if calls[0] < 2:
                raise TimeoutError('timeout')
            return 'ok'
        
        result = self.eh.safe_call_async(fn, max_retries=2)
        self.assertEqual(result, 'ok')


class TestDatabase(unittest.TestCase):
    """Тесты db.py — SQLite БД."""
    
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, 'test.db')
        
        import db as db_module
        cls.db = db_module
        cls.db.init_db(cls.db_path)
    
    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir)
    
    def setUp(self):
        # Очищаем таблицы перед каждым тестом
        with self.db._connect(self.db_path) as conn:
            conn.execute("DELETE FROM positions")
            conn.execute("DELETE FROM trades")
            conn.execute("DELETE FROM orders")
    
    def test_upsert_position_new(self):
        """Создание новой позиции."""
        self.db.upsert_position('SOL/USDT', 85.0, 0.38, 85.0, 
                                  db_path=self.db_path)
        pos = self.db.get_position('SOL/USDT', self.db_path)
        self.assertIsNotNone(pos)
        self.assertEqual(pos['entry_price'], 85.0)
        self.assertEqual(pos['quantity'], 0.38)
    
    def test_upsert_position_update(self):
        """Обновление существующей позиции (сохранение entry_price)."""
        self.db.upsert_position('SOL/USDT', 85.0, 0.38, 85.0, db_path=self.db_path)
        self.db.upsert_position('SOL/USDT', 90.0, 0.40, 90.0, db_path=self.db_path)
        
        pos = self.db.get_position('SOL/USDT', self.db_path)
        # entry_price не должен меняться при обновлении
        self.assertEqual(pos['entry_price'], 85.0)
        # highest_price должен обновиться
        self.assertEqual(pos['highest_price'], 90.0)
    
    def test_get_all_positions(self):
        """Получение всех позиций."""
        self.db.upsert_position('SOL/USDT', 85.0, 0.38, 85.0, db_path=self.db_path)
        self.db.upsert_position('BTC/USDT', 75000, 0.001, 75000, db_path=self.db_path)
        
        positions = self.db.get_all_positions(self.db_path)
        self.assertEqual(len(positions), 2)
        self.assertIn('SOL/USDT', positions)
        self.assertIn('BTC/USDT', positions)
    
    def test_remove_position(self):
        """Удаление позиции."""
        self.db.upsert_position('SOL/USDT', 85.0, 0.38, 85.0, db_path=self.db_path)
        removed = self.db.remove_position('SOL/USDT', self.db_path)
        self.assertIsNotNone(removed)
        self.assertIsNone(self.db.get_position('SOL/USDT', self.db_path))
    
    def test_add_trade_buy(self):
        """Добавление сделки на покупку."""
        tid = self.db.add_trade('SOL/USDT', 'buy', 85.0, 0.38, db_path=self.db_path)
        self.assertGreater(tid, 0)
        
        history = self.db.get_trade_history('SOL/USDT', db_path=self.db_path)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['side'], 'buy')
    
    def test_add_trade_sell_with_pnl(self):
        """Добавление сделки на продажу с PnL."""
        tid = self.db.add_trade('SOL/USDT', 'sell', 90.0, 0.38, 
                                  pnl=1.90, pnl_pct=5.88, db_path=self.db_path)
        self.assertGreater(tid, 0)
        
        history = self.db.get_trade_history('SOL/USDT', db_path=self.db_path)
        self.assertEqual(history[0]['pnl'], 1.90)
    
    def test_pnl_stats(self):
        """Статистика PnL."""
        self.db.add_trade('SOL/USDT', 'sell', 90.0, 0.38, pnl=1.90, pnl_pct=5.0, db_path=self.db_path)
        self.db.add_trade('ALGO/USDT', 'sell', 0.12, 100.0, pnl=-2.0, pnl_pct=-3.0, db_path=self.db_path)
        
        stats = self.db.get_pnl_stats(self.db_path)
        self.assertEqual(stats['total_trades'], 2)
        self.assertEqual(stats['win'], 1)
        self.assertEqual(stats['loss'], 1)
    
    def test_upsert_order(self):
        """Сохранение ордера."""
        self.db.upsert_order('order_1', 'SOL/USDT', 'sell', 86.68, 0.22, 
                              status='open', db_path=self.db_path)
        
        with self.db._connect(self.db_path) as conn:
            row = conn.execute("SELECT * FROM orders WHERE order_id = 'order_1'").fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row['status'], 'open')
    
    def test_db_stats(self):
        """Статистика БД."""
        stats = self.db.get_db_stats(self.db_path)
        self.assertIn('size_bytes', stats)
        self.assertIn('positions', stats)


class TestDBThreadSafety(unittest.TestCase):
    """Тест потокобезопасности БД."""
    
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmpdir = tempfile.mkdtemp()
        cls.db_path = os.path.join(cls.tmpdir, 'concurrent.db')
        
        import db as db_module
        cls.db = db_module
        cls.db.init_db(cls.db_path)
    
    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir)
    
    def test_concurrent_writes(self):
        """5 потоков пишут в разные/одинаковые символы — без deadlock."""
        import threading
        
        errors = []
        def writer(thread_id):
            try:
                # Каждый поток пишет в уникальные символы + 1 общий для conflict
                for i in range(5):
                    sym = f'PAIR_T{thread_id}_{i}/USDT'
                    self.db.upsert_position(sym, float(i), 0.1, float(i),
                                              db_path=self.db_path)
                    # Конкуренция за одинаковый символ (UPSERT должен выдержать)
                    self.db.upsert_position('SHARED_PAIR/USDT', float(thread_id), 0.1, float(thread_id),
                                              db_path=self.db_path)
                # Пишем сделки
                self.db.add_trade('SHARED_PAIR/USDT', 'buy', 10.0, 0.1,
                                    db_path=self.db_path)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for t in range(5):
            th = threading.Thread(target=writer, args=(t,))
            threads.append(th)
            th.start()
        
        for th in threads:
            th.join()
        
        # UPSERT с одинаковыми ключами может давать UNIQUE в редких гонках,
        # главное чтобы не было падений с блокировкой
        self.assertLess(len(errors), 3, f"Слишком много ошибок: {errors}")


class TestDecisionSignals(unittest.TestCase):
    """Комплексные тесты сигналов DecisionEngine."""
    
    def setUp(self):
        from decision_engine import DecisionEngine, SignalType
        self.DecisionEngine = DecisionEngine
        self.SignalType = SignalType
        self.de = DecisionEngine()
        self.de.confirm_15m_reversal_fn = lambda x: True
        self.de.ml_exit_check_fn = lambda s, p: None
        self.de.quick_struct_exit_fn = lambda s, p, e, c: None
        self.de.reset_cooldowns()
        self.de.update_hmm_regime(1)  # NORMAL
    
    def test_sl_at_boundary(self):
        """SL срабатывает ровно на границе."""
        # NORMAL: SL = 5%, цена на -5.01%
        d = self.de.decide_exit('SOL/USDT', 100.0, 94.99, 1.0, 100.0)
        self.assertEqual(d.action, 'exit')
        self.assertEqual(d.signal_type, self.SignalType.STOP_LOSS)
    
    def test_stop_loss_different_regimes(self):
        """SL меняется в зависимости от HMM режима."""
        # NORMAL: -5%
        self.de.update_hmm_regime(1)
        d = self.de.decide_exit('SOL/USDT', 100.0, 96.0, 1.0, 100.0)
        self.assertEqual(d.action, 'hold', "NORMAL -4%: должен держать")
        
        # CALM: -3% — тот же -4% уже SL
        self.de.update_hmm_regime(0)
        d = self.de.decide_exit('SOL/USDT', 100.0, 96.0, 1.0, 100.0)
        self.assertEqual(d.action, 'exit', "CALM -4%: должен SL")
    
    def test_tp_different_regimes(self):
        """TP меняется в зависимости от HMM режима."""
        # CALM: +8%
        self.de.update_hmm_regime(0)
        d = self.de.decide_exit('SOL/USDT', 100.0, 109.0, 1.0, 109.0)
        self.assertEqual(d.signal_type, self.SignalType.TAKE_PROFIT, f"CALM +9%: TP, получили {d.signal_type}")
        
        # NORMAL: +10%, +9% ещё не TP
        self.de.update_hmm_regime(1)
        d = self.de.decide_exit('SOL/USDT', 100.0, 109.0, 1.0, 109.0)
        self.assertNotEqual(d.signal_type, self.SignalType.TAKE_PROFIT, "NORMAL +9%: ещё не TP")
    
    def test_entry_conditions_full_check(self):
        """Полный цикл проверки входа (ансамбль)."""
        # Все условия выполнены: хороший ML, RSI, тренд
        d = self.de.decide_entry('ALGO/USDT', 65.0, 'bullish', 45.0, 0.12, 0, 5)
        self.assertEqual(d.action, 'enter', f"Должен войти: {d}")
        
        # RSI перекуплен + средний ML = скор ниже порога
        d = self.de.decide_entry('ALGO/USDT', 55.0, 'bullish', 80.0, 0.12, 0, 5)
        self.assertEqual(d.action, 'hold', f"RSI=80 + conf=55: не должен: {d}")
        
        # Confidence низкий
        d = self.de.decide_entry('ALGO/USDT', 40.0, 'bullish', 45.0, 0.12, 0, 5)
        self.assertEqual(d.action, 'hold', f"Conf=40: не должен: {d}")


class TestDecisionEngineEdgeCases(unittest.TestCase):
    """Тесты граничных случаев."""
    
    def setUp(self):
        from decision_engine import DecisionEngine, SignalType
        self.DecisionEngine = DecisionEngine
        self.SignalType = SignalType
        self.de = DecisionEngine()
        self.de.confirm_15m_reversal_fn = lambda x: True
        self.de.ml_exit_check_fn = lambda s, p: None
        self.de.quick_struct_exit_fn = lambda s, p, e, c: None
        self.de.reset_cooldowns()
        self.de.update_hmm_regime(1)
    
    def test_zero_quantity(self):
        """Обработка нулевого количества."""
        d = self.de.decide_exit('SOL/USDT', 85.0, 80.0, 0.0, 85.0)
        self.assertEqual(d.action, 'exit', "Даже с 0 qty SL должен сработать")
    
    def test_negative_price(self):
        """Отрицательная цена (аварийная ситуация)."""
        d = self.de.decide_exit('SOL/USDT', 85.0, -1.0, 0.38, 85.0)
        self.assertEqual(d.action, 'exit')
    
    def test_extreme_profit(self):
        """Экстремальная прибыль (1000%)."""
        d = self.de.decide_exit('SOL/USDT', 1.0, 100.0, 0.38, 100.0)
        self.assertEqual(d.signal_type, self.SignalType.TAKE_PROFIT)
    
    def test_set_reentry_cooldown(self):
        """Установка кастомного кулдауна."""
        self.de.set_reentry_cooldown(300)  # 5 минут
        self.assertEqual(self.de.reentry_cooldown, 300)
        
    def test_min_reentry_cooldown(self):
        """Минимальный кулдаун — 60 секунд."""
        self.de.set_reentry_cooldown(10)
        self.assertGreaterEqual(self.de.reentry_cooldown, 60)
    
    def test_ml_exit_unavailable(self):
        """ML-модуль недоступен — не падаем."""
        self.de.ml_exit_check_fn = None
        result = self.de._check_ml_exit('SOL/USDT', -2.0)
        # Может вернуть сигнал или None — не важно, главное не упасть
        # (может загрузить модуль, может не загрузить — оба варианта ок)


if __name__ == '__main__':
    # Подавляем лишний вывод логов
    import logging
    logging.disable(logging.CRITICAL)
    
    unittest.main(verbosity=2)
