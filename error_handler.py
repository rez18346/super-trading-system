#!/usr/bin/env python3
"""
error_handler.py — Надёжная обёртка для ccxt с ретраями, логированием и классификацией ошибок.

Профессиональная торговая система не должна падать от:
  - Таймаута сети → ретрай
  - Rate limit → экспоненциальная задержка
  - Временной ошибки API → ретрай
  - Недостатка баланса → специфичная обработка

Принцип: внешние вызовы делаются ТОЛЬКО через ErrorHandler.
"""

import logging
import time
import random
from typing import Callable, Any, Optional, Dict, Tuple
from enum import Enum

logger = logging.getLogger('error_handler')


class ErrorCategory(Enum):
    """Категории ошибок для разной стратегии обработки."""
    NETWORK = 0       # Таймаут, соединение — ретрай с эксп. задержкой
    RATE_LIMIT = 1    # 429 — подождать
    BALANCE = 2       # Недостаточно средств — не ретраить, логировать
    NOT_FOUND = 3     # 404 — не ретраить
    INVALID = 4       # Неверные параметры — не ретраить
    UNKNOWN = 5       # Всё остальное — ретрай + алерт


class RetryConfig:
    """Конфигурация повторных попыток."""
    
    def __init__(self, max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 30.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter


class ErrorHandler:
    """
    Единый обработчик ошибок для всех внешних вызовов.
    
    Пример использования:
    
        handler = ErrorHandler()
        balance = handler.safe_call(
            exchange.fetch_balance,
            retry_config=RetryConfig(max_retries=3)
        )
        if balance is None:
            # Обработка ошибки
            pass
    """
    
    def __init__(self):
        self.error_count = 0
        self.consecutive_errors = 0
        self.last_error_time = 0
        self.error_history: list = []
        self.max_history = 50
    
    def classify_error(self, exception: Exception) -> ErrorCategory:
        """Классифицировать ошибку для выбора стратегии обработки."""
        error_msg = str(exception).lower()
        error_type = type(exception).__name__
        
        # Сетевые ошибки
        if any(word in error_msg for word in [
            'timeout', 'timed out', 'connection', 'econnreset',
            'econnrefused', 'network', 'unreachable', 'dns',
        ]):
            return ErrorCategory.NETWORK
        
        # Rate limit
        if any(word in error_msg for word in [
            'rate limit', '429', 'too many requests', 'ddos',
        ]):
            return ErrorCategory.RATE_LIMIT
        
        # Баланс
        if any(word in error_msg for word in [
            'insufficient balance', 'insufficient funds',
            'not enough', 'account has insufficient',
        ]):
            return ErrorCategory.BALANCE
        
        # Не найдено
        if any(word in error_msg for word in [
            'not found', '404', 'symbol not found', 'market not found',
        ]):
            return ErrorCategory.NOT_FOUND
        
        # Неверные параметры
        if any(word in error_msg for word in [
            'invalid', 'bad request', 'parameter', 'required',
            'precision', 'min notional', 'minimum amount',
        ]):
            return ErrorCategory.INVALID
        
        # Всё остальное — неизвестно
        return ErrorCategory.UNKNOWN
    
    def safe_call(self, fn: Callable, *args,
                  retry_config: RetryConfig = None,
                  fallback: Any = None,
                  **kwargs) -> Any:
        """
        Безопасный вызов внешней функции с ретраем.
        
        Args:
            fn: функция для вызова
            *args: позиционные аргументы для fn
            retry_config: конфигурация ретраев (None = не ретраить)
            fallback: значение при полном провале
            **kwargs: именованные аргументы для fn
        
        Returns:
            Результат fn или fallback при ошибке
        """
        if retry_config is None:
            retry_config = RetryConfig()
        
        last_error = None
        
        for attempt in range(retry_config.max_retries + 1):
            try:
                result = fn(*args, **kwargs)
                self.consecutive_errors = 0
                return result
            
            except Exception as e:
                last_error = e
                category = self.classify_error(e)
                
                # Обновляем статистику
                self.error_count += 1
                self.consecutive_errors += 1
                self.last_error_time = time.time()
                
                # Логируем
                self._log_error(fn.__name__, e, category, attempt, retry_config.max_retries)
                
                # Критические ошибки — не ретраим
                if category in (ErrorCategory.BALANCE, ErrorCategory.INVALID, ErrorCategory.NOT_FOUND):
                    # Ретраем один раз, но логируем как критическое
                    if attempt >= 1:
                        logger.warning(f"❌ Критическая ошибка {category.name}: {e}")
                        return fallback
                
                # Rate limit — ждём дольше
                if category == ErrorCategory.RATE_LIMIT:
                    wait = min(retry_config.base_delay * 5, retry_config.max_delay)
                    if attempt < retry_config.max_retries:
                        logger.warning(f"⏳ Rate limit, жду {wait:.1f}с...")
                        time.sleep(wait)
                    continue
                
                # Сетевые ошибки — ретраим с экспонентой
                if attempt < retry_config.max_retries:
                    delay = min(
                        retry_config.base_delay * (retry_config.exponential_base ** attempt),
                        retry_config.max_delay
                    )
                    if retry_config.jitter:
                        delay *= 0.5 + random.random() * 0.5  # 50-150%
                    
                    logger.info(f"🔄 Ретрай {attempt+1}/{retry_config.max_retries} через {delay:.1f}с...")
                    time.sleep(delay)
                else:
                    logger.error(f"❌ Все {retry_config.max_retries} ретраев исчерпаны: {e}")
        
        return fallback
    
    def safe_call_async(self, fn: Callable, *args,
                         max_retries: int = 3,
                         fallback: Any = None,
                         **kwargs) -> Any:
        """
        Версия без экспоненциальной задержки для быстрых операций.
        """
        config = RetryConfig(max_retries=max_retries, base_delay=0.5, max_delay=5)
        return self.safe_call(fn, *args, retry_config=config, fallback=fallback, **kwargs)
    
    def _log_error(self, fn_name: str, error: Exception,
                    category: ErrorCategory, attempt: int, max_retries: int) -> None:
        """Логировать ошибку с категорией."""
        entry = {
            'time': time.time(),
            'function': fn_name,
            'error': str(error)[:200],
            'error_type': type(error).__name__,
            'category': category.name,
            'attempt': attempt,
        }
        self.error_history.append(entry)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]
        
        if attempt == 0:
            # Первая ошибка — логируем
            logger.debug(f"[{category.name}] {fn_name}: {str(error)[:120]}")
        elif attempt < max_retries:
            logger.warning(f"[{category.name}] {fn_name} (попытка {attempt+1}): {str(error)[:120]}")
        else:
            logger.error(f"[{category.name}] {fn_name} (все ретраи): {str(error)[:200]}")
    
    def get_error_stats(self) -> Dict:
        """Вернуть статистику ошибок."""
        categories = {}
        for entry in self.error_history[-20:]:
            cat = entry['category']
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_errors': self.error_count,
            'consecutive_errors': self.consecutive_errors,
            'last_error_ago': time.time() - self.last_error_time if self.last_error_time else -1,
            'recent_categories': categories,
        }
    
    def is_healthy(self) -> bool:
        """
        Проверить здоровье системы.
        False если слишком много последовательных ошибок.
        """
        # Если больше 10 последовательных ошибок — что-то не так
        if self.consecutive_errors > 10:
            return False
        # Если последняя ошибка была меньше 1 секунды назад и их больше 5
        if (self.last_error_time and 
            time.time() - self.last_error_time < 1.0 and 
            self.consecutive_errors > 5):
            return False
        return True
