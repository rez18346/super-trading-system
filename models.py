#!/usr/bin/env python3
"""
models.py - Единые модели данных для всей торговой системы.

Архитектура:
  Все сущности системы — dataclasses. Никаких dict-позиций, никаких
  размазанных по файлам структур. Один source of truth.

  OrderManager   →  Order, OrderResult, OrderStatus
  PositionManager → Position, PositionFilter
  RiskManager    →  RiskConfig, ApprovalResult, BtcState
  DecisionEngine →  Decision, ModuleSignal, SignalDirection

  direction = LONG | SHORT     — направление сделки
  mode      = SPOT | MARGIN | FUTURES — тип торговли (заложено сразу)
  status    = PENDING | OPEN | CLOSED | CANCELED — lifecycle
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any


# ─── Перечисления ────────────────────────────────────────────────────────────

class Direction(Enum):
    """Направление сделки."""
    LONG = "long"
    SHORT = "short"

    def invert(self) -> Direction:
        return Direction.SHORT if self == Direction.LONG else Direction.LONG

    @property
    def is_long(self) -> bool:
        return self == Direction.LONG

    @property
    def is_short(self) -> bool:
        return self == Direction.SHORT


class Mode(Enum):
    """Тип торговли (расширяется под плечо позже)."""
    SPOT = "spot"
    MARGIN = "margin"    # спот с плечом
    FUTURES = "futures"  # фьючерсы

    @property
    def has_leverage(self) -> bool:
        return self in (Mode.MARGIN, Mode.FUTURES)

    @property
    def is_spot(self) -> bool:
        return self == Mode.SPOT


class OrderStatus(Enum):
    """Статус ордера на бирже."""
    PENDING = "pending"        # создан, ждёт исполнения
    OPEN = "open"              # в книге ордеров
    PARTIAL = "partial"        # частично исполнен
    CLOSED = "closed"          # полностью исполнен
    CANCELED = "canceled"      # отменён
    REJECTED = "rejected"      # отклонён биржей
    EXPIRED = "expired"        # истёк
    FAILED = "failed"          # ошибка при создании


class Action(Enum):
    """Действие, которое может принять DecisionEngine."""
    ENTER = "enter"
    EXIT = "exit"
    HOLD = "hold"

    @property
    def is_entry(self) -> bool:
        return self == Action.ENTER

    @property
    def is_exit(self) -> bool:
        return self == Action.EXIT

    @property
    def is_hold(self) -> bool:
        return self == Action.HOLD


class SignalDirection(Enum):
    """Направление сигнала от модуля."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

    @property
    def is_bullish(self) -> bool:
        return self == SignalDirection.BULLISH

    @property
    def is_bearish(self) -> bool:
        return self == SignalDirection.BEARISH

    @property
    def is_neutral(self) -> bool:
        return self == SignalDirection.NEUTRAL

    def matches(self, direction: Direction) -> bool:
        """Совпадает ли сигнал с направлением сделки."""
        if self == SignalDirection.NEUTRAL:
            return False
        return (self == SignalDirection.BULLISH and direction == Direction.LONG) or \
               (self == SignalDirection.BEARISH and direction == Direction.SHORT)


class ExitOverride(Enum):
    """Как умный выход должен поступить."""
    EXIT = "exit"                # выходим
    HOLD_WIDEN_SL = "hold_widen_sl"  # держим, расширяем SL
    HOLD_TIGHTEN_SL = "hold_tighten_sl"  # держим, подтягиваем SL
    NONE = "none"                # ensemble не дал решения


# ─── Данные модулей ──────────────────────────────────────────────────────────

@dataclass
class ModuleSignal:
    """Сигнал от одного аналитического модуля (VSA, CVD, VV, Liq, ML-Pro, Advisor)."""
    module: str                          # 'vsa' | 'cvd' | 'vv' | 'liq' | 'ml_pro' | 'advisor'
    direction: SignalDirection           # BULLISH | BEARISH | NEUTRAL
    strength: float = 0.0                # 0.0-1.0 сила сигнала
    confidence: float = 0.0              # 0-100 уверенность (процент)
    weight: float = 1.0                  # вес модуля в ensemble
    detail: str = ""                     # детали (для логов)

    def score_for_direction(self, direction: Direction) -> float:
        """Вклад модуля в итоговый скор для указанного направления."""
        if self.direction == SignalDirection.NEUTRAL:
            return self.confidence * 0.3  # нейтрал даёт половину
        if self.matches(direction):
            return self.confidence
        return 0.0

    def matches(self, direction: Direction) -> bool:
        """Совпадает ли сигнал с направлением сделки."""
        if self.direction == SignalDirection.NEUTRAL:
            return False
        return (self.direction == SignalDirection.BULLISH and direction == Direction.LONG) or \
               (self.direction == SignalDirection.BEARISH and direction == Direction.SHORT)

    def __repr__(self) -> str:
        return f"{self.module}:{self.direction.value}(str={self.strength:.2f},conf={self.confidence:.0f})"


# ─── Ордер ───────────────────────────────────────────────────────────────────

@dataclass
class Order:
    """Полная запись об ордере.

    Создаётся OrderManager.submit_order().
    Статус обновляется синхронизацией с биржей.
    """
    symbol: str
    side: str                           # 'buy' | 'sell' — как у биржи
    direction: Direction                # LONG | SHORT — наша интерпретация
    quantity: float
    price: float                        # market price или лимит
    order_type: str = "market"          # 'market' | 'limit'
    order_id: str = ""                  # ID биржи
    client_id: str = ""                 # наш internal ID
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_price: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)  # scores, reason и т.д.

    # Связь с позицией (заполняется после исполнения)
    position_id: Optional[str] = None

    @property
    def is_filled(self) -> bool:
        return self.status in (OrderStatus.CLOSED, OrderStatus.PARTIAL)

    @property
    def is_open(self) -> bool:
        return self.status in (OrderStatus.OPEN, OrderStatus.PENDING)

    @property
    def is_failed(self) -> bool:
        return self.status in (OrderStatus.REJECTED, OrderStatus.CANCELED,
                               OrderStatus.EXPIRED, OrderStatus.FAILED)

    @property
    def is_buy(self) -> bool:
        return self.side == 'buy'

    @property
    def is_sell(self) -> bool:
        return self.side == 'sell'


@dataclass
class OrderResult:
    """Результат отправки ордера на биржу."""
    success: bool
    order: Optional[Order] = None
    error: Optional[str] = None
    retryable: bool = False  # можно ли повторить


# ─── Позиция ─────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """Единая модель открытой позиции.

    Заменяет dict-позиции из industrial_trader.py.
    Одно хранилище для LONG и SHORT.
    Отличается от Order тем, что это активная удерживаемая позиция.

    SL/TP/трейлинг — поля, а не _sl_price в dict.
    leverage=1 для спота.
    """
    symbol: str
    direction: Direction                 # LONG | SHORT
    mode: Mode = Mode.SPOT               # SPOT | MARGIN | FUTURES

    # Размер и цена
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0           # обновляется каждый цикл
    leverage: int = 1

    # Мета-данные для маржи/фьючерсов
    margin_used: float = 0.0
    liquidation_price: Optional[float] = None
    margin_mode: str = "isolated"        # 'isolated' | 'cross'

    # SL/TP (явно заданные или None)
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trail_activation: Optional[float] = None  # % от entry для активации
    trail_distance: Optional[float] = None    # % отката от пика

    # Трейлинг runtime
    highest_price: float = 0.0
    lowest_price: float = 0.0            # для шортов (SL когда цена растёт)
    max_profit_pct: float = 0.0          # максимальный PnL в %

    # Время
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: float = field(default_factory=time.time)  # timestamp для sync-защиты

    # Связь с ордерами
    entry_order_id: Optional[str] = None
    exit_order_id: Optional[str] = None

    # Настраиваемый таймаут
    max_hold_hours: Optional[float] = None  # None = конфиг по умолчанию

    # Идентификатор
    position_id: str = ""

    def __post_init__(self):
        if not self.position_id:
            self.position_id = f"{self.symbol}_{int(self.created_at)}"
        if self.highest_price <= 0:
            self.highest_price = self.entry_price
        if self.lowest_price <= 0:
            self.lowest_price = self.entry_price

    # ── PnL расчёты ─────────────────────────────────────────────────────

    @property
    def pnl_pct(self) -> float:
        """Текущий PnL в процентах от entry (положительный = прибыль)."""
        if self.entry_price <= 0:
            return 0.0
        if self.direction == Direction.LONG:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

    @property
    def pnl_usd(self) -> float:
        """Текущий PnL в USD."""
        if self.direction == Direction.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def position_value(self) -> float:
        """Текущая стоимость позиции в USD."""
        return self.quantity * self.current_price

    @property
    def entry_cost(self) -> float:
        """Стоимость при входе (с учётом плеча)."""
        return self.quantity * self.entry_price / self.leverage

    @property
    def age_hours(self) -> float:
        """Возраст позиции в часах."""
        return (time.time() - self.created_at) / 3600

    @property
    def hold_timeout_hours(self) -> float:
        """Максимальное время удержания (конфиг или установленное)."""
        return self.max_hold_hours or 48.0

    @property
    def is_old(self) -> bool:
        """Позиция висит дольше max_hold_hours."""
        return self.age_hours >= self.hold_timeout_hours

    @property
    def is_stale(self) -> bool:
        """Зависшая позиция: >36ч и PnL < +2%."""
        return self.age_hours >= 36 and self.pnl_pct < 2.0

    @property
    def should_break_even(self) -> bool:
        """Пора выйти в безубыток: ≥4ч, не было профита >70% от TP."""
        return self.age_hours >= 4.0 and self.max_profit_pct < 2.0 and \
               -0.2 <= self.pnl_pct <= 1.0

    # ── SL/TP проверки ─────────────────────────────────────────────────

    @property
    def is_stop_loss(self) -> bool:
        """Цена достигла SL (с учётом направления)."""
        if self.sl_price is None:
            return False
        if self.direction == Direction.LONG:
            return self.current_price <= self.sl_price
        else:
            return self.current_price >= self.sl_price

    @property
    def is_take_profit(self) -> bool:
        """Цена достигла TP (с учётом направления)."""
        if self.tp_price is None:
            return False
        if self.direction == Direction.LONG:
            return self.current_price >= self.tp_price
        else:
            return self.current_price <= self.tp_price

    @property
    def is_trail_triggered(self) -> bool:
        """Трейлинг активирован (цена прошла trail_activation %)."""
        if self.trail_activation is None:
            return False
        if self.direction == Direction.LONG:
            return self.max_profit_pct >= self.trail_activation
        else:
            return self.max_profit_pct >= self.trail_activation

    @property
    def is_trail_hit(self) -> bool:
        """Цена откатила на trail_distance от пика."""
        if self.trail_activation is None or not self.is_trail_triggered:
            return False
        if self.trail_distance is None:
            return False
        if self.direction == Direction.LONG:
            if self.highest_price <= self.entry_price:
                return False
            trail_level = self.highest_price * (1 - self.trail_distance / 100)
            return self.current_price <= trail_level
        else:
            if self.lowest_price <= 0 or self.lowest_price >= self.entry_price:
                return False
            trail_level = self.lowest_price * (1 + self.trail_distance / 100)
            return self.current_price >= trail_level

    # ── Обновление ─────────────────────────────────────────────────────

    def update_price(self, price: float) -> None:
        """Обновить текущую цену и экстремумы."""
        self.current_price = price
        if self.direction == Direction.LONG:
            if price > self.highest_price:
                self.highest_price = price
                self.max_profit_pct = max(self.max_profit_pct, self.pnl_pct)
        else:
            if price < self.lowest_price or self.lowest_price <= 0:
                self.lowest_price = price
                self.max_profit_pct = max(self.max_profit_pct, self.pnl_pct)

    def update_sl_to_breakeven(self) -> bool:
        """Подтянуть SL до entry (безубыток). True если SL изменён."""
        if self.direction == Direction.LONG:
            if self.pnl_pct >= 1.5:
                target = self.entry_price
                if self.pnl_pct >= 3.0:
                    target = self.entry_price * 1.01  # +1%
                if self.sl_price is None or target > self.sl_price:
                    self.sl_price = target
                    return True
        else:
            if self.pnl_pct >= 1.5:
                target = self.entry_price
                if self.pnl_pct >= 3.0:
                    target = self.entry_price * 0.99  # -1%
                if self.sl_price is None or target < self.sl_price:
                    self.sl_price = target
                    return True
        return False

    # ── Конвертация из/в dict (для PG совместимости) ───────────────────

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'mode': self.mode.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'leverage': self.leverage,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'trail_activation': self.trail_activation,
            'trail_distance': self.trail_distance,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'max_profit_pct': self.max_profit_pct,
            'entry_time': self.entry_time.isoformat(),
            'created_at': self.created_at,
            'max_hold_hours': self.max_hold_hours,
            'position_id': self.position_id,
            'pnl_pct': self.pnl_pct,
            'pnl_usd': self.pnl_usd,
            'position_value': self.position_value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Position:
        """Восстановление из dict (из PG или дампа)."""
        entry_time = data.get('entry_time')
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        return cls(
            symbol=data['symbol'],
            direction=Direction(data.get('direction', 'long')),
            mode=Mode(data.get('mode', 'spot')),
            quantity=data.get('quantity', 0),
            entry_price=data.get('entry_price', 0),
            current_price=data.get('current_price', 0),
            leverage=data.get('leverage', 1),
            sl_price=data.get('sl_price'),
            tp_price=data.get('tp_price'),
            trail_activation=data.get('trail_activation'),
            trail_distance=data.get('trail_distance'),
            highest_price=data.get('highest_price', 0),
            lowest_price=data.get('lowest_price', 0),
            max_profit_pct=data.get('max_profit_pct', 0),
            entry_time=entry_time or datetime.now(timezone.utc),
            created_at=data.get('created_at', time.time()),
            max_hold_hours=data.get('max_hold_hours'),
            position_id=data.get('position_id', ''),
        )

    def __repr__(self) -> str:
        return (f"<Position {self.symbol} {self.direction.value.upper()} "
                f"qty={self.quantity:.4f} entry={self.entry_price:.4f} "
                f"PnL={self.pnl_pct:+.2f}% age={self.age_hours:.1f}h>")


# ─── Решение ─────────────────────────────────────────────────────────────────

@dataclass
class Decision:
    """Решение DecisionEngine: ENTER | EXIT | HOLD.

    Один объект, одно решение. Если action == 'enter' — direction показывает
    направление (LONG или SHORT).
    """
    action: Action
    direction: Direction = Direction.LONG
    symbol: str = ""

    # Для ENTER
    score: float = 0.0                     # 0-100 итоговый скор
    position_size_pct: float = 0.0         # доля капитала (0.0-1.0)
    leverage: int = 1                      # заложено сразу
    tp_levels: Optional[List[Tuple[float, float]]] = None  # частичный тейк

    # SL/TP/трейлинг (переопределяют конфиг)
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trail_activation: Optional[float] = None
    trail_distance: Optional[float] = None
    max_hold_hours: Optional[float] = None

    # Для EXIT
    exit_override: ExitOverride = ExitOverride.NONE
    exit_reason: str = ""

    # Мета-данные
    reason: str = ""
    signal_type: Optional[Any] = None      # SignalType из decision_engine (для совместимости)
    priority: int = 50
    metadata: Dict[str, Any] = field(default_factory=dict)
    exit_vote: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_enter(self) -> bool:
        return self.action == Action.ENTER

    @property
    def is_exit(self) -> bool:
        return self.action == Action.EXIT

    @property
    def is_hold(self) -> bool:
        return self.action == Action.HOLD

    @property
    def is_long(self) -> bool:
        return self.direction == Direction.LONG

    @property
    def is_short(self) -> bool:
        return self.direction == Direction.SHORT

    def __repr__(self) -> str:
        if self.is_hold:
            return f"[HOLD] {self.symbol}: {self.reason}"
        tag = "ENTER" if self.is_enter else "EXIT"
        dir_tag = self.direction.value.upper()
        sz = f" size={self.position_size_pct:.2f}" if self.position_size_pct else ""
        return f"[{tag}({dir_tag})] {self.symbol}: score={self.score:.0f}{sz} | {self.reason}"


# ─── Рынок ──────────────────────────────────────────────────────────────────

@dataclass
class BtcState:
    """Состояние BTC, влияющее на фильтрацию сделок."""
    price: float = 0.0
    trend: str = "sideways"                # up_trend | down_trend | sideways
    regime: str = "normal"                 # calm | normal | volatile | stress
    regime_code: int = 1                   # 0 calm, 1 normal, 2 volatile, 3 stress
    direction: str = "neutral"             # из btc_direction predictor
    confidence: float = 0.0
    up_probability: float = 0.5
    down_probability: float = 0.5
    short_blocked: bool = False            # True если BTC сильно растёт
    long_blocked: bool = False             # True если BTC в глубоком дампе
    change_1h: float = 0.0                 # изменение BTC за 1 час

    def __post_init__(self):
        self._check_blocked()

    def _check_blocked(self) -> None:
        """Автоматически вычисляем blocked-флаги на основе тренда."""
        if self.regime == 'accumulation':
            self.long_blocked = False
            self.short_blocked = True       # на accumulation не шортим
        elif self.regime in ('dump', 'distribution'):
            self.long_blocked = True
            self.short_blocked = False
        elif self.trend == 'up_trend':
            self.long_blocked = False
            self.short_blocked = True
        elif self.trend == 'down_trend':
            self.long_blocked = True
            self.short_blocked = False
        else:
            self.long_blocked = False
            self.short_blocked = False

    @property
    def can_long(self) -> bool:
        return not self.long_blocked

    @property
    def can_short(self) -> bool:
        return not self.short_blocked

    @property
    def is_up_trend(self) -> bool:
        return self.trend == 'up_trend'

    @property
    def is_down_trend(self) -> bool:
        return self.trend == 'down_trend'

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self._check_blocked()


# ─── Риск ────────────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Параметры риск-менеджмента (из config + переопределения)."""
    # Long
    max_long_positions: int = 25
    long_position_pct: float = 15.0        # % капитала на 1 лонг
    long_sl_pct: float = 1.5
    long_tp_pct: float = 10.0
    long_trail_act: float = 2.5            # % для активации трейлинга
    long_trail_dist: float = 1.5           # % отката от пика
    long_max_consecutive_losses: int = 3

    # Short
    max_short_positions: int = 10
    short_position_pct: float = 10.0       # % капитала на 1 шорт
    short_sl_pct: float = 1.5
    short_tp_pct: float = 6.0              # шорты короче
    short_trail_act: float = 2.0
    short_trail_dist: float = 1.2
    short_max_consecutive_losses: int = 2  # шорты жёстче

    # Общие
    max_open_positions: int = 35           # всего
    max_daily_loss_pct: float = 3.0
    portfolio_guard_drawdown: float = 5.0  # $5 или 1.5% капитала
    max_order_value: float = 90.0
    min_position_value: float = 1.0
    entry_cooldown_seconds: int = 300
    max_hold_hours: float = 48.0
    stale_cleanup_hours: float = 36.0
    min_confidence_pct: float = 15.0
    max_daily_trades: int = 100
    reentry_cooldown: float = 600.0        # 10 мин после SL

    # Маржа/плечо (заложено для будущего)
    max_leverage: float = 5.0
    max_margin_positions: int = 5
    min_margin_ratio: float = 0.25         # минимум 25% маржи

    def for_direction(self, direction: Direction) -> dict:
        """Вернуть параметры для конкретного направления."""
        prefix = 'short_' if direction == Direction.SHORT else 'long_'
        return {
            'max_positions': getattr(self, f'max_{prefix}positions'),
            'position_pct': getattr(self, f'{prefix}position_pct'),
            'sl_pct': getattr(self, f'{prefix}sl_pct'),
            'tp_pct': getattr(self, f'{prefix}tp_pct'),
            'trail_act': getattr(self, f'{prefix}trail_act'),
            'trail_dist': getattr(self, f'{prefix}trail_dist'),
            'max_consecutive_losses': getattr(self, f'{prefix}max_consecutive_losses'),
        }


@dataclass
class ApprovalResult:
    """Результат проверки RiskManager'ом."""
    approved: bool
    reason: str = ""
    max_position_size: float = 0.0        # макс размер в USD
    suggested_leverage: int = 1

    @property
    def is_approved(self) -> bool:
        return self.approved

    @classmethod
    def reject(cls, reason: str) -> ApprovalResult:
        return cls(approved=False, reason=reason)

    @classmethod
    def approve(cls, max_size: float = 0.0, leverage: int = 1) -> ApprovalResult:
        return cls(approved=True, reason="ok", max_position_size=max_size,
                   suggested_leverage=leverage)


# ─── Фильтры ─────────────────────────────────────────────────────────────────

@dataclass
class PositionFilter:
    """Фильтр для запроса позиций."""
    symbol: Optional[str] = None
    direction: Optional[Direction] = None
    mode: Optional[Mode] = None
    min_age_seconds: float = 0
    max_age_seconds: Optional[float] = None
    min_pnl: Optional[float] = None
    max_pnl: Optional[float] = None
    stale_only: bool = False
    sort_by: str = "created_at"           # created_at | pnl | age
    sort_desc: bool = True
    limit: Optional[int] = None


# ─── Счётчик результатов ────────────────────────────────────────────────────

@dataclass
class TradeOutcome:
    """Результат закрытой сделки (для VoteTracker, ML-Pro обучения)."""
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    quantity: float
    pnl_pct: float
    pnl_usd: float
    hold_hours: float
    exit_reason: str
    entry_time: datetime
    exit_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    module_signals: Dict[str, ModuleSignal] = field(default_factory=dict)
