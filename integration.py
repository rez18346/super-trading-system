#!/usr/bin/env python3
"""
integration.py — Интеграционный слой между IndustrialTrader и новыми модулями.

Стратегия:
  - Все новые модули (PM, OM, RM) инициализируются и работают параллельно со старым кодом
  - Новые модули = SOURCE OF TRUTH для состояния
  - Старые self.positions / execute_trade / check_risk_limits становятся обёртками
  - Постепенное переключение: сначала entry, потом exit, потом full

  Фазы:
    1. ✅ Модули созданы (models, order_manager, position_manager, risk_manager)
    2. 🔄 Сейчас: интеграция с IndustrialTrader — init + entry + exit + status
    3. ⏳ Позже: деприкация старых структур
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

from models import (
    Direction, Mode, Position, Order, OrderStatus, Decision,
    ApprovalResult, BtcState, RiskConfig, TradeOutcome,
    ExitOverride,
)
from order_manager import OrderManager
from position_manager import PositionManager
from risk_manager import RiskManager

logger = logging.getLogger("integration")


def init_new_modules(trader_self) -> dict:
    """Инициализировать PM, OM, RM внутри IndustrialTrader.

    ⚡ ШОРТ: OM создаётся в Mode.SPOT (настраивается при активации шорта).
    ⚡ ПЛЕЧО: default_leverage=1.0 — настраивается при активации маржи.

    Вызывается из __init__ после load_config() и setup_exchange().

    Returns:
        {'pm': PositionManager, 'om': OrderManager, 'rm': RiskManager}
    """
    config = trader_self.config
    exchange = trader_self.exchange

    # PositionManager — единое хранилище позиций
    pm = PositionManager(
        exchange=exchange,
        config=config,
        on_position_closed=_make_on_closed(trader_self),
    )

    # 🗄️ Восстанавливаем открытые позиции из PG (включая SHORT)
    pm.load_from_db()

    # RiskManager — центр проверок
    rm = RiskManager(
        config=config,
        position_manager=pm,
        on_block=lambda r: _on_risk_block(trader_self, r),
    )

    # ⚡ OrderManager — ордера через CCXT (SPOT по умолчанию)
    trading_config = config.get('trading', {})
    default_leverage = trading_config.get('max_leverage', 1.0)

    om = OrderManager(
        exchange=exchange,
        config=config,
        on_order_filled=pm.on_order_filled,
        default_mode=Mode.SPOT,
        default_leverage=default_leverage,
    )

    # ⚡ Если в конфиге указан margin/futures — активируем
    trade_mode = trading_config.get('trade_mode', 'spot')
    if trade_mode in ('margin', 'futures'):
        mode = Mode.MARGIN if trade_mode == 'margin' else Mode.FUTURES
        om.set_mode(mode)
        # Устанавливаем плечо для основных пар
        for sym in config.get('recommended_pairs', []):
            om.set_leverage(sym, default_leverage)
        logger.info(f"🔧 [INIT] Режим торговли: {mode.value}, плечо: {default_leverage}x")

    # Синхронизируем RM с PM
    pm._available_capital = getattr(trader_self, 'available_capital', trading_config.get('capital', 300))

    # Загружаем позиции из PG
    loaded = pm.load_from_db()
    if loaded:
        logger.info(f"📦 Загружено {loaded} позиций из БД в PositionManager")

    return {'pm': pm, 'om': om, 'rm': rm}


def process_entry_decision(trader_self, symbol: str, decision: Any,
                           current_price: float, btc_state: Optional[dict] = None,
                           pyramid_pct: float = 1.0) -> Optional[Dict]:
    """Обработать решение DE о входе через новые модули.

    ⚡ ШОРТ: direction=SHORT, side='sell' (sell to open).
    ⚡ ПЛЕЧО: leverage передаётся в submit_order через OrderManager.mode.

    Args:
        pyramid_pct: доля полного размера для входа (1.0 = 100%, 0.3 = 30%)

    Вызывается из trading_cycle когда decision.action == 'enter'.

    Returns:
        {'order': Order, 'position': Position} при успехе
        None при блокировке
    """
    pm: PositionManager = getattr(trader_self, '_pm', None)
    om: OrderManager = getattr(trader_self, '_om', None)
    rm: RiskManager = getattr(trader_self, '_rm', None)

    if not pm or not om or not rm:
        return None  # нет модулей — старый код

    # ── 1. Определяем направление ─────────────────────────────────────
    direction = Direction.SHORT if getattr(decision, 'side', 'long') == 'short' else Direction.LONG

    # ── 2. BTC State ──────────────────────────────────────────────────
    btc_state_obj = _build_btc_state(trader_self, btc_state)

    # ── 3. RiskManager: check_entry ───────────────────────────────────
    score = getattr(decision, 'score', 50) or 50
    approval = rm.check_entry(symbol, direction, current_price, score, btc_state_obj, decision)
    if not approval.approved:
        logger.info(f"⛔ [RM→BLOCK] {symbol} {direction.value.upper()}: {approval.reason}")
        return None

    # ── 4. Размер позиции ────────────────────────────────────────────
    max_usd = rm.adjust_position_size(decision, btc_state_obj, approval, direction)

    # 🏔️ PYRAMID: берём только указанный процент от полного размера
    if pyramid_pct < 1.0:
        full_max_usd = max_usd
        max_usd = max_usd * pyramid_pct
        max_usd = max(0.01, max_usd)
        logger.info(f"🏔️ [PYRAMID] {symbol}: {pyramid_pct*100:.0f}% от ${full_max_usd:.2f} = ${max_usd:.2f}")

    # ── 5. SL/TP уровни ──────────────────────────────────────────────
    sl_price = getattr(decision, 'sl_price', None) or rm.calc_sl_price(current_price, direction, btc_state_obj)
    tp_price = getattr(decision, 'tp_price', None) or rm.calc_tp_price(current_price, direction)
    trail_act, trail_dist = rm.calc_trail_params(direction)

    # ── 6. Количество ────────────────────────────────────────────────
    quantity = max_usd / current_price if current_price > 0 else 0
    if quantity <= 0:
        logger.warning(f"⏭️ [PM] {symbol}: qty={quantity:.6f} (max_usd={max_usd:.2f} / price={current_price:.4f})")
        return None

    # ── 6. Количество ────────────────────────────────────────────────
    quantity = max_usd / current_price if current_price > 0 else 0
    if quantity <= 0:
        logger.warning(f"⏭️ [PM] {symbol}: qty={quantity:.6f} (max_usd={max_usd:.2f} / price={current_price:.4f})")
        return None

    # 🛡️ SANITY CHECK: позиция не должна превышать 30% свободного капитала
    _free_capital = getattr(trader_self, 'available_capital', 0) or getattr(trader_self, 'capital', 100)
    _position_value = quantity * current_price
    if _position_value > _free_capital * 0.30:
        logger.warning(f"🛡️ [PM→BLOCK] {symbol}: позиция ${_position_value:.2f} > 30% капитала (${_free_capital:.2f}). Блокирую.")
        return None

    # ── 7. OrderManager: submit ──────────────────────────────────────
    # ⚡ ШОРТ: side='sell' = открытие позиции
    # ⚡ ЛОНГ: side='buy' = открытие позиции
    side = 'buy' if direction == Direction.LONG else 'sell'
    # 🔒 Buy_lock: после одобрения RM, до отправки ордера — защита от дублей
    rm.set_buy_lock(symbol)
    metadata = {
        'sl_price': sl_price,
        'tp_price': tp_price,
        'trail_activation': trail_act,
        'trail_distance': trail_dist,
        'max_hold_hours': decision.max_hold_hours if hasattr(decision, 'max_hold_hours') else None,
        'reason': getattr(decision, 'reason', ''),
        'score': score,
        'direction': direction.value,
    }
    order_result = om.submit_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=current_price,
        direction=direction,
        order_type='market',
        reason=getattr(decision, 'reason', ''),
        metadata=metadata,
    )

    # ── 8. Обработка результата ──────────────────────────────────────
    if order_result.success and order_result.order:
        order = order_result.order
        rm.set_buy_lock(symbol)  # защита от двойного входа

        if order.is_filled:
            rm.release_buy_lock(symbol)
            # PositionManager.on_order_filled уже создал позицию
            position = pm.get_position(symbol)
            if position:
                # Регистрируем в self.positions для совместимости со старым SL/TP циклом
                _tp = tp_price or (position.tp_price if hasattr(position, 'tp_price') else None)
                _sp = sl_price or (position.sl_price if hasattr(position, 'sl_price') else None)
                _ta = trail_act or (position.trail_activation if hasattr(position, 'trail_activation') else None)
                _td = trail_dist or (position.trail_distance if hasattr(position, 'trail_distance') else None)
                trader_self.positions[symbol] = {
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'entry_time': datetime.now(timezone.utc).isoformat(),
                    'side': direction.value if hasattr(direction, 'value') else str(direction),
                    '_sl_price': _sp,
                    '_tp_price': _tp,
                    '_trail_act': _ta,
                    '_trail_dist': _td,
                    '_created_at': time.time(),
                }
                logger.info(f"✅ [INTEGRATION] {direction.value.upper()} {symbol}: "
                           f"{position.quantity:.4f} @ ${position.entry_price:.4f} "
                           f"SL=${_sp or 'auto'}, TP=${_tp or 'auto'}, "
                           f"Score={score:.0f}")
                return {'order': order, 'position': position}

        logger.info(f"⏳ [INTEGRATION] {direction.value.upper()} {symbol}: "
                   f"ордер {order.client_id} {order.status.value}, "
                   f"qty={quantity:.4f} @ ${current_price:.4f}")
        return {'order': order, 'position': None}

    else:
        # Ордер не выполнился — снимаем buy_lock
        rm.release_buy_lock(symbol)
        logger.warning(f"❌ [INTEGRATION] {direction.value.upper()} {symbol}: "
                      f"{order_result.error}")
        return None


def process_exit_decision(trader_self, symbol: str, exit_reason: str,
                          current_price: float) -> Optional[TradeOutcome]:
    """Обработать решение DE о выходе через новые модули.

    ⚡ ШОРТ: side='buy' (buy to close). PositionManager сам инвертирует PnL.
    ⚡ ЛОНГ: side='sell' (sell to close).

    Вызывается из trading_cycle когда should_sell == True.

    Returns:
        TradeOutcome при успехе
        None при ошибке
    """
    pm: PositionManager = getattr(trader_self, '_pm', None)
    om: OrderManager = getattr(trader_self, '_om', None)
    rm: RiskManager = getattr(trader_self, '_rm', None)

    if not pm or not om or not rm:
        return None

    # ── 1. Проверяем, есть ли позиция в PM ───────────────────────────
    pos = pm.get_position(symbol)
    if not pos:
        logger.warning(f"⏭️ [EXIT] {symbol}: нет в PositionManager")
        return None

    # ── 2. Проверка реального баланса ────────────────────────────────
    try:
        balance = trader_self.exchange.fetch_balance()
        currency = symbol.split('/')[0]
        
        if pos.direction == Direction.SHORT and trader_self._om.mode != Mode.SPOT:
            # ⚡ ШОРТ в MARGIN: позиция на залоге, не на балансе
            # Проверяем через маржу, а не баланс актива
            total_asset = 0  # не проверяем asset для margin short
            pos_value = abs(pos.entry_cost)  # используем залог
        else:
            total_asset = balance['total'].get(currency, 0)
            pos_value = total_asset * current_price
    except Exception as e:
        logger.error(f"❌ [EXIT] balance check {symbol}: {e}")
        return None

    # ⚡ ШОРТ: пропускаем проверку asset, т.к. нет актива на балансе
    if pos.direction != Direction.SHORT:
        if total_asset < 0.000001:
            logger.warning(f"⏭️ [EXIT] {symbol}: нет на бирже")
            pm.close_position(symbol, current_price, f"нет_на_бирже:{exit_reason}")
            return None

        MIN_VALUE = 1.0
        if pos_value < MIN_VALUE:
            logger.warning(f"⏭️ [EXIT] {symbol}: пыль ${pos_value:.2f}")
            pm.close_position(symbol, current_price, f"пыль:{exit_reason}")
            return None

    # ── 3. Определяем сторону ⚡ ШОРТ = buy to close ─────────────────
    # ⚡ ЛОНГ: side='sell' для выхода
    # ⚡ ШОРТ (MARGIN/FUTURES): side='buy' для выкупа
    side = 'sell' if pos.direction == Direction.LONG else 'buy'

    # ── 4. Количество ────────────────────────────────────────────────
    if pos.direction == Direction.SHORT and trader_self._om.mode != Mode.SPOT:
        # ⚡ ШОРТ: используем количество из позиции
        safe_qty = pos.quantity
    else:
        safe_qty = total_asset * 0.999

    # ── 5. OrderManager: submit ──────────────────────────────────────
    order_result = om.submit_order(
        symbol=symbol,
        side=side,
        quantity=safe_qty,
        price=current_price,
        direction=pos.direction,
        order_type='market',
        reason=exit_reason,
        metadata={'reason': exit_reason, 'direction': pos.direction.value},
    )

    # ── 6. Обработка результата ──────────────────────────────────────
    if order_result.success and order_result.order:
        order = order_result.order
        rm.set_reentry_cooldown(symbol)

        if order.is_filled:
            outcome = pm.close_position(symbol, order.avg_price, exit_reason)
            if outcome:
                logger.info(f"✅ [EXIT] {symbol}: PnL={outcome.pnl_usd:+.2f}$ ({outcome.pnl_pct:+.2f}%) | {exit_reason}")
                return outcome
        else:
            # 🩹 Ордер отправлен, но не исполнен — закрываем PM позицию в любом случае
            # (на spot/margin market orders могут не сразу исполниться, а позицию уже выставили)
            logger.info(f"⏳ [EXIT] {symbol}: ордер {order.status}, закрываю позицию в PM")
        
        # 🩹 Всегда закрываем PM позицию и записываем сделку
        outcome = pm.close_position(symbol, current_price, exit_reason)
        if outcome:
            logger.info(f"✅ [EXIT] {symbol}: PnL={outcome.pnl_usd:+.2f}$ ({outcome.pnl_pct:+.2f}%) | {exit_reason}")
            return outcome

        return TradeOutcome(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=current_price,
            quantity=safe_qty,
            pnl_pct=pos.pnl_pct,
            pnl_usd=pos.pnl_usd,
            hold_hours=pos.age_hours,
            exit_reason=exit_reason,
            entry_time=pos.entry_time,
        )

    logger.warning(f"❌ [EXIT] {symbol}: ордер не отправлен — {order_result.error}")
    return None


def sync_all(trader_self) -> None:
    """Синхронизировать все модули с биржей и БД.

    Вызывается каждый цикл (заменяет прямые вызовы exchange.fetch_balance)."""
    pm: PositionManager = getattr(trader_self, '_pm', None)
    om: OrderManager = getattr(trader_self, '_om', None)
    rm: RiskManager = getattr(trader_self, '_rm', None)

    if om:
        om.sync_open_orders()
        om.cleanup_old_orders()

    if pm:
        pm.sync_with_exchange()
        # Обновляем available_capital в trader
        trader_self.available_capital = pm.available_capital

    if rm:
        rm.clean_expired_locks()


def get_module_status(trader_self) -> dict:
    """Собрать статус из новых модулей."""
    pm: PositionManager = getattr(trader_self, '_pm', None)
    om: OrderManager = getattr(trader_self, '_om', None)
    rm: RiskManager = getattr(trader_self, '_rm', None)

    status = {}
    if pm:
        status['position_manager'] = {
            'count': pm.positions_count,
            'long': pm.long_count,
            'short': pm.short_count,
            'exposure': round(pm.total_exposure, 2),
            'capital': pm.available_capital,
        }
    if om:
        status['order_manager'] = om.get_stats()
    if rm:
        status['risk_manager'] = rm.get_status()

    return status


# ── Приватные ─────────────────────────────────────────────────────────────

def _make_on_closed(trader_self) -> Callable:
    """Создать callback для PositionManager.on_position_closed."""
    def on_closed(symbol: str, direction: Direction, pnl_pct: float):
        # Считаем USD PnL
        _usd_pnl = pnl_pct / 100 * (
            trader_self.positions.get(symbol, {}).get('quantity', 0) *
            trader_self.positions.get(symbol, {}).get('entry_price', 0)
        ) if symbol in trader_self.positions else 0
        # Обновляем daily_pnl
        trader_self.daily_pnl += _usd_pnl
        # 🏦 Profit Lock: прибыль уходит в защищённый капитал
        if _usd_pnl > 0.001:
            trader_self._protected_capital += _usd_pnl
            trader_self._protected_capital_alltime += _usd_pnl
            trader_self.available_capital = max(0, trader_self.available_capital - _usd_pnl)
            logger.info(f"🏦 [PROFIT LOCK] {symbol}: +${_usd_pnl:.2f} защищено (всего: ${trader_self._protected_capital:.2f})")
        # Обновляем daily_trades
        trader_self.daily_trades += 1
        # Portfolio Guard check
        rm: RiskManager = getattr(trader_self, '_rm', None)
        if rm:
            drawdown = trader_self._daily_peak_pnl - trader_self.daily_pnl
            rm.check_portfolio_guard(trader_self.daily_pnl, drawdown)
    return on_closed


def _on_risk_block(trader_self, reason: str) -> None:
    """Callback при блокировке рисков."""
    logger.warning(f"⛔ [RISK BLOCK] {reason}")
    # Если Portfolio Guard — блокируем входы
    if 'Portfolio Guard' in reason:
        trader_self._portfolio_guard_triggered = True


def _build_btc_state(trader_self, btc_data: Optional[dict] = None) -> BtcState:
    """Собрать BtcState из доступных данных.

    ⚡ ШОРТ: short_blocked=True когда BTC в up-trend или accumulation.
    """
    btc_price = btc_data.get('price', 0) if btc_data else 0
    trend = btc_data.get('trend', 'sideways') if btc_data else 'sideways'

    # ⚡ Определяем, заблокирован ли шорт
    short_blocked = True
    long_blocked = False
    regime = 'unknown'
    direction = 'neutral'

    # Пытаемся получить из BTC Regime Tracker
    if hasattr(trader_self, '_btc_regime'):
        try:
            regime = trader_self._btc_regime.get_regime()
            # 🩹 Направление берём из get_direction(), а не из строки рекомендации
            direction = trader_self._btc_regime.get_direction()

            # BTC-direction правила:
            # 'up'  → шорт заблокирован (любая бычья фаза)
            # 'down' → шорт разрешён (dump, distribution)
            # 'neutral' → шорт разрешён (accumulation, боковик)
            if direction == 'down':
                short_blocked = False
            elif direction == 'neutral':
                short_blocked = False  # разрешаем шорт в нейтрале
            # else direction == 'up' → short_blocked = True (умолчание)
            logger.debug(f"[_btc_state] regime={regime} dir={direction} has_rt=True → short_blocked={short_blocked}")

        except Exception as _bse:
            logger.debug(f"[_btc_state] regime exception: {_bse}")
    else:
        # 🩹 Fallback: regime tracker нет — используем trend
        if trend in ('up_trend', 'bullish', 'accumulation'):
            short_blocked = True
        elif trend in ('down_trend', 'bearish', 'distribution'):
            short_blocked = False
        elif trend in ('neutral', 'sideways'):
            short_blocked = False  # нейтрал → шорт разрешён

    _state = BtcState(
        price=btc_price,
        trend=trend,
        regime=regime,
        direction=direction,
        short_blocked=short_blocked,
        long_blocked=long_blocked,
    )
    # ⚡ _check_blocked() в __post_init__ может перетереть short_blocked
    # (напр. regime=accumulation → short_blocked=True) — восстанавливаем
    _state.short_blocked = short_blocked
    _state.long_blocked = long_blocked
    return _state
