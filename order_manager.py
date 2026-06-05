#!/usr/bin/env python3
"""
order_manager.py - Управление ордерами.

Архитектура:
  Единственный класс, который взаимодействует с биржей через CCXT.
  Никакой другой код не отправляет ордера напрямую.

  Жизненный цикл ордера:
    1. submit_order() → Order(status=PENDING) → CCXT create_order()
    2. CCXT подтверждает → Order(status=OPEN или CLOSED)
    3. sync_open_orders() обновляет статусы
    4. PositionManager забирает исполненные ордера

  Ключевое отличие от execute_trade():
    - Ордер и позиция — разные вещи. OrderManager отвечает только за ордер.
    - Позиция создаётся PositionManager'ом ПОСЛЕ того, как ордер исполнился.
    - Баг "buy_lock ставится до проверки баланса" невозможен в принципе.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Callable, Any

from models import (
    Direction, Order, OrderResult, OrderStatus, Position,
    RiskConfig, ApprovalResult, Mode,
)

logger = logging.getLogger("order_manager")


class OrderManager:
    """Управление ордерами на бирже.

    ⚡ Поддерживает SPOT, MARGIN (кросс/изолированный), FUTURES.
    ⚡ Short entry: sell to open (margin/futures).
    ⚡ Short exit: buy to close (margin/futures).
    ⚡ Leverage: управляется через set_leverage().
    """

    def __init__(self, exchange, config: dict, risk_config: Optional[RiskConfig] = None,
                 on_order_filled: Optional[Callable] = None,
                 default_mode: Mode = Mode.SPOT,
                 default_leverage: float = 1.0):
        """
        exchange: CCXT биржа
        config: api_config_final.json (полный конфиг)
        risk_config: RiskConfig (если None — читает из config)
        on_order_filled: callback(position_data) при исполнении
        default_mode: SPOT | MARGIN | FUTURES (режим по умолчанию)
        default_leverage: плечо по умолчанию (1.0 = без плеча)
        """
        self.exchange = exchange
        self.config = config
        self.risk_config = risk_config or self._build_risk_config(config)
        self.on_order_filled = on_order_filled

        # ⚡ Режим и плечо
        self.mode: Mode = default_mode
        self._leverage: Dict[str, float] = {}  # symbol -> leverage
        self.default_leverage = default_leverage

        # Все ордера: {client_id: Order}
        self._orders: Dict[str, Order] = {}

        # Ордера, отправленные на биржу (но ещё не подтверждённые)
        self._pending_orders: Dict[str, Order] = {}

        # История отменённых/исполненных
        self._history: List[Order] = []

        # Кэш установленного плеча (чтобы не дёргать биржу без нужды)
        self._leverage_set: Dict[str, float] = {}

        logger.info(f"OrderManager инициализирован (mode={default_mode.value}, leverage={default_leverage}x)")

    def _build_risk_config(self, config: dict) -> RiskConfig:
        """Собрать RiskConfig из конфига."""
        rm = config.get('risk_management', {})
        tr = config.get('trading', {})
        pm = rm.get('portfolio_management', {})

        max_positions = rm.get('max_open_positions', 25)
        # short max = треть от общего (но не больше 10)
        short_max = min(max_positions // 3, 10)

        return RiskConfig(
            max_long_positions=max_positions,
            max_short_positions=short_max,
            max_open_positions=max_positions + short_max,
            long_position_pct=tr.get('max_position_size_percent', 15),
            short_position_pct=10.0,
            long_sl_pct=1.5,
            short_sl_pct=1.5,
            long_tp_pct=10.0,
            short_tp_pct=6.0,
            max_daily_loss_pct=rm.get('max_daily_loss_percent', 3.0),
            max_order_value=rm.get('max_buy_order_usd', 90.0),
            min_position_value=rm.get('min_position_usd', 5.0),
            max_daily_trades=rm.get('max_daily_trades', 100),
            entry_cooldown_seconds=300,
            max_leverage=tr.get('max_leverage', 5.0),
        )


    # ── Публичные методы ────────────────────────────────────────────────

    # ── Управление режимом и плечом ───────────────────────────────────

    def _enable_spot_margin(self) -> bool:
        """Включить спот-маржинальную торговлю на Bybit UTA.

        Вызывает POST /v5/spot-margin-trade/switch-mode с spotMarginMode=1.
        Необходимо для isLeverage=1 в ордерах.
        """
        try:
            resp = self.exchange.private_post_v5_spot_margin_trade_switch_mode(
                {'spotMarginMode': '1'}
            )
            ret_code = resp.get('retCode', -1)
            if ret_code == 0:
                logger.info("✅ [SPOT MARGIN] Успешно включена")
                return True
            elif ret_code == 170036:
                logger.warning(f"⚠️ [SPOT MARGIN] Уже включена или недоступна: {resp.get('retMsg', '')}")
                return True
            else:
                logger.warning(f"⚠️ [SPOT MARGIN] switch-mode: {resp.get('retCode')}: {resp.get('retMsg', '')}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ [SPOT MARGIN] Ошибка включения: {e}")
            return False

    def _set_spot_margin_leverage(self, leverage: int = 2) -> bool:
        """Установить плечо для спот-маржинальной торговли Bybit UTA.

        Вызывает POST /v5/spot-margin-trade/set-leverage.
        leverage: 2 или 5 (макс)
        """
        try:
            resp = self.exchange.private_post_v5_spot_margin_trade_set_leverage(
                {'leverage': str(leverage)}
            )
            ret_code = resp.get('retCode', -1)
            if ret_code == 0:
                logger.info(f"✅ [SPOT MARGIN LEVERAGE] Установлено {leverage}x")
                return True
            else:
                logger.warning(f"⚠️ [SPOT MARGIN LEVERAGE] set-leverage: {resp.get('retCode')}: {resp.get('retMsg', '')}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ [SPOT MARGIN LEVERAGE] Ошибка: {e}")
            return False

    def set_mode(self, mode: Mode) -> None:
        """Установить режим торговли: SPOT | MARGIN | FUTURES.

        Args:
            mode: Mode.SPOT — спот (только long, актив в собственности)
                  Mode.MARGIN — кросс-маржинальная торговля (short разрешён)
                  Mode.FUTURES — фьючерсы (short/long с плечом)
        """
        old_mode = self.mode
        self.mode = mode
        logger.info(f"🔧 [MODE] {old_mode.value} → {mode.value}")

        # Если переключаемся на MARGIN — включаем спот-маржу на Bybit
        if mode == Mode.MARGIN:
            self._enable_spot_margin()
            self._set_spot_margin_leverage(int(self.default_leverage))

        # Если переключаемся на MARGIN/FUTURES — устанавливаем позиционный режим
        if mode != Mode.SPOT:
            try:
                # Bybit: setPositionMode для удержания long+short одновременно
                if hasattr(self.exchange, 'set_position_mode'):
                    self.exchange.set_position_mode(hedge=False)
                    logger.info(f"🔧 [MODE] Установлен однонаправленный режим позиций")
            except Exception as e:
                logger.warning(f"⚠️ [MODE] set_position_mode: {e}")

    def get_leverage(self, symbol: str) -> float:
        """Получить текущее плечо для symbol."""
        return self._leverage.get(symbol, self.default_leverage)

    def set_leverage(self, symbol: str, leverage: float) -> bool:
        """Установить плечо для symbol на бирже.

        Args:
            symbol: e.g. 'BTC/USDT'
            leverage: 1.0 = без плеча, 2.0 = 2x, 5.0 = 5x, etc.

        Returns:
            True если плечо установлено (или уже было установлено)
        """
        current = self._leverage_set.get(symbol)
        if current == leverage:
            return True

        if self.mode == Mode.SPOT:
            self._leverage[symbol] = 1.0
            self._leverage_set[symbol] = 1.0
            return True

        try:
            if hasattr(self.exchange, 'set_leverage'):
                self.exchange.set_leverage(leverage, symbol)
            self._leverage[symbol] = leverage
            self._leverage_set[symbol] = leverage
            logger.info(f"🔧 [LEVERAGE] {symbol}: {leverage}x")
            return True
        except Exception as e:
            logger.warning(f"⚠️ [LEVERAGE] {symbol} {leverage}x: {e}")
            self._leverage[symbol] = leverage
            return False

    def get_mode(self) -> Mode:
        """Текущий режим торговли."""
        return self.mode

    def get_all_leverages(self) -> Dict[str, float]:
        """Получить все установленные плечи."""
        return dict(self._leverage_set)

    def set_margin_mode(self, symbol: str, margin_mode: str = 'cross') -> bool:
        """Установить режим маржи (cross/isolated)."""
        if self.mode == Mode.SPOT:
            return False
        try:
            set_margin = getattr(self.exchange, 'set_margin_mode', None)
            if callable(set_margin):
                set_margin(margin_mode, symbol)
            logger.info(f"🔧 [MARGIN_MODE] {symbol}: {margin_mode}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ [MARGIN_MODE] {symbol}: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════════════
    # ⚡ ОСНОВНОЙ МЕТОД: submit_order (market/limit)
    # ═══════════════════════════════════════════════════════════════════════

    def submit_order(self, symbol: str, side: str, quantity: float, price: float,
                     direction: Direction = Direction.LONG,
                     order_type: str = "market",
                     reason: str = "",
                     metadata: Optional[Dict] = None) -> OrderResult:
        """Отправить ордер на биржу.

        Args:
            symbol: BTC/USDT
            side: 'buy' | 'sell' (CCXT side: buy=long entry/short exit, sell=long exit/short entry)
            quantity: количество в base currency
            price: ориентировочная цена (market) или лимит
            direction: LONG | SHORT (наша интерпретация)
            order_type: 'market' | 'limit'
            reason: для логов
            metadata: scores, decision и т.д.

        ⚡ ШОРТ (MARGIN/FUTURES):
            - direction=SHORT + side='sell' = короткая продажа (открытие шорта)
            - direction=SHORT + side='buy' = выкуп шорта (закрытие)
            - В params добавляется leverage если mode != SPOT

        ⚡ ПЛЕЧО (MARGIN/FUTURES):
            - Если mode не SPOT — устанавливаем плечо перед ордером
            - Плечо расширяет позицию, но не меняет логику SL/TP

        Returns:
            OrderResult(success=True, order=Order) при успехе
            OrderResult(success=False, error="...", retryable=True) при ошибке
        """
        # ═══ 0. Установка плеча (для MARGIN/FUTURES) ═══════════════════
        ccxt_params = {}
        if self.mode != Mode.SPOT:
            # leverage для логов и расчётов
            _mode_leverage = self._leverage.get(symbol, self.default_leverage)
            # Bybit UTA spot margin: category='spot' + isLeverage
            ccxt_params['category'] = 'spot'
            # MARGIN = спот-маржинальная (UTA): isLeverage=1 включает заимствование
            # marginMode НЕ ПЕРЕДАЁТСЯ — Bybit V5 /order/create не принимает этот параметр
            if self.mode == Mode.MARGIN:
                ccxt_params['isLeverage'] = 1  # UTA: 1 = borrow for margin trading
                if direction == Direction.SHORT:
                    logger.info(f"🔧 [SHORT] {symbol}: MARGIN mode, category=spot, isLeverage=1")
                else:
                    logger.info(f"🔧 [LONG] {symbol}: MARGIN mode, category=spot, isLeverage=1")
            elif self.mode == Mode.FUTURES:
                ccxt_params['marginMode'] = 'isolated'
                self.set_leverage(symbol, _mode_leverage)
                ccxt_params['leverage'] = _mode_leverage
                ccxt_params['category'] = 'linear'
            else:
                ccxt_params['leverage'] = 1.0

            logger.debug(f"[ORDER] {symbol} {direction.value}: leverage={_mode_leverage}x, mode={self.mode.value}")

        # ═══ 1. Проверка баланса ═══════════════════════════════════════
        balance_ok, balance_msg = self._check_balance(symbol, side, quantity, price)
        if not balance_ok:
            logger.warning(f"❌ [ORDER] {symbol} {side}: {balance_msg}")
            return OrderResult(
                success=False,
                error=balance_msg,
                retryable=True,
            )

        # ═══ 2. Лимит стоимости ════════════════════════════════════════
        order_value = quantity * price
        if self.mode != Mode.SPOT:
            # С плечом: корректируем залог, а не стоимость
            leverage = self._leverage.get(symbol, self.default_leverage)
            margin_required = order_value / leverage
            max_value_with_leverage = self.risk_config.max_order_value * leverage
            if margin_required > max_value_with_leverage:
                new_qty = (max_value_with_leverage * leverage) / price
                logger.info(f"📏 [ORDER] {symbol}: лимит ${self.risk_config.max_order_value}*{leverage}x -> qty {quantity:.4f} → {new_qty:.4f}")
                quantity = new_qty
        else:
            max_value = self.risk_config.max_order_value
            if order_value > max_value and side == 'buy':
                new_qty = max_value / price
                logger.info(f"📏 [ORDER] {symbol}: лимит ${max_value} -> qty {quantity:.4f} → {new_qty:.4f}")
                quantity = new_qty

        # ═══ 3. Создаём Order ══════════════════════════════════════════
        client_id = f"om_{int(time.time() * 1000)}_{symbol.replace('/', '')}"
        order = Order(
            symbol=symbol,
            side=side,
            direction=direction,
            quantity=quantity,
            price=price,
            order_type=order_type,
            client_id=client_id,
            status=OrderStatus.PENDING,
            metadata=metadata or {},
        )
        self._orders[client_id] = order
        self._pending_orders[client_id] = order

        # ═══ 4. Отправка на биржу с плечом/режимом ════════════════════
        try:
            # ⚡ ШОРТ: для MARGIN/FUTURES side='sell' = открытие, side='buy' = закрытие
            # ⚡ ЛОНГ: для MARGIN/FUTURES side='buy' = открытие, side='sell' = закрытие
            # Для SPOT: как обычно
            if self.mode != Mode.SPOT and direction == Direction.SHORT:
                # short entry = sell to open, short exit = buy to close
                pass  # уже правильная сторона передана
            elif self.mode != Mode.SPOT and direction == Direction.LONG:
                pass  # buy = long entry, sell = long exit

            ccxt_order = self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=quantity,
                params=ccxt_params if ccxt_params else {},
            )
            # Обновляем из ответа биржи
            self._update_from_ccxt(order, ccxt_order)
            logger.info(f"✅ [ORDER] {side.upper()} {quantity:.4f} {symbol} @ ${price:.4f} "
                       f"→ ID={order.order_id} status={order.status.value}")

            # Если ордер исполнился мгновенно (market order) — зовём callback
            if order.is_filled:
                self._on_filled(order)
                self._pending_orders.pop(client_id, None)
                return OrderResult(success=True, order=order)

            # 🕐 Market order: ждём заполнения до 5 секунд
            if order_type == 'market':
                for _attempt in range(5):
                    time.sleep(1)
                    try:
                        ccxt_o = self.exchange.fetch_order(order.order_id, order.symbol)
                        self._update_from_ccxt(order, ccxt_o)
                        if order.is_filled:
                            self._on_filled(order)
                            self._pending_orders.pop(client_id, None)
                            logger.info(f"✅ [ORDER] {symbol}: заполнен через {_attempt+1}с")
                            return OrderResult(success=True, order=order)
                    except Exception as fetch_err:
                        # fetch_order не нашёл ордер — возможно исполнился и удалён
                        logger.debug(f"[ORDER] {symbol} fetch попытка {_attempt+1}/5: {fetch_err}")
                        continue
                
                # Все 5 попыток fetch провалились — проверяем через баланс
                self._update_filled_from_balance(order, symbol, side, quantity)
                if order.is_filled:
                    self._on_filled(order)
                    self._pending_orders.pop(client_id, None)
                    logger.info(f"✅ [ORDER] {symbol}: filled (по балансу, fetch не найден)")
                    return OrderResult(success=True, order=order)
                else:
                    self._pending_orders.pop(client_id, None)
                    logger.warning(f"❌ [ORDER] {symbol}: не найден на бирже, баланс не подтвердил исполнение")
                    return OrderResult(success=False, error="Order not found on exchange, balance unchanged")

            return OrderResult(success=True, order=order)

        except Exception as e:
            error_msg = str(e)
            order.status = OrderStatus.FAILED
            order.updated_at = time.time()
            self._pending_orders.pop(client_id, None)

            # Определяем, можно ли ретраить
            retryable = self._is_retryable_error(error_msg)

            if retryable:
                logger.warning(f"⚠️ [ORDER] {symbol} {side}: {error_msg} (retryable)")
            else:
                logger.error(f"❌ [ORDER] {symbol} {side}: {error_msg} (NOT retryable)")

            return OrderResult(
                success=False,
                order=order,
                error=error_msg,
                retryable=retryable,
            )

    def get_order(self, client_id: str) -> Optional[Order]:
        """Получить ордер по client_id."""
        return self._orders.get(client_id)

    def get_order_by_exchange_id(self, exchange_id: str) -> Optional[Order]:
        """Найти ордер по ID биржи."""
        for o in self._orders.values():
            if o.order_id == exchange_id:
                return o
        return None

    def cancel_order(self, client_id: str) -> bool:
        """Отменить ордер."""
        order = self._orders.get(client_id)
        if not order:
            logger.warning(f"[CANCEL] ордер {client_id} не найден")
            return False
        if order.status not in (OrderStatus.PENDING, OrderStatus.OPEN):
            logger.info(f"[CANCEL] ордер {client_id} уже {order.status.value}")
            return False
        try:
            self.exchange.cancel_order(order.order_id, order.symbol)
            order.status = OrderStatus.CANCELED
            order.updated_at = time.time()
            self._pending_orders.pop(client_id, None)
            logger.info(f"✅ [CANCEL] {order.symbol} ID={order.order_id}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ [CANCEL] {order.symbol}: {e}")
            return False

    def sync_open_orders(self) -> int:
        """Синхронизировать все открытые ордера с биржей.

        Returns:
            число обновлённых ордеров
        """
        updated = 0
        try:
            ccxt_orders = self.exchange.fetch_open_orders()
            # Строим map: ex_id -> ccxt_order
            ex_map = {o['id']: o for o in ccxt_orders}

            for client_id, order in list(self._pending_orders.items()):
                if order.order_id in ex_map:
                    ccxt_o = ex_map[order.order_id]
                    old_status = order.status
                    self._update_from_ccxt(order, ccxt_o)
                    if order.status != old_status:
                        updated += 1
                        logger.info(f"🔄 [SYNC] {order.symbol} {old_status.value} → {order.status.value}")
                        if order.is_filled:
                            self._on_filled(order)
                else:
                    # Ордера нет на бирже — вероятно, исполнился
                    # Пробуем fetch_order
                    try:
                        ccxt_o = self.exchange.fetch_order(order.order_id, order.symbol)
                        old_status = order.status
                        self._update_from_ccxt(order, ccxt_o)
                        if order.status != old_status:
                            updated += 1
                            logger.info(f"🔄 [SYNC] {order.symbol} (fetch) {old_status.value} → {order.status.value}")
                            if order.is_filled:
                                self._on_filled(order)
                    except Exception:
                        # Ордер не найден — проверяем баланс
                        if order.status in (OrderStatus.PENDING, OrderStatus.OPEN):
                            _prev_status = order.status
                            self._update_filled_from_balance(order, order.symbol, order.side, order.quantity)
                            if order.is_filled:
                                updated += 1
                                logger.info(f"🔄 [SYNC] {order.symbol}: не найден на бирже, filled (подтверждён балансом)")
                                self._on_filled(order)
                            elif order.status == OrderStatus.FAILED:
                                updated += 1
                                order.updated_at = time.time()
                                logger.info(f"🔄 [SYNC] {order.symbol}: не найден на бирже и не подтверждён балансом → FAILED")

            # Чистим завершённые из pending
            for client_id in list(self._pending_orders.keys()):
                if self._orders[client_id].is_filled or self._orders[client_id].is_failed:
                    self._pending_orders.pop(client_id, None)

        except Exception as e:
            logger.warning(f"⚠️ [SYNC] ошибка: {e}")

        return updated

    def retry_failed_orders(self) -> int:
        """Повторить отправку упавших ордеров.

        Returns:
            число успешно ретраенутых
        """
        retried = 0
        for client_id, order in list(self._orders.items()):
            if order.status == OrderStatus.FAILED and order.quantity > 0:
                # Проверяем age — ретраим только свежие (< 60с)
                if time.time() - order.created_at < 60:
                    # Повторяем с той же ценой
                    result = self.submit_order(
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=order.price,
                        direction=order.direction,
                        order_type=order.order_type,
                        reason=f"retry_{order.client_id}",
                        metadata=order.metadata,
                    )
                    if result.success:
                        retried += 1
                        logger.info(f"🔄 [RETRY] {order.symbol} → успешно")
                else:
                    # Старый — архивируем
                    self._archive_order(client_id)
        return retried

    def get_open_orders(self, symbol: Optional[str] = None,
                        direction: Optional[Direction] = None) -> List[Order]:
        """Получить открытые ордера (с фильтрацией)."""
        result = []
        for order in self._orders.values():
            if not order.is_open:
                continue
            if symbol and order.symbol != symbol:
                continue
            if direction and order.direction != direction:
                continue
            result.append(order)
        return result

    def get_pending_count(self) -> int:
        """Сколько ордеров ещё не подтверждено биржей."""
        return len(self._pending_orders)

    def get_stats(self) -> Dict:
        """Статистика по ордерам."""
        total = len(self._orders)
        open_count = sum(1 for o in self._orders.values() if o.is_open)
        filled = sum(1 for o in self._orders.values() if o.is_filled)
        failed = sum(1 for o in self._orders.values() if o.is_failed)
        return {
            'total': total,
            'open': open_count,
            'filled': filled,
            'failed': failed,
            'pending': len(self._pending_orders),
        }

    # ── Приватные методы ────────────────────────────────────────────────

    def _check_balance(self, symbol: str, side: str,
                       quantity: float, price: float) -> Tuple[bool, str]:
        """Проверить, хватит ли средств на ордер.

        ⚡ SPOT: как обычно (USDT для buy, base для sell).
        ⚡ MARGIN/FUTURES: проверяем свободную маржу (collateral).
        """
        try:
            balance = self.exchange.fetch_balance()
            order_value = quantity * price

            if self.mode != Mode.SPOT:
                # ═══ MARGIN/FUTURES: проверяем залог, а не актив ───────
                leverage = self._leverage.get(symbol, self.default_leverage)
                margin_needed = order_value / leverage

                # Для маржи: free_balance = доступный залог
                free_total = balance['free'].get('USDT', 0)

                # Также проверяем equity
                total_equity = balance.get('total', {}).get('USDT', 0)

                if free_total < margin_needed * 1.1:  # 10% запас
                    return False, (f"Недостаточно маржи: нужно ${margin_needed:.2f} при {leverage}x, "
                                   f"доступно ${free_total:.2f}")
                return True, ""

            # ═══ SPOT: оригинальная логика ─────────────────────────────
            if side == 'buy':
                free_usdt = balance['free'].get('USDT', 0)
                if free_usdt < order_value:
                    return False, (f"Недостаточно USDT: нужно ${order_value:.2f}, "
                                   f"доступно ${free_usdt:.2f}")
            else:  # sell
                currency = symbol.split('/')[0]
                free_asset = balance['free'].get(currency, 0)
                if free_asset < quantity * 0.999:
                    return False, (f"Недостаточно {currency}: нужно {quantity:.6f}, "
                                   f"доступно {free_asset:.6f}")
            return True, ""
        except Exception as e:
            logger.warning(f"⚠️ [BALANCE] ошибка проверки: {e}")
            return True, "balance_check_failed"  # пропускаем проверку

    def _update_from_ccxt(self, order: Order, ccxt_order: dict) -> None:
        """Обновить Order из ответа/статуса биржи."""
        ccxt_status = ccxt_order.get('status', 'open')
        order.order_id = ccxt_order.get('id', order.order_id)

        # CCXT status → наш OrderStatus
        status_map = {
            'open': OrderStatus.OPEN,
            'closed': OrderStatus.CLOSED,
            'canceled': OrderStatus.CANCELED,
            'expired': OrderStatus.EXPIRED,
            'rejected': OrderStatus.REJECTED,
            'partially_filled': OrderStatus.PARTIAL,
        }
        new_status = status_map.get(ccxt_status, OrderStatus.OPEN)
        order.status = new_status

        # Данные исполнения
        filled = ccxt_order.get('filled', 0)
        remaining = ccxt_order.get('remaining', 0)
        if filled:
            order.filled_qty = float(filled)
        if remaining:
            pass  # остаток в книге

        avg = ccxt_order.get('average', ccxt_order.get('price', 0))
        if avg:
            order.avg_price = float(avg)

        cost = ccxt_order.get('cost', 0)
        if cost:
            order.cost = float(cost)

        fee_data = ccxt_order.get('fee', {})
        if fee_data:
            order.fee = float(fee_data.get('cost', 0))

        order.updated_at = time.time()

    def _update_filled_from_balance(self, order, symbol: str, side: str, quantity: float) -> None:
        """Проверить, исполнился ли ордер, по изменению баланса (запасной метод).
        
        Используется когда fetch_order не может найти ордер на бирже
        (market ордер исполнился и исчез из active orders).
        """
        try:
            base_currency = symbol.replace('/USDT', '').replace('/USDC', '')
            bal = self.exchange.fetch_balance()
            
            # Если buy: проверяем что base currency появился
            if side == 'buy':
                base_free = float(bal.get(base_currency, {}).get('free', 0))
                usdt_free = float(bal.get('USDT', {}).get('free', 0))
                
                # Если токен появился (>50% от запрошенного) — считаем filled
                if base_free > quantity * 0.5:
                    order.status = OrderStatus.CLOSED
                    order.filled_qty = min(base_free, quantity)
                    order.avg_price = order.price  # используем цену из ордера
                    order.updated_at = time.time()
                    logger.info(f"✅ [BALANCE] {symbol} {side}: обнаружен {base_currency}={base_free:.4f} "
                               f"(ордер был на {quantity:.4f}), USDT={usdt_free:.2f}")
                    return
            
            # Если sell: проверяем что USDT увеличился
            if side == 'sell':
                usdt_free = float(bal.get('USDT', {}).get('free', 0))
                # TODO: нужен баланс до ордера для сравнения
                pass
                
        except Exception as e:
            logger.warning(f"[BALANCE] ошибка проверки {symbol}: {e}")
        
        # Если ничего не нашли — ордер не исполнился
        order.status = OrderStatus.FAILED

    def _on_filled(self, order: Order) -> None:
        """Ордер исполнился — зовём callback для PositionManager."""
        if self.on_order_filled:
            try:
                self.on_order_filled(order)
            except Exception as e:
                logger.error(f"❌ [FILLED] callback error: {e}")

    def _is_retryable_error(self, error_msg: str) -> bool:
        """Можно ли повторить ордер после этой ошибки."""
        non_retryable = [
            'Insufficient balance',
            'Insufficient funds',
            'Invalid quantity',
            'MIN_NOTIONAL',
            'LOT_SIZE',
            'PRICE_FILTER',
            'Order would trigger immediately.',
            'Account has insufficient balance',
        ]
        for phrase in non_retryable:
            if phrase.lower() in error_msg.lower():
                return False
        return True

    def _archive_order(self, client_id: str) -> None:
        """Переместить завершённый ордер в историю."""
        order = self._orders.pop(client_id, None)
        if order:
            self._history.append(order)

    def cleanup_old_orders(self, max_age_seconds: float = 86400) -> int:
        """Архивировать старые завершённые ордера (старше max_age)."""
        now = time.time()
        archived = 0
        for client_id in list(self._orders.keys()):
            order = self._orders[client_id]
            if order.is_filled or order.is_failed:
                if now - order.updated_at > max_age_seconds:
                    self._archive_order(client_id)
                    archived += 1
        return archived
