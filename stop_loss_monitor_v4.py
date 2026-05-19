#!/usr/bin/env python3
"""
НЕЗАВИСИМЫЙ МОНИТОР СТОП-ЛОССОВ V4
Профессиональная версия с трекером позиций в JSON.
Не парсит логи — использует файл positions.json для точного entry price.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import logging
import threading
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKER_PATH = os.path.join(BASE_DIR, "data/positions_tracker.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import ccxt
import numpy as np

# ТРЕЙЛИНГ-СТОП ПАРАМЕТРЫ
TRAIL_ACTIVATE_PCT = 2.0   # Активация трейла при +2% от entry
TRAIL_DISTANCE_PCT = 1.5   # Стоп на -1.5% ниже highest

# ─── HMM-режимы: динамические SL/TP ──────────────────────────────────────
REGIME_SL_TP = {
    0: {'sl': 3.0, 'tp': 8.0,   'trail_activate': 1.2, 'trail_dist': 0.8},  # CALM — быстро фиксируем
    1: {'sl': 5.0, 'tp': 10.0,  'trail_activate': 1.5, 'trail_dist': 1.0},  # NORMAL — фиксация с +1.5%
    2: {'sl': 8.0, 'tp': 14.0,  'trail_activate': 2.0, 'trail_dist': 1.5},  # VOLATILE — шире
}

_HMM_INSTANCE = None

def _get_hmm_regime():
    """Получить текущий HMM-режим (лениво)."""
    global _HMM_INSTANCE
    if _HMM_INSTANCE is None:
        try:
            from hmm_regime import get_regime
            _HMM_INSTANCE = get_regime()
        except Exception as e:
            logger.warning(f"[SL-MON] HMM не подключился: {e}")
            _HMM_INSTANCE = None
    return _HMM_INSTANCE

def get_regime_sl_tp():
    """Вернуть (sl_percent, tp_percent, trail_activate, trail_dist) по режиму."""
    r = _get_hmm_regime()
    if r and r.trained:
        state = r.current_state
        params = REGIME_SL_TP.get(state, REGIME_SL_TP[1])
        name = r.get_state_name()
        if state != 1:  # Логируем только если не NORMAL (чтобы не спамить)
            logger.info(f"[SL-MON] HMM режим: {name} → SL={params['sl']}%, TP={params['tp']}%")
        return params['sl'], params['tp'], params['trail_activate'], params['trail_dist']
    return 5.0, 10.0, 2.0, 1.5  # fallback


_ML_CACHE = {'model': None, 'time': 0}
_ML_PAIRS_CACHE = {}
_OHLCV_CACHE = {}

def _fetch_ohlcv_fast(symbol, timeframe='1h', limit=48):
    """Быстрая загрузка свечей с кэшем (10 сек)."""
    now = time.time()
    key = f'{symbol}_{timeframe}'
    if key in _OHLCV_CACHE and now - _OHLCV_CACHE[key].get('t', 0) < 10:
        return _OHLCV_CACHE[key]['data']
    try:
        import ccxt
        e = ccxt.bybit({'enableRateLimit':True,'options':{'defaultType':'spot'}})
        ohlcv = e.fetch_ohlcv(symbol, timeframe, limit)
        _OHLCV_CACHE[key] = {'data': ohlcv, 't': now}
        return ohlcv
    except:
        return None


def _quick_struct_exit(symbol, pnl_pct, entry_price, current_price):
    """
    Быстрый детектор разворота структуры без ML:
    1. HH/HL — пробой последнего минимума = разворот
    2. RSI — падение с перекупленности
    Возвращает (True, причина) если пора выходить.
    """
    # Загружаем 1H свечи
    ohlcv = _fetch_ohlcv_fast(symbol)
    if not ohlcv or len(ohlcv) < 20:
        return False, ""
    
    closes = [c[4] for c in ohlcv]
    highs = [c[2] for c in ohlcv]
    lows = [c[3] for c in ohlcv]  # c[3] = low в ccxt
    
    # 1️⃣ СТРУКТУРА HH/HL
    # Ищем последние 2 минимума (период 8 свечей)
    half = len(closes) // 2
    
    # Первая половина: минимум и максимум
    low1 = min(lows[half:])
    high1 = max(highs[:half])
    
    # Вторая половина: минимум
    low2 = min(lows[-8:])  # последние 8 свечей
    
    # Если текущая цена ниже low1 (первый минимум) — разворот
    struct_broken = current_price < low1 and low2 < low1
    
    # 2️⃣ RSI разворот
    def _rsi(cl, period=14):
        if len(cl) < period + 1:
            return 50
        gains = losses = 0
        for i in range(-period, 0):
            diff = cl[i] - cl[i-1]
            if diff > 0: gains += diff
            else: losses += abs(diff)
        avg_g = gains / period
        avg_l = losses / period if losses > 0 else 0.001
        rs = avg_g / avg_l
        return 100 - (100 / (1 + rs))
    
    rsi = _rsi(closes[-20:])
    
    # RSI разворот: был выше 65 → упал ниже 50
    # Берём последние 3 RSI
    rsi_now = _rsi(closes)
    rsi_prev = _rsi(closes[:-3])
    rsi_fall = rsi_prev > 60 and rsi_now < 50
    
    # Если оба признака — точно разворот
    if struct_broken and rsi_fall:
        return True, f"Структура сломана + RSI упал ({rsi_prev:.0f}→{rsi_now:.0f})"
    
    # Если структура сломана ИЛИ RSI резко упал — тоже выходим
    if struct_broken:
        return True, f"Структура сломана (low1={low1:.4f}, cur={current_price:.4f})"
    
    if rsi_fall:
        # Выходим только если PnL > 0 или минус небольшой
        if pnl_pct > -1:
            return True, f"RSI разворот ({rsi_prev:.0f}→{rsi_now:.0f})"
    
    return False, ""


def ml_exit_check(symbol, pnl_pct):
    """
    Проверить через ML, не пора ли выходить из позиции.
    Возвращает True если ML рекомендует EXIT.
    """
    now = time.time()
    
    # Загружаем ML лениво (раз в 30 сек максимум)
    if _ML_CACHE['model'] is None or now - _ML_CACHE['time'] > 30:
        try:
            sys.path.insert(0, BASE_DIR)
            from ml_professional_v2 import get_ml
            _ML_CACHE['model'] = get_ml()
            _ML_CACHE['time'] = now
        except Exception as e:
            _ML_CACHE['model'] = False
            logger.warning(f"[ML-EXIT] Не загрузился: {e}")
    
    ml = _ML_CACHE['model']
    if not ml or not ml.trained:
        return False
    
    # Кэшируем 1H/4H данные для пары (раз в 5 минут)
    if symbol not in _ML_PAIRS_CACHE or now - _ML_PAIRS_CACHE[symbol].get('time', 0) > 300:
        try:
            import ccxt
            exchange = ccxt.bybit({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'},
            })
            
            ohlcv_1h = exchange.fetch_ohlcv(symbol, '1h', 100)
            ohlcv_4h = exchange.fetch_ohlcv(symbol, '4h', 50)
            
            h1 = [{'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]} for c in ohlcv_1h]
            h4 = [{'o':c[1],'h':c[2],'l':c[3],'c':c[4],'v':c[5],'t':c[0]} for c in ohlcv_4h]
            
            # Синхронизируем
            t1e = h1[-1]['t']
            h4_sync = [c for c in h4 if c['t'] <= t1e]
            
            _ML_PAIRS_CACHE[symbol] = {
                'h1': h1,
                'h4': h4_sync,
                'time': now,
            }
        except Exception as e:
            logger.warning(f"[ML-EXIT] {symbol}: {e}")
            return False
    
    
    cached = _ML_PAIRS_CACHE.get(symbol)
    if not cached or not cached['h1'] or not cached['h4']:
        return False
    
    try:
        decision, prob, feats = ml.evaluate(
            symbol, None, cached['h1'][-200:], cached['h4'][-50:], 
            50, 'neutral', 50
        )
        if decision is None:
            return False
        
        # Порог выхода — ниже BUY_THRESHOLD из HMM
        # Если NORMAL (0.55): выходим если prob < 0.50 (сигнал слабее покупки)
        # Если CALM (0.50): выходим если prob < 0.45
        # Если VOLATILE (0.65): выходим если prob < 0.60
        from ml_professional_v2 import _get_regime, BUY_THRESHOLD
        regime = _get_regime()
        if regime and regime.trained:
            t = regime.get_thresholds()
            exit_threshold = t['buy'] - 0.05  # на 0.05 ниже порога покупки
        else:
            exit_threshold = 0.50  # fallback
        
        if pnl_pct >= 0:
            # В плюсе: фиксируем если ML разуверен (prob упала ниже покупки)
            need_exit = prob < exit_threshold
            reason = f"ML={prob:.2f} (<{exit_threshold}), PnL={pnl_pct:+.2f}%" if need_exit else ""
        else:
            # В минусе: выходим если ML считает что будет ещё хуже
            need_exit = prob < exit_threshold - 0.10  # <0.40
            reason = f"ML={prob:.2f} (<{exit_threshold-0.10}), PnL={pnl_pct:+.2f}%" if need_exit else ""
        
        if need_exit:
            logger.info(f"🧠 ML-ВЫХОД {symbol}: {reason}")
            logger.info(f"    feats: rf={feats.get('rsi_1h','?')}/{feats.get('rsi_4h','?')}, "
                       f"t={feats.get('trend_1h','?')}/{feats.get('trend_4h','?')}")
            return True
        
        # Логируем если в минусе с низкой prob
        if pnl_pct < -1 and prob < 0.40:
            logger.info(f"   ⚠️ {symbol}: PnL={pnl_pct:.1f}% ML-prob={prob:.2f} — близко к выходу")
            
    except Exception as e:
        logger.warning(f"[ML-EXIT] Ошибка оценки {symbol}: {e}")
    
    return False


def save_trade_to_tracker(symbol: str, side: str, quantity: float, price: float, pnl_before: float = None):
    """
    Сохраняет трейд в трекер позиций.
    Вызывается из основного кода при покупке/продаже.
    """
    os.makedirs(os.path.dirname(TRACKER_PATH), exist_ok=True)
    
    # Читаем текущий трекер
    tracker = {}
    if os.path.exists(TRACKER_PATH):
        try:
            with open(TRACKER_PATH, 'r') as f:
                tracker = json.load(f)
        except:
            pass
    
    if side == 'buy':
        # Средняя цена для DCA (если докупаем)
        if symbol in tracker:
            old = tracker[symbol]
            total_qty = old['quantity'] + quantity
            total_cost = old['quantity'] * old['entry_price'] + quantity * price
            new_entry = total_cost / total_qty
            tracker[symbol] = {
                'entry_price': new_entry,
                'quantity': total_qty,
                'entry_time': old['entry_time'],
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'highest_price': old.get('highest_price', price)
            }
            logger.info(f"📝 Трекер: обновлен {symbol}: entry=${new_entry:.4f}, qty={total_qty:.6f}")
        else:
            tracker[symbol] = {
                'entry_price': price,
                'quantity': quantity,
                'entry_time': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'highest_price': price
            }
            logger.info(f"📝 Трекер: добавлен {symbol}: entry=${price:.4f}, qty={quantity:.6f}")
    
    elif side == 'sell':
        if symbol in tracker:
            # Если продана часть — уменьшаем количество
            old = tracker[symbol]
            remaining = old['quantity'] - quantity
            if remaining <= 0.000001:
                del tracker[symbol]
                logger.info(f"📝 Трекер: удален {symbol} (полная продажа, PnL: {pnl_before:.2f}%)")
            else:
                tracker[symbol]['quantity'] = remaining
                tracker[symbol]['updated_at'] = datetime.now(timezone.utc).isoformat()
                logger.info(f"📝 Трекер: частичная продажа {symbol}, осталось {remaining:.6f}")
    
    # Записываем
    with open(TRACKER_PATH, 'w') as f:
        json.dump(tracker, f, indent=2, default=str)


class StopLossMonitor:
    """Независимый монитор стоп-лоссов с трекером позиций"""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.exchange = self._setup_exchange()
        self.running = True
        self.pairs = self.config['trading']['enabled_pairs']
        
        # Динамические SL/TP от HMM-режима
        sl_pct, tp_pct, trail_act, trail_dist = get_regime_sl_tp()
        self.sl_percent = sl_pct
        self.tp_percent = tp_pct
        
        logger.info("=" * 50)
        logger.info("🛡️ МОНИТОР СТОП-ЛОССОВ V4 ЗАПУЩЕН")
        logger.info(f"   Стоп-лосс: -{self.sl_percent}% | Тейк-профит: +{self.tp_percent}%")
        logger.info(f"   Динамика от HMM-режима (CALM→3%/8%, VOLATILE→8%/14%)")
        logger.info(f"   Отслеживаемых пар: {len(self.pairs)}")
        logger.info(f"   Трекер: {TRACKER_PATH}")
        logger.info("=" * 50)
        
        # При старте: очищаем трекер от мусора
        self._clean_tracker()
    
    def _clean_tracker(self):
        """Удалить из трекера позиции которых нет на бирже."""
        try:
            balance = self.exchange.fetch_balance()
            tracker = self.load_tracker()
            changed = False
            for symbol in list(tracker.keys()):
                currency = symbol.split('/')[0]
                qty = balance['total'].get(currency, 0)
                if qty < 0.000001:
                    logger.info(f"🧹 Очистка трекера: {symbol} (нет на бирже)")
                    del tracker[symbol]
                    changed = True
                else:
                    # Проверяем минимальную стоимость позиции
                    try:
                        ticker = self.exchange.fetch_ticker(symbol)
                        val = qty * ticker['last']
                        if val < 1.0:  # меньше $1 — мусор
                            logger.info(f"🧹 Очистка трекера: {symbol} (остаток ${val:.2f})")
                            del tracker[symbol]
                            changed = True
                    except:
                        pass
            
            if changed:
                with open(TRACKER_PATH, 'w') as f:
                    json.dump(tracker, f, indent=2, default=str)
                logger.info(f"🧹 Трекер очищен: осталось {len(tracker)} позиций")
        except Exception as e:
            logger.warning(f"[CLEAN] Ошибка очистки: {e}")
    
    def _setup_exchange(self):
        bybit_config = self.config['bybit']
        exchange = ccxt.bybit({
            'apiKey': bybit_config['api_key'],
            'secret': bybit_config['secret'],
            'password': bybit_config['password'],
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
        })
        return exchange
    
    def load_tracker(self):
        """Загрузка трекера позиций"""
        if not os.path.exists(TRACKER_PATH):
            return {}
        try:
            with open(TRACKER_PATH, 'r') as f:
                return json.load(f)
        except:
            return {}
    
    def get_real_price(self, symbol):
        """Получение реальной цены с биржи"""
        for attempt in range(3):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker['last']
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                logger.error(f"Ошибка цены {symbol}: {e}")
        return None
    
    def close_position_market(self, symbol, quantity, pnl_pct):
        """Рыночное закрытие позиции"""
        try:
            balance = self.exchange.fetch_balance()
            currency = symbol.split('/')[0]
            real_qty = balance['free'].get(currency, 0)
            
            if real_qty <= 0.000001:
                logger.warning(f"❌ Нет {currency} для продажи")
                # Очищаем трекер в любом случае
                save_trade_to_tracker(symbol, 'sell', quantity, 0, pnl_pct)
                return False
            
            # Округляем до точности монеты
            market = self.exchange.market(symbol)
            amount_precision = market.get('precision', {}).get('amount', 8)
            dec = int(-np.log10(amount_precision)) if amount_precision < 1 else 3
            safe_qty = round(real_qty * 0.999, min(dec, 10))
            min_notional = market.get('limits', {}).get('cost', {}).get('min', 1.0)
            safe_cost = safe_qty * current_price
            if safe_cost < min_notional and safe_cost > 0:
                logger.warning(f"🧹 {symbol}: остаток ${safe_cost:.2f} < мин ${min_notional}, чищу трекер")
                # Очищаем трекер (позиция уже продана на бирже)
                save_trade_to_tracker(symbol, 'sell', quantity, current_price, 0)
                return True
            order = self.exchange.create_order(
                symbol=symbol, type='market', side='sell', amount=safe_qty
            )
            
            # Получаем цену продажи
            filled_price = order.get('price') or order.get('average') or 0
            if 'fills' in order and order['fills']:
                total_cost = sum(f['cost'] for f in order['fills'])
                total_filled = sum(f['amount'] for f in order['fills'])
                if total_filled > 0:
                    filled_price = total_cost / total_filled
            
            logger.info(f"✅ ПРИНУДИТЕЛЬНО ЗАКРЫТ {symbol}: {safe_qty:.6f} @ ${filled_price:.4f}")
            logger.info(f"   Тип: рыночный ордер, ID: {order.get('id', 'N/A')}")
            
            # Обновляем трекер
            save_trade_to_tracker(symbol, 'sell', quantity, filled_price, pnl_pct)
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка закрытия {symbol}: {e}")
            return False
    
    def check_positions(self):
        """Проверка всех отслеживаемых позиций"""
        tracker = self.load_tracker()
        
        if not tracker:
            return  # Нет отслеживаемых позиций
        
        # Динамически обновляем SL/TP от HMM-режима (каждый раз)
        sl_pct, tp_pct, trail_act, trail_dist = get_regime_sl_tp()
        self.sl_percent = sl_pct
        self.tp_percent = tp_pct
        
        balance = self.exchange.fetch_balance()
        
        # Черный список: позиции которые мы уже пытались закрыть (защита от спама)
        closing_blacklist = set()
        
        for symbol, pos in list(tracker.items()):
            entry_price = pos['entry_price']
            quantity = pos['quantity']
            
            # Получаем реальную цену
            current_price = self.get_real_price(symbol)
            if current_price is None:
                continue
            
            # Проверяем, есть ли ещё актив на бирже
            currency = symbol.split('/')[0]
            real_qty = balance['total'].get(currency, 0)
            
            if real_qty < 0.000001:
                # Позиции нет на бирже — удаляем из трекера
                del tracker[symbol]
                logger.info(f"🧹 {symbol}: удален из трекера (нет на бирже)")
                save_trade_to_tracker(symbol, 'sell', quantity, current_price, 0)
                continue
            
            # Проверка остатков — если меньше 10% от нормальной позиции
            normal_qty = 10.0 / current_price  # ~$10 нормальная позиция
            if quantity < normal_qty * 0.1 and quantity > 0.000001:
                balance_dust = self.exchange.fetch_balance()
                real_dust = balance_dust['free'].get(currency, 0)
                if real_dust > 0.000001:
                    logger.info(f"🧹 ОСТАТОК {symbol}: продаю {real_dust:.6f} (quant={quantity:.6f})")
                    try:
                        market = self.exchange.market(symbol)
                        amount_precision = market.get('precision', {}).get('amount', 8)
                        dec = int(-np.log10(amount_precision)) if amount_precision < 1 else 3
                        safe_qty = round(real_dust * 0.999, min(dec, 10))
                        order = self.exchange.create_order(
                            symbol=symbol, type='market', side='sell', amount=safe_qty
                        )
                        logger.info(f"   ✅ Остаток продан: {safe_qty:.6f}")
                    except Exception as e:
                        logger.warning(f"   Ошибка продажи остатка: {e}")
                # Удаляем из трекера
                del tracker[symbol]
                save_trade_to_tracker(symbol, 'sell', quantity, current_price, 0)
                continue
            
            # Расчет PnL
            pnl_pct = (current_price - entry_price) / entry_price * 100
            
            # ТРЕЙЛИНГ-СТОП: обновляем highest_price
            highest = pos.get('highest_price', entry_price)
            if current_price > highest:
                highest = current_price
                tracker[symbol]['highest_price'] = highest
                with open(TRACKER_PATH, 'w') as f:
                    json.dump(tracker, f, indent=2, default=str)
            
            # Используем динамические параметры трейла от HMM
            # trail_act, trail_dist уже обновлены в check_positions()
            _, _, trail_act, trail_dist = get_regime_sl_tp()
            
            trail_active_pct = (highest - entry_price) / entry_price * 100
            if trail_active_pct >= trail_act:
                trail_stop = highest * (1 - trail_dist / 100.0)
                from_entry_to_stop = (trail_stop - entry_price) / entry_price * 100
                if current_price <= trail_stop:
                    logger.info(f"🔄 ТРЕЙЛИНГ-СТОП {symbol}: PnL={pnl_pct:.2f}% (max=${highest:.4f})")
                    logger.info(f"   Стоп по трейлу: ${trail_stop:.4f} ({from_entry_to_stop:+.2f}% от entry)")
                    self.close_position_market(symbol, quantity, pnl_pct)
                    time.sleep(3)
                    continue
            
            # БЫСТРЫЙ ДЕТЕКТОР РАЗВОРОТА (каждые 20 сек): структура HH/HL + RSI
            if trail_active_pct < trail_act and symbol not in closing_blacklist:
                ck = getattr(self, '_ml_check_count', 0)
                if ck % 4 == 0:
                    exit_signal, reason = _quick_struct_exit(symbol, pnl_pct, entry_price, current_price)
                    if exit_signal:
                        closing_blacklist.add(symbol)
                        logger.info(f"🔴 РАЗВОРОТ {symbol}: {reason}, PnL={pnl_pct:.2f}%, закрываю")
                        self.close_position_market(symbol, quantity, pnl_pct)
                        time.sleep(3)
                        continue
            
            # ML-ВЫХОД: проверяем не пора ли выйти по тренду (раз в 30 сек)
            ml_exit_triggered = False
            if trail_active_pct < trail_act and symbol not in closing_blacklist:
                ck = getattr(self, '_ml_check_count', 0)
                if ck % 6 == 0:
                    if ml_exit_check(symbol, pnl_pct):
                        ml_exit_triggered = True
                        closing_blacklist.add(symbol)
                        logger.info(f"🧠 ML-ВЫХОД {symbol}: PnL={pnl_pct:.2f}%, закрываю по тренду")
                        self.close_position_market(symbol, quantity, pnl_pct)
                        time.sleep(3)
            
            # Проверка обычного стоп-лосса (только если трейл не активирован)
            if not ml_exit_triggered and trail_active_pct < trail_act and pnl_pct <= -self.sl_percent:
                if symbol not in closing_blacklist:
                    closing_blacklist.add(symbol)
                    logger.info(f"🔴 СТОП-ЛОСС {symbol}: PnL={pnl_pct:.2f}% (порог: -{self.sl_percent}%)")
                    logger.info(f"   Entry: ${entry_price:.4f} → Текущая: ${current_price:.4f}")
                    self.close_position_market(symbol, quantity, pnl_pct)
                    time.sleep(3)
            
            # Проверка тейк-профита
            elif pnl_pct >= self.tp_percent:
                if symbol not in closing_blacklist:
                    closing_blacklist.add(symbol)
                    logger.info(f"🟢 ТЕЙК-ПРОФИТ {symbol}: PnL={pnl_pct:.2f}% (порог: +{self.tp_percent}%)")
                    logger.info(f"   Entry: ${entry_price:.4f} → Текущая: ${current_price:.4f}")
                    self.close_position_market(symbol, quantity, pnl_pct)
                    time.sleep(3)
            
            # Мониторинг (логируем если PnL > 1%)
            elif abs(pnl_pct) > 1:
                emoji = "🟢" if pnl_pct > 0 else "🟡"
                logger.info(f"   {emoji} {symbol}: PnL={pnl_pct:.2f}% (${current_price:.4f})")
    
    def run(self):
        """Основной цикл мониторинга"""
        self._ml_check_count = 0
        while self.running:
            try:
                self.check_positions()
                self._ml_check_count += 1
                time.sleep(5)
            except Exception as e:
                logger.error(f"Ошибка цикла: {e}")
                time.sleep(10)


def main():
    config_path = os.path.join(BASE_DIR, "config/api_config_final.json")
    
    monitor = StopLossMonitor(config_path)
    monitor.run()


if __name__ == "__main__":
    main()
