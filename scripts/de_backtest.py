#!/usr/bin/env python3
"""
Бэктест DecisionEngine: прошёлся бы по историческим сигналам,
купил бы по цене входа, зафиксировал бы результат через 1/4/24 часа.

Использует реальные данные из лога — парсит сигналы трейдера и
проверяет что было бы если бы DE принимал решение.
"""

import sys
import os
import json
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict

BASE_DIR = '/home/ksysha/.openclaw/industrial_super_system'
sys.path.insert(0, BASE_DIR)

LOG_FILE = '/tmp/system_v4.log'

# Будем симулировать DecisionEngine
from decision_engine import DecisionEngine
de = DecisionEngine()

def parse_signals_from_log():
    """
    Парсит лог: находит все строки с 'Уверенность=', извлекает
    символ, цену, тренд, уверенность, RSI, время.
    """
    signals = []
    pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?'
        r'(\w+/USDT).*?'
        r'Цена=\$?([0-9.]+).*?'
        r'Тренд=(\w+).*?'
        r'Уверенность=([0-9.]+)%.*?'
        r'RSI=([0-9.]+)'
    )
    
    with open(LOG_FILE, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                ts_str, symbol, price_str, trend, conf_str, rsi_str = m.groups()
                signals.append({
                    'timestamp': datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S'),
                    'symbol': symbol,
                    'price': float(price_str),
                    'trend': trend,
                    'confidence': float(conf_str),
                    'rsi': float(rsi_str),
                })
    
    return signals


def check_de_decision(signal):
    """Спрашивает DecisionEngine: одобрил бы вход?"""
    try:
        decision = de._evaluate_entry_ensemble(
            symbol=signal['symbol'],
            confidence=signal['confidence'],
            trend=signal['trend'],
            rsi=signal['rsi'],
            current_price=signal['price'],
        )
        return decision.get('approved', False), decision.get('final_score', 0)
    except:
        return False, 0


def get_price_history(symbol, start_ts, hours_ahead):
    """
    Эмуляция: получает цену из кеша или лога через N часов после сигнала.
    Используем кеш (WebSocket) — он живёт в процессе, поэтому для
    исторических данных используем второй проход по логу.
    """
    return None  # будет заполнено во втором проходе


def backtest():
    print("=" * 60)
    print("📊 БЭКТЕСТ DECISION ENGINE")
    print("=" * 60)
    
    # Фаза 1: собираем сигналы
    print("\n⏳ Парсим сигналы из лога...")
    all_signals = parse_signals_from_log()
    print(f"   Найдено: {len(all_signals)} сигналов")
    
    if not all_signals:
        print("❌ Нет сигналов для анализа")
        return
    
    # Фаза 2: фильтруем — только уникальные (уникальный символ + временное окно)
    # Группируем по символу, берём каждый уникальный сигнал
    seen = set()
    unique_signals = []
    for s in all_signals:
        key = (s['symbol'], s['timestamp'].strftime('%Y%m%d%H%M'))
        if key not in seen:
            seen.add(key)
            unique_signals.append(s)
    
    print(f"   Уникальных: {len(unique_signals)}")
    
    # Фаза 3: спрашиваем DE
    print("\n⏳ Спрашиваем DecisionEngine...")
    de_approved = []
    de_rejected = []
    
    for s in unique_signals:
        approved, score = check_de_decision(s)
        if approved:
            de_approved.append({**s, 'score': score})
        else:
            de_rejected.append({**s, 'score': score})
    
    print(f"\n📊 Результаты DE:")
    print(f"   Одобрено: {len(de_approved)}")
    print(f"   Отклонено: {len(de_rejected)}")
    print(f"   Всего проверок: {len(de_approved) + len(de_rejected)}")
    
    if de_approved:
        # Топ-10 одобренных
        de_approved.sort(key=lambda x: x['score'], reverse=True)
        print(f"\n🏆 ТОП-10 одобренных сигналов:")
        print(f"   {'Score':>5} | {'Символ':>10} | {'Цена':>8} | {'Тренд':>12} | {'RSI':>5} | {'Время'}")
        print(f"   {'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*12}-+-{'-'*5}-+-{'-'*16}")
        for s in de_approved[:10]:
            t = s['timestamp'].strftime('%H:%M')
            print(f"   {s['score']:5.0f} | {s['symbol']:>10} | ${s['price']:<6.2f} | {s['trend']:>12} | {s['rsi']:5.1f} | {t}")
    
    # Фаза 4: проверка цены через 1 час после сигнала
    print(f"\n⏳ Проверяем что было через 1 час...")
    
    # Строим индекс цен по времени из лога
    price_index = defaultdict(list)
    for s in all_signals:
        price_index[s['symbol']].append((s['timestamp'], s['price']))
    
    results = {'profit': 0, 'loss': 0, 'flat': 0, 'total_pnl': 0.0, 'details': []}
    
    for s in de_approved:
        target_time = s['timestamp'] + timedelta(hours=4)
        prices = price_index.get(s['symbol'], [])
        
        # Ищем цену через ~1 час
        future_price = None
        for ts, p in prices:
            if abs((ts - target_time).total_seconds()) < 300:  # ±5 мин
                future_price = p
                break
        
        if future_price is not None:
            change_pct = (future_price - s['price']) / s['price'] * 100
            if change_pct > 0.5:
                results['profit'] += 1
            elif change_pct < -0.5:
                results['loss'] += 1
            else:
                results['flat'] += 1
            results['total_pnl'] += change_pct
            results['details'].append({
                'symbol': s['symbol'],
                'entry': s['price'],
                'future': future_price,
                'change': change_pct,
                'score': s['score'],
                'time': s['timestamp'].strftime('%H:%M'),
            })
    
    if results['details']:
        print(f"\n📈 РЕЗУЛЬТАТ через 1 час:")
        print(f"   Прибыльных: {results['profit']}")
        print(f"   Убыточных:  {results['loss']}")
        print(f"   Нейтральных: {results['flat']}")
        print(f"   Всего:       {len(results['details'])}")
        total = len(results['details'])
        if total > 0:
            print(f"   Win Rate:    {results['profit']/total*100:.1f}%")
            print(f"   Средний PnL: {results['total_pnl']/total:.2f}%")
            print(f"   Общий PnL:   {results['total_pnl']:.2f}%")
        
        # Топ-5 лучших и худших
        results['details'].sort(key=lambda x: x['change'], reverse=True)
        print(f"\n🥇 ТОП-5 лучших:")
        for r in results['details'][:5]:
            print(f"   +{r['change']:.2f}% | {r['symbol']} | ${r['entry']:.2f}→${r['future']:.2f} | score={r['score']:.0f}")
        
        print(f"\n🥇 ТОП-5 худших:")
        for r in results['details'][-5:]:
            print(f"   {r['change']:.2f}% | {r['symbol']} | ${r['entry']:.2f}→${r['future']:.2f} | score={r['score']:.0f}")
    
    # Фаза 5: распределение по символам
    print(f"\n📊 ПО СИМВОЛАМ:")
    sym_stats = defaultdict(lambda: {'count': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0})
    for r in results['details']:
        sym = r['symbol']
        sym_stats[sym]['count'] += 1
        sym_stats[sym]['pnl'] += r['change']
        if r['change'] > 0.5:
            sym_stats[sym]['wins'] += 1
        elif r['change'] < -0.5:
            sym_stats[sym]['losses'] += 1
    
    for sym, stats in sorted(sym_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = stats['wins']/stats['count']*100 if stats['count'] > 0 else 0
        print(f"   {sym:>10}: {stats['count']:3d} сигналов | WR {wr:5.1f}% | PnL {stats['pnl']:+.2f}% | {stats['wins']}✅/{stats['losses']}❌")


if __name__ == '__main__':
    backtest()
