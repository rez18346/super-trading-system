"""
Vote Parser — чистая замена _parse_votes_from_log() в control_api.py.

Читает /tmp/system_v4.log, находит последнюю DE→ строку для каждой пары
и извлекает все поля голосования, включая RP и ВХОД/Score= форматы.
"""
import re
import os

SYSTEM_LOG_FILE = '/tmp/system_v4.log'

# Поле ALL_SYMBOLS (только торгуемые)
ALL_SYMBOLS = ['BTC','ETH','SOL','XRP','ADA','AVAX','DOT','DOGE','LINK',
               'ATOM','ALGO','EGLD','APT','ARB','ROSE','MNT','FIL','NEAR','OP']

def parse_votes() -> dict:
    """Возвращает {symbol: {...score, mlpro, adv, mtf, rvb, liq, vv, cvd, rp, ...}}."""
    if not os.path.exists(SYSTEM_LOG_FILE):
        return {}

    # Читаем хвост лога (2MB)
    fsize = os.path.getsize(SYSTEM_LOG_FILE)
    chunk = min(fsize, 2 * 1024 * 1024)  # 2MB всегда

    with open(SYSTEM_LOG_FILE, 'r', errors='replace') as f:
        f.seek(max(0, fsize - chunk))
        if f.tell() > 0:
            f.readline()  # skip partial
        lines = f.readlines()

    if not lines:
        return {}

    result = {}
    prices = {}  # цена из лога

    # Проходим С КОНЦА (свежие строки первыми)
    for line in reversed(lines):
        # Извлекаем цены
        pm = re.search(r'(\w+)/USDT:?\s*Цена=\$([\d.]+)', line)
        if pm:
            prices[pm.group(1) + '/USDT'] = float(pm.group(2))

        # Пропускаем не-DE→ строки
        if 'DE→' not in line:
            continue

        sym_m = re.search(r'\[DE→(HOLD|BUY|SELL|CANDIDATE|SHORT)\] (\w+)/USDT:', line)
        if not sym_m:
            continue

        raw_sym = sym_m.group(2)
        sym = raw_sym + '/USDT'
        if sym in result:
            continue  # уже нашли

        signal = sym_m.group(1)

        # === Основной формат (Score= или ВХОД) ===
        pf = re.search(
            r'ML-Pro:(\d+)\((.+?)\)\s+'
            r'Adv:(\d+)\((.+?)\)\s+'
            r'MTF:(\d+).*?RVB:(\d+).*?Liq:(\d+)\([^)]*\)\s+'
            r'VV:(\d+)\(.+?\)\s+'
            r'VSA:\d+\([^)]*\)\s+'
            r'CVD:(\d+)\(([^)]*)\)'
            r'(?:\s+RP:([+-]\d+)\(([^)]*)\))?',
            line
        )

        if pf:
            vs_match = re.search(r'score=(\d+)', line)

            # IDM/OB
            idm_m = re.search(r'OB\d+\s+(idm|ext)(?:\s+(bullish|bearish))?', line)
            idm_info = ''
            if idm_m:
                ot = idm_m.group(1)
                if ot == 'idm':
                    dr = idm_m.group(2)
                    idm_info = f'IDM{"↑" if dr == "bullish" else "↓" if dr == "bearish" else ""}' if dr else 'IDM'
                else:
                    idm_info = 'EXT'

            # OI heat
            oi_m = re.search(r'OI🔥(\d+)\(\+(\d+)\)', line)
            oi_heat = int(oi_m.group(1)) if oi_m else 0
            oi_bonus = int(oi_m.group(2)) if oi_m else 0

            # bonus/rev/btc
            bm = re.search(r'bonus=([-\d]+) rev=([-\d]+) btc=([-+]\d+)', line)
            bonus = int(bm.group(1)) if bm else 0
            rev = int(bm.group(2)) if bm else 0
            btc = int(bm.group(3)) if bm else 0

            # RP
            rp = f"{pf.group(11)}({pf.group(12)})" if pf.lastindex and pf.lastindex >= 11 else '0(N/A)'

            # Score: из Score= или score=
            score_str = re.search(r'Score=(\d+)', line)
            if score_str:
                score = int(score_str.group(1))
            elif vs_match:
                score = int(vs_match.group(1))
            else:
                score = 0

            result[sym] = {
                'score': score,
                'signal': signal,
                'mlpro': f"{pf.group(1)}({pf.group(2)})",
                'adv': f"{pf.group(3)}({pf.group(4)})",
                'mtf': int(pf.group(5)),
                'rvb': int(pf.group(6)),
                'liq': int(pf.group(7)),
                'vv': int(pf.group(8)),
                'cvd': f"{pf.group(9)}({pf.group(10)})",
                'rp': rp,
                'bonus': bonus,
                'rev': rev,
                'btc': btc,
                'idm': idm_info,
                'oi_heat': oi_heat,
                'oi_bonus': oi_bonus,
                'price': prices.get(sym, 0),
                'direction': 'SHORT' if signal == 'SHORT' else 'LONG',
            }

        # === Формат DE→BUY (для входа) ===
        bm2 = re.search(r'\[DE→BUY\] (\w+)/USDT:\s*score=(\d+)/100.*?ML-Pro:(\d+)\((.+?)\)', line)
        if bm2 and sym not in result:
            score = int(bm2.group(2))
            result[sym] = {
                'score': score,
                'signal': 'BUY',
                'mlpro': f"{bm2.group(3)}({bm2.group(4)})",
                'price': prices.get(sym, 0),
                'direction': 'LONG',
            }

        if len(result) >= len(ALL_SYMBOLS):
            break

    return result


if __name__ == '__main__':
    import json
    v = parse_votes()
    print(f"Symbols: {len(v)}")
    for sym in sorted(v.keys())[:5]:
        data = v[sym]
        print(f"  {sym}: score={data.get('score')}, rp={data.get('rp')}, signal={data.get('signal')}")
    print("...")
    for sym in sorted(v.keys())[5:]:
        data = v[sym]
        print(f"  {sym}: score={data.get('score')}, rp={data.get('rp')}, signal={data.get('signal')}")
    print(f"\nJSON preview:")
    print(json.dumps(list(v.items())[:2], indent=2, ensure_ascii=False)[:400])
