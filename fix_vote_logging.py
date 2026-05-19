#!/usr/bin/env python3
"""
Fix vote logging in DecisionEngine — adds vote history write before VETO returns.
Patches decision_engine.py by line number (index-based) to handle exact whitespace.
"""
import re

with open('/home/ksysha/.openclaw/industrial_super_system/decision_engine.py', 'r') as f:
    lines = f.readlines()

changes = 0

# 1. Add _save_veto_vote method before line 544 (def _calc_trend_from_candles)
# Line 544 is: '    \n' (4 spaces + newline)
# Line 545 is: '    def _calc_trend_from_candles...
# We insert the method between 544's line break and 545

veto_method_lines = [
    '\n',
    '    def _save_veto_vote(self, symbol: str, current_price: float, veto_reason: str, side: str = \'long\') -> None:\n',
    '        """Сохранить запись голоса для VETO-случаев (чтобы дашборд не зависал)."""\n',
    '        try:\n',
    '            now = time.time()\n',
    '            if not hasattr(self, \'_last_vote_ts\'):\n',
    '                self._last_vote_ts = {}\n',
    '            last_ts = self._last_vote_ts.get(symbol, 0)\n',
    '            if now - last_ts < 3.0:\n',
    '                return\n',
    '            self._last_vote_ts[symbol] = now\n',
    '            vote_record = {\n',
    '                \'ts\': datetime.now(timezone.utc).isoformat(),\n',
    '                \'symbol\': symbol,\n',
    '                \'approved\': False,\n',
    '                \'final_score\': 0.0,\n',
    '                \'threshold\': 0,\n',
    '                \'price\': round(current_price, 6),\n',
    '                \'veto_reason\': veto_reason,\n',
    '                \'votes\': {},\n',
    '                \'strong_count\': 0,\n',
    '            }\n',
    '            vote_log_path = os.path.join(BASE_DIR, \'data\', \'vote_history.json\')\n',
    '            os.makedirs(os.path.dirname(vote_log_path), exist_ok=True)\n',
    '            history = []\n',
    '            if os.path.exists(vote_log_path):\n',
    '                try:\n',
    '                    with open(vote_log_path, \'r\') as f:\n',
    '                        history = json.load(f)\n',
    '                except:\n',
    '                    history = []\n',
    '            history.append(vote_record)\n',
    '            if len(history) > 10000:\n',
    '                history = history[-10000:]\n',
    '            with open(vote_log_path, \'w\') as f:\n',
    '                json.dump(history, f, indent=2, default=str)\n',
    '        except Exception as e:\n',
    '            pass  # тихо — не критично\n',
]

# Insert before line 545 (1-indexed) = index 544 (0-indexed)
insert_idx = 544  # before line 545 (1-indexed)
# But we need to keep the blank line spacing proper
# Actually, inserting before the def on line 545. line 544 is '    \n'
# We'll replace line 544 entirely
assert lines[543] == '            )\n', f"Expected ) at index 543, got: {repr(lines[543])}"
assert lines[544] == '    \n', f"Expected spaces at index 544, got: {repr(lines[544])}"
assert 'def _calc_trend_from_candles' in lines[545], f"Expected def at 545, got: {repr(lines[545])}"

# Replace line 544 with the method + a blank line
new_544 = ['    \n'] + veto_method_lines + ['    \n']
lines[544:545] = new_544
changes += 1
print("✓ Added _save_veto_vote method")

# Now lines shifted. Need to re-find each VETO return by content

# Helper function
def find_line(pattern, start=None):
    for i, line in enumerate(lines):
        if start is not None and i < start:
            continue
        if pattern in line:
            return i
    return -1

# 2. Max positions veto
idx = find_line("return Decision(symbol, 'hold', side=side, reason=f\"Максимум")
assert idx >= 0, "Cannot find max_positions veto"
# Add vote save before this line
vote_save_line = '            self._save_veto_vote(symbol, current_price, f"Максимум {max_positions} позиций", side)\n'
lines.insert(idx, vote_save_line)
changes += 1
print(f"✓ Added vote save for max_positions veto (line {idx+1})")

# 3. Cooldown veto
idx = find_line("return Decision(")  # first occurrence after cooldown
# Find the exact cooldown block: look for "Повторный вход"
idx = find_line("Повторный вход")
assert idx >= 0, "Cannot find cooldown veto"
# Before this line, we need to extract the reason
# Find the 'if time_since < self.reentry_cooldown:' line
idx_if = find_line("if time_since < self.reentry_cooldown:")
assert idx_if >= 0
# After the if, we have:
#   remain = self.reentry_cooldown - time_since
#   return Decision(...)
# We need to change from direct return to reason + vote save + return

# Find the exact lines
idx_remain = find_line("remain = self.reentry_cooldown - time_since", start=idx_if)
idx_return = find_line("return Decision(", start=idx_remain)
# Add reason assignment before remain and modify the return
assert idx_remain >= 0
assert idx_return >= 0

# We need to: add 'reason = ...' before remain, and change return to use reason
# First, find the end of the Decision(...) block
# The block spans multiple lines. Let's check.
idx_return_line = idx_return
end_of_block = idx_return_line
for j in range(idx_return_line, min(idx_return_line + 10, len(lines))):
    if lines[j].strip().endswith(')') and not lines[j].strip().startswith('#'):
        end_of_block = j
        break

# Replace: add reason assignment
lines.insert(idx_remain, f'                reason = f"Повторный вход через {{{{remain/60:.1f}}}} мин"\n')
# Adjust: the return line was originally without reason extraction, now it uses reason
# But the reason f-string has {{ which needs proper handling
# Actually, let me just replace the block entirely
for j in range(idx_remain, end_of_block + 1):
    lines[j] = None  # mark for removal

# Insert new block
new_block = [
    '                remain = self.reentry_cooldown - time_since\n',
    '                reason = f"Повторный вход через {remain/60:.1f} мин"\n',
    '                self._save_veto_vote(symbol, current_price, reason, side)\n',
    '                return Decision(\n',
    '                    symbol, \'hold\', side=side,\n',
    '                    reason=reason\n',
    '                )\n',
]
lines[idx_remain:end_of_block+1] = new_block
changes += 1
print(f"✓ Added vote save for cooldown veto")

# Re-find everything (indices shifted)
idx = find_line("VETO: BTC упал")
if idx >= 0:
    # Add reason before it
    for j in range(idx-10, idx):
        if 'return Decision(symbol' in lines[j] and 'btc_change_4h' in lines[j]:
            # This is the return line - change it
            lines[j] = f'                reason = f"VETO: BTC упал на {btc_change_4h:.1f}% за период"\n'
            vote_line = f'                self._save_veto_vote(symbol, current_price, reason, side)\n'
            lines.insert(j+1, vote_line)
            changes += 1
            print(f"✓ Added vote save for BTC drop veto")
            break
else:
    # Might already be done or different format
    print("  BTC drop veto skip:", idx)

print(f"\n✅ Total changes: {changes}")
if changes > 0:
    with open('/home/ksysha/.openclaw/industrial_super_system/decision_engine.py', 'w') as f:
        f.writelines(lines)
    print("✅ File written!")
