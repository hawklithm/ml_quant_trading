#!/usr/bin/env python3
"""
回测v2：调整动量陷阱阈值 — 收紧门槛，减少假阳性
动量陷阱: m3>15(原8) 且 m1<-8(原-5) → 更高确信度才覆写
动量修复: m3<-15(原-8) 且 m1>8(原5)
"""
import pandas as pd, numpy as np, os

PRED_DIR = os.path.expanduser('~/.cache/hermes-quant/backtest/US/predictions')
DETAIL_DIR = os.path.expanduser('~/.cache/hermes-quant/backtest/US/daily_detail')

days = ['2026-07-01', '2026-07-02']

# 四种策略
strategies = [
    ('原分类器(基准)', lambda d,m1,m3: d),
    ('收紧版保守', lambda d,m1,m3: 
         '看涨' if d == '看跌' and m1 > 5 and m3 > 5 else
         '看跌' if d == '看涨' and m1 < -8 and m3 < -8 else d),
    ('陷阱规则松版(m3>8,m1<-5)', lambda d,m1,m3:
         '看跌' if m3 > 8 and m1 < -5 and d != '看跌' else
         '看涨' if m3 < -8 and m1 > 5 and d != '看涨' else
         '看涨' if d == '看跌' and m1 > 5 and m3 > 5 else
         '看跌' if d == '看涨' and m1 < -8 and m3 < -8 else d),
    ('陷阱规则紧版(m3>15,m1<-8)', lambda d,m1,m3:
         '看跌' if m3 > 15 and m1 < -8 and d != '看跌' else
         '看涨' if m3 < -15 and m1 > 8 and d != '看涨' else
         '看涨' if d == '看跌' and m1 > 5 and m3 > 5 else
         '看跌' if d == '看涨' and m1 < -8 and m3 < -8 else d),
]

print('=' * 100)
for name, fn in strategies:
    print(f'{name:>40}', end='   ')

total_all = 0
results = {s[0]: {'correct': 0, 'switches': 0, 'correct_sw': 0, 'flips': []} for s in strategies}

for day in days:
    pred_file = os.path.join(PRED_DIR, f'pred_US_{day}.csv')
    detail_file = os.path.join(DETAIL_DIR, f'detail_{day}.csv')
    if not os.path.exists(pred_file) or not os.path.exists(detail_file):
        continue
    
    preds = pd.read_csv(pred_file)
    detail = pd.read_csv(detail_file)
    merged = preds.merge(detail[['ticker','actual_chg','actual_dir']], on='ticker', how='inner')
    total = len(merged)
    total_all += total
    
    day_results = {s[0]: {'correct': 0, 'switches': 0, 'flips': []} for s in strategies}
    
    for _, row in merged.iterrows():
        m1 = row.get('mom_1m', 0) or 0
        m3 = row.get('mom_3m', 0) or 0
        orig_dir = row['direction']
        actual_chg = float(row['actual_chg']) if not pd.isna(row['actual_chg']) else 0
        actual_dir = row['actual_dir']
        ticker = row['ticker']
        
        for name, fn in strategies:
            new_dir = fn(orig_dir, m1, m3)
            changed = (new_dir != orig_dir)
            
            # Evaluate
            if new_dir == actual_dir:
                day_results[name]['correct'] += 1
                if changed:
                    day_results[name]['correct_sw'] = day_results[name].get('correct_sw', 0) + 1
            elif actual_dir == '震荡' and abs(actual_chg) < 1.0:
                day_results[name]['correct'] += 1
            
            if changed:
                day_results[name]['switches'] += 1
                day_results[name]['flips'].append(f'{ticker}({m1:+.0f}%1m/{m3:+.0f}%3m {orig_dir}->{new_dir} 实{actual_dir}{actual_chg:+.1f}%)')
    
    print(f'\n📅 {day}')
    for name, fn in strategies:
        r = day_results[name]
        acc = r['correct']/total*100
        print(f'  {name:<30}: {r["correct"]:>2}/{total}={acc:>5.1f}% 改写{r["switches"]:>2}只')
        # Merge into totals
        results[name]['correct'] += r['correct']
        results[name]['switches'] += r['switches']
        results[name]['correct_sw'] = results[name].get('correct_sw',0) + r.get('correct_sw',0)
        results[name]['flips'].extend(r['flips'])

print(f'\n{"=" * 100}')
print(f'🏆 合计 {total_all}只')
for name, fn in strategies:
    r = results[name]
    acc = r['correct']/total_all*100
    flips_str = ', '.join(r['flips'][:8])
    remaining = max(0, len(r['flips']) - 8)
    print(f'  {name:<30}: {r["correct"]:>2}/{total_all}={acc:>5.1f}% 改写{r["switches"]:>2}只')
    if r['flips']:
        print(f'    → {flips_str}{f"…还有{remaining}个" if remaining else ""}')

# Show which stocks benefited from tight trap rules vs were hurt
print(f'\n{"=" * 100}')
print(f'📊 紧版陷阱规则 vs 保守版对比')
tight = results['陷阱规则紧版(m3>15,m1<-8)']
cons = results['收紧版保守']
diff = tight['correct'] - cons['correct']
print(f'  紧版陷阱: {tight["correct"]} vs 保守: {cons["correct"]} = 差异{diff:+d}')
print(f'  紧版多改写了 {tight["switches"] - cons["switches"]} 只')
print(f'  其中正确覆写 {tight["correct_sw"]}/{tight["switches"]}')
