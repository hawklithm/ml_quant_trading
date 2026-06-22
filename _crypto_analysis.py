#!/usr/bin/env python3
"""Crypto ML analysis - same framework as v5 stock system"""
import yfinance as yf
import numpy as np
import pandas as pd
import json
from datetime import datetime

now = datetime.now()

# ─── Download data ───
all_data = {}
for t in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
    df = yf.download(t, period='1y', auto_adjust=True, progress=False)
    if df.empty:
        df = yf.download(t, period='6mo', auto_adjust=True, progress=False)
    if df.empty:
        print(f"{t}: NO DATA")
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    all_data[t] = df
    print(f"{t}: {len(df)} rows, ${df['Close'].iloc[-1]:,.2f}")

print()

results = []
for t in ['BTC-USD', 'ETH-USD', 'SOL-USD']:
    if t not in all_data:
        continue
    df = all_data[t]
    c = df['Close'].values.astype(float)
    v = df['Volume'].values.astype(float)
    ret = np.diff(np.log(c))
    name = t.replace('-USD', '')
    price = c[-1]

    # Momenta
    mom_21d = (c[-1] / c[-22] - 1) * 100 if len(c) >= 22 else 0
    mom_63d = (c[-1] / c[-64] - 1) * 100 if len(c) >= 64 else 0
    mom_126d = (c[-1] / c[-127] - 1) * 100 if len(c) >= 127 else 0
    mom_252d = (c[-1] / c[-253] - 1) * 100 if len(c) >= 253 else 0

    # Volatility
    vol_7d = float(np.std(ret[-7:]) * np.sqrt(365) * 100) if len(ret) >= 7 else 0
    vol_21d = float(np.std(ret[-21:]) * np.sqrt(365) * 100) if len(ret) >= 21 else 0
    vol_63d = float(np.std(ret[-63:]) * np.sqrt(365) * 100) if len(ret) >= 63 else vol_21d

    # Max drawdown (6mo)
    peak = np.maximum.accumulate(c[-126:])
    dd = (c[-126:] - peak) / peak
    max_dd = float(dd.min() * 100)

    # RSI(14)
    gains = ret[-14:].copy()
    losses = -gains.copy()
    losses[losses < 0] = 0
    gains[gains < 0] = 0
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses)) if np.mean(losses) > 0 else 0.001
    rsi = float(100 - 100 / (1 + avg_gain / avg_loss))

    # Volume
    vol_21d_avg = float(np.mean(v[-21:]))
    vol_63d_avg = float(np.mean(v[-63:]))
    vol_ratio = vol_21d_avg / vol_63d_avg if vol_63d_avg > 0 else 1.0
    vol_trend_ratio = vol_21d / vol_63d if vol_63d > 0 else 1.0

    # ═══ Sub-scores ═══
    mom_score = np.clip(0.6 * max(0, (mom_21d + 25) / 50) + 0.4 * max(0, (mom_63d + 35) / 70), 0, 1)
    vol_score = max(0, 1 - vol_21d / 100)
    rsi_score = 1 - abs(rsi - 55) / 45
    vol2_score = max(0, 1 - abs(vol_ratio - 1.15) / 0.5)
    dd_score = max(0, 1 - abs(max_dd) / 50)
    vt_score = max(0, 1 - abs(vol_trend_ratio - 1) / 0.5)

    # Trend strength (EMA alignment)
    ema10 = float(np.mean(c[-10:]))
    ema30 = float(np.mean(c[-30:])) if len(c) >= 30 else ema10
    ema60 = float(np.mean(c[-60:])) if len(c) >= 60 else ema30
    if c[-1] > ema10 > ema30 > ema60:
        trend = 0.8
    elif c[-1] > ema10 and ema10 > ema30:
        trend = 0.5
    elif c[-1] < ema10 < ema30 < ema60:
        trend = 0.1
    elif c[-1] > ema10:
        trend = 0.3
    else:
        trend = 0.2

    # Composite score (mirrors v5 logic)
    score = (0.30 * mom_score + 0.15 * vol_score + 0.15 * rsi_score +
             0.10 * vol2_score + 0.20 * trend + 0.05 * dd_score + 0.05 * vt_score)

    # Direction
    if score > 0.55:
        direction = '看涨'
    elif score < 0.40:
        direction = '看跌'
    else:
        direction = '震荡'

    results.append({
        'name': name, 'price': round(price, 2), 'score': round(score, 4),
        'direction': direction,
        'mom_21d': round(mom_21d, 2), 'mom_63d': round(mom_63d, 2),
        'mom_126d': round(mom_126d, 2), 'mom_252d': round(mom_252d, 2),
        'vol_7d': round(vol_7d, 1), 'vol_21d': round(vol_21d, 1), 'vol_63d': round(vol_63d, 1),
        'rsi': round(rsi, 1), 'max_dd': round(max_dd, 1),
        'vol_ratio': round(vol_ratio, 2), 'vol_trend': round(vol_trend_ratio, 2),
        'sub_mom': round(mom_score, 4), 'sub_vol': round(vol_score, 4),
        'sub_rsi': round(rsi_score, 4), 'sub_trend': trend,
        'sub_vol2': round(vol2_score, 4), 'sub_dd': round(dd_score, 4),
        'sub_vt': round(vt_score, 4),
    })

    print(f"══════════ {name} ══════════")
    print(f"  价格: ${price:,.2f}")
    print(f"  综合评分: {results[-1]['score']:.4f} | 方向: {results[-1]['direction']}")
    print(f"  动量: 21d={mom_21d:+.2f}% 63d={mom_63d:+.2f}% 126d={mom_126d:+.2f}% 1y={mom_252d:+.2f}%")
    print(f"  波动率: 7d={vol_7d:.0f}% 21d={vol_21d:.0f}% 63d={vol_63d:.0f}%")
    print(f"  RSI: {rsi:.1f} | 回撤: {max_dd:.1f}% | 量比: {vol_ratio:.2f}x | 波动率趋势: {vol_trend_ratio:.2f}x")
    print(f"  子项: 动量{mom_score:.3f} 波动率{vol_score:.3f} RSI{rsi_score:.3f} 趋势{trend:.1f} 量能{vol2_score:.3f} 回撤{dd_score:.3f} 波动趋势{vt_score:.3f}")
    print()

# Ranking
results.sort(key=lambda x: -x['score'])
print("══════════ 排名 ══════════")
for i, r in enumerate(results):
    print(f"  #{i+1} {r['name']:>4} 评分{r['score']:.4f} {r['direction']} ${r['price']:>8,.2f}")

# ═══ Correlation ═══
print("\n══════════ 60日日收益相关性 ══════════")
tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
for i in range(3):
    for j in range(i + 1, 3):
        t1, t2 = tickers[i], tickers[j]
        if t1 not in all_data or t2 not in all_data:
            continue
        r1 = np.diff(np.log(all_data[t1]['Close'].values.astype(float)[-61:]))
        r2 = np.diff(np.log(all_data[t2]['Close'].values.astype(float)[-61:]))
        corr = np.corrcoef(r1, r2)[0, 1]
        print(f"  {t1[:3]} ↔ {t2[:3]}: {corr:.3f}")

# ═══ Distance from high ═══
print("\n══════════ 偏离半年高点 ══════════")
for t in tickers:
    if t not in all_data:
        continue
    c = all_data[t]['Close'].values.astype(float)
    hi = np.max(c[-126:])
    dist = (c[-1] / hi - 1) * 100
    print(f"  {t[:3]}: 偏离高点 {dist:+.1f}% (高点 ${hi:,.0f})")

# ═══ Recent 7-day returns ═══
print("\n══════════ 近7日日收益 ══════════")
for t in tickers:
    if t not in all_data:
        continue
    c = all_data[t]['Close'].values.astype(float)
    rets = np.diff(np.log(c[-8:])) * 100
    print(f"  {t[:3]}: {' '.join(f'{r:+.2f}%' for r in rets)}")

# ═══ Key support/resistance (Fibonacci) ═══
print("\n══════════ 关键技术位 (斐波那契) ══════════")
for t in tickers:
    if t not in all_data:
        continue
    c = all_data[t]['Close'].values.astype(float)
    hi = np.max(c[-126:])
    lo = np.min(c[-126:])
    rng = hi - lo
    fib = {
        '0.236': hi - 0.236 * rng,
        '0.382': hi - 0.382 * rng,
        '0.500': hi - 0.500 * rng,
        '0.618': hi - 0.618 * rng,
        '0.786': hi - 0.786 * rng,
    }
    name = t[:3]
    print(f"  {name}: 当前${c[-1]:,.0f}")
    print(f"    阻力: 0.236=${fib['0.236']:,.0f}  0.382=${fib['0.382']:,.0f}  0.500=${fib['0.500']:,.0f}")
    print(f"    支撑: 0.618=${fib['0.618']:,.0f}  0.786=${fib['0.786']:,.0f}  (半年低${lo:,.0f})")
