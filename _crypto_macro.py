#!/usr/bin/env python3
"""Crypto macro context"""
import yfinance as yf
import numpy as np
import pandas as pd

# BTC MA analysis
for t in ['BTC-USD', 'ETH-USD']:
    df = yf.download(t, period='1y', auto_adjust=True, progress=False)
    if df.empty:
        continue
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    c = df['Close'].values.astype(float)
    price = c[-1]
    ma200 = float(np.mean(c[-200:])) if len(c)>=200 else 0
    ma50 = float(np.mean(c[-50:])) if len(c)>=50 else 0
    ma20 = float(np.mean(c[-20:]))
    print(f'{t[:3]}: ${price:,.0f}')
    print(f'  MA20=${ma20:,.0f} MA50=${ma50:,.0f} MA200=${ma200:,.0f}')
    print(f'  离MA200: {(price/ma200-1)*100:+.1f}% 离MA50: {(price/ma50-1)*100:+.1f}%')
    if ma50 > ma200:
        print(f'  金叉: MA50 > MA200 ✅')
    else:
        print(f'  死叉: MA50 < MA200 ❌')
    print()
