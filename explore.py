#!/usr/bin/env python3
"""
多标的策略探索 — 一次看十几个标的的统计特征
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# 一组代表性标的
TICKERS = {
    "A股指数": ["000300.SS", "000001.SS", "399001.SZ"],
    "美股": ["SPY", "QQQ", "DIA"],
    "加密货币": ["BTC-USD", "ETH-USD"],
    "商品": ["GC=F", "CL=F"],
    "汇率": ["EURUSD=X", "JPY=X"],
}

print(f"{'标的':<14} {'年化收益':>10} {'年化波动':>10} {'夏普比率':>10} {'最大回撤':>10} {'日均成交量':>12}")
print("-" * 68)

for category, tickers in TICKERS.items():
    print(f"\n--- {category} ---")
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
            if df.empty:
                continue
            df.columns = [c[0] for c in df.columns]
            ret = df["Close"].pct_change().dropna()

            ann_ret = (df["Close"].iloc[-1] / df["Close"].iloc[0]) - 1
            ann_vol = ret.std() * np.sqrt(252)
            sharpe = (ret.mean() / ret.std()) * np.sqrt(252)

            cum = (1 + ret).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            max_dd = dd.min()

            # 日均成交量(美元估算)
            avg_vol = df["Volume"].mean()
            if "BTC" in ticker or "ETH" in ticker:
                avg_price = df["Close"].mean()
                avg_vol = avg_vol * avg_price  # 转美元
            if "JPY" in ticker:
                avg_vol_str = "N/A"
            else:
                if avg_vol > 1e9:
                    avg_vol_str = f"${avg_vol/1e9:.1f}B"
                elif avg_vol > 1e6:
                    avg_vol_str = f"${avg_vol/1e6:.1f}M"
                else:
                    avg_vol_str = f"${avg_vol/1e3:.0f}K"

            print(f"{ticker:<14} {ann_ret*100:>8.1f}% {ann_vol*100:>8.1f}% {sharpe:>8.2f} {max_dd*100:>8.1f}% {avg_vol_str:>12}")
        except Exception as e:
            print(f"{ticker:<14} 错误: {str(e)[:30]}")

print()
print("提示：一些标的可能因地区限制无法获取数据。")
print("试试用 A 股数据平台（如 Tushare/AkShare）获取更全面的数据。")
