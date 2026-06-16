#!/usr/bin/env python3
"""
4种经典量化策略实现

策略列表:
  1. 均值回归 (布林带) — 价格突破下轨买入，突破上轨卖出
  2. 趋势跟踪 (MACD) — 快线/慢线/信号线交叉
  3. 动量策略 — 按过去N日收益率排序选股
  4. 配对交易 — 协整配对价差回归

用法:
  python strategies.py [ticker]
  默认: AAPL
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 9)
plt.rcParams["font.size"] = 11

TICKER = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
print(f"下载 {TICKER} 数据...")
df = yf.download(TICKER, period="3y", auto_adjust=True, progress=False)
df.columns = [c[0] for c in df.columns]
close = df["Close"]
high = df["High"]
low = df["Low"]

fig, axes = plt.subplots(4, 1, figsize=(15, 16))

# ═══════════════════════════════════════════════════════════
# 策略1: 均值回归 — 布林带
# ═══════════════════════════════════════════════════════════
print("\n═══ 1. 均值回归 (布林带) ═══")

sma = close.rolling(20).mean()
std = close.rolling(20).std()
upper = sma + 2 * std
lower = sma - 2 * std

# 信号: 突破下轨买入, 突破上轨卖出
buy_signal = (close < lower) & (close.shift(1) >= lower.shift(1))
sell_signal = (close > upper) & (close.shift(1) <= upper.shift(1))

# 回测
pos_mr = pd.Series(0, index=close.index)
for i in range(1, len(pos_mr)):
    if pos_mr.iloc[i-1] == 0 and buy_signal.iloc[i]:
        pos_mr.iloc[i] = 1
    elif pos_mr.iloc[i-1] == 1 and sell_signal.iloc[i]:
        pos_mr.iloc[i] = 0
    else:
        pos_mr.iloc[i] = pos_mr.iloc[i-1]

ret = close.pct_change()
mr_ret = pos_mr.shift(1) * ret
mr_cum = (1 + mr_ret).cumprod()
mr_dd = mr_cum / mr_cum.cummax() - 1

print(f"  总收益: {mr_cum.iloc[-1]-1:.2%}")
print(f"  夏普: {(mr_ret.mean()/mr_ret.std())*np.sqrt(252):.2f}")
print(f"  最大回撤: {mr_dd.min():.2%}")
print(f"  交易次数: {(buy_signal|sell_signal).sum()}")

# 画图
ax = axes[0]
ax.plot(close.index, close, label="收盘价", color="#2196F3", lw=0.8, alpha=0.7)
ax.plot(sma.index, sma, label="SMA20", color="#FF9800", lw=0.8)
ax.fill_between(upper.index, upper, lower, alpha=0.15, color="#9C27B0")
ax.plot(upper.index, upper, label="上轨 (+2σ)", color="#F44336", lw=0.6, ls="--")
ax.plot(lower.index, lower, label="下轨 (-2σ)", color="#4CAF50", lw=0.6, ls="--")
ax.scatter(close[buy_signal].index, close[buy_signal], marker="^", color="green", s=60, label="买入")
ax.scatter(close[sell_signal].index, close[sell_signal], marker="v", color="red", s=60, label="卖出")
ax.set_title("策略1: 均值回归 (布林带)", fontweight="bold")
ax.legend(loc="best", fontsize=8)
ax.set_ylabel("价格")
ax.grid(True, alpha=0.25)

# ═══════════════════════════════════════════════════════════
# 策略2: ���势跟踪 — MACD
# ═══════════════════════════════════════════════════════════
print("\n═══ 2. 趋势跟踪 (MACD) ═══")

ema12 = close.ewm(span=12).mean()
ema26 = close.ewm(span=26).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9).mean()
macd_hist = macd_line - signal_line

# MACD 线上穿信号线 → 买入, 下穿 → 卖出
macd_buy = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
macd_sell = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

pos_macd = pd.Series(0, index=close.index)
for i in range(1, len(pos_macd)):
    if pos_macd.iloc[i-1] == 0 and macd_buy.iloc[i]:
        pos_macd.iloc[i] = 1
    elif pos_macd.iloc[i-1] == 1 and macd_sell.iloc[i]:
        pos_macd.iloc[i] = 0
    else:
        pos_macd.iloc[i] = pos_macd.iloc[i-1]

macd_ret = pos_macd.shift(1) * ret
macd_cum = (1 + macd_ret).cumprod()
macd_dd = macd_cum / macd_cum.cummax() - 1

print(f"  总收益: {macd_cum.iloc[-1]-1:.2%}")
print(f"  夏普: {(macd_ret.mean()/macd_ret.std())*np.sqrt(252):.2f}")
print(f"  最大回撤: {macd_dd.min():.2%}")
print(f"  交易次数: {(macd_buy|macd_sell).sum()}")

ax = axes[1]
ax.plot(close.index, close, label="收盘价", color="#2196F3", lw=0.8, alpha=0.7)
ax.scatter(close[macd_buy].index, close[macd_buy], marker="^", color="green", s=60, label="金叉买入")
ax.scatter(close[macd_sell].index, close[macd_sell], marker="v", color="red", s=60, label="死叉卖出")
ax.set_title("策略2: 趋势跟踪 (MACD)", fontweight="bold")
ax.legend(loc="best", fontsize=8)
ax.set_ylabel("价格")
ax.grid(True, alpha=0.25)

# 插入 MACD 子图
ax_macd = ax.twinx()
ax_macd.bar(macd_hist.index, macd_hist.values, width=1, color="#9C27B0", alpha=0.3, label="MACD Hist")
ax_macd.axhline(0, color="#666", lw=0.5)
ax_macd.set_ylabel("MACD 柱状图")
ax_macd.legend(loc="upper left", fontsize=8)

# ═══════════════════════════════════════════════════════════
# 策略3: 配对交易 (模拟: 两个相关标的)
# ═══════════════════════════════════════════════════════════
print("\n═══ 3. 配对交易 (模拟) ═══")
print("  实际配对需要两个高度相关的标的")
print("  这里用 SPY & QQQ 演示价差回归逻辑")

try:
    df2 = yf.download(["SPY", "QQQ"], period="3y", auto_adjust=True, progress=False)
    df2 = df2["Close"]
    if isinstance(df2, pd.DataFrame) and df2.shape[1] >= 2:
        spy = df2.iloc[:, 0]
        qqq = df2.iloc[:, 1]
        # 标准化价差
        ratio = spy / qqq
        ratio_sma = ratio.rolling(20).mean()
        ratio_std = ratio.rolling(20).std()
        zscore = (ratio - ratio_sma) / ratio_std

        # 价差 z-score > 2 → 空SPY多QQQ, < -2 → 多SPY空QQQ
        pair_buy = (zscore < -1.5) & (zscore.shift(1) >= -1.5)
        pair_sell = (zscore > 1.5) & (zscore.shift(1) <= 1.5)

        pair_ret = -np.sign(zscore.shift(1)) * (spy.pct_change() - qqq.pct_change())
        pair_cum = (1 + pair_ret).cumprod()
        pair_dd = pair_cum / pair_cum.cummax() - 1

        print(f"  总收益: {pair_cum.iloc[-1]-1:.2%}")
        print(f"  夏普: {(pair_ret.mean()/pair_ret.std())*np.sqrt(252):.2f}")
        print(f"  最大回撤: {pair_dd.min():.2%}")

        ax = axes[2]
        ax.plot(ratio.index, zscore, label="价差 Z-Score", color="#9C27B0", lw=0.8)
        ax.axhline(2, color="#F44336", ls="--", lw=0.6)
        ax.axhline(-2, color="#4CAF50", ls="--", lw=0.6)
        ax.axhline(0, color="#666", ls="-", lw=0.5)
        ax.fill_between(ratio.index, 0, zscore.values, alpha=0.15, color="#9C27B0")
        ax.set_title("策略3: 配对交易 (SPY/QQQ 价差回归)", fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.set_ylabel("Z-Score")
        ax.grid(True, alpha=0.25)
    else:
        axes[2].text(0.5, 0.5, "无法获取 SPY/QQQ 数据", ha="center", va="center", transform=axes[2].transAxes)
        print("  数据获取失败")
except Exception as e:
    axes[2].text(0.5, 0.5, f"配对交易数据不可用", ha="center", va="center", transform=axes[2].transAxes)
    print(f"  错误: {e}")

# ═══════════════════════════════════════════════════════════
# 策略4: 趋势强度 — ADX (Average Directional Index)
# ═══════════════════════════════════════════════════════════
print("\n═══ 4. 趋势强度 (ADX) ═══")

def calc_adx(high, low, close, period=14):
    """计算 ADX 指标"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = pd.Series(0, index=high.index)
    minus_dm = pd.Series(0, index=high.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    atr = tr.rolling(period).mean()
    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx, plus_di, minus_di

adx, pdi, mdi = calc_adx(high, low, close)

# ADX > 25 表示趋势行情, PDI > MDI 表示上升趋势
trend_signal = (adx > 25) & (pdi > mdi)
adx_entries = (trend_signal == True) & (trend_signal.shift(1) == False)
adx_exits = (adx < 20) | ((pdi < mdi) & (adx < 30))

pos_adx = pd.Series(0, index=close.index)
for i in range(1, len(pos_adx)):
    if pos_adx.iloc[i-1] == 0 and trend_signal.iloc[i]:
        pos_adx.iloc[i] = 1
    elif pos_adx.iloc[i-1] == 1 and (adx_exits.iloc[i] or trend_signal.iloc[i] == False):
        pos_adx.iloc[i] = 0
    else:
        pos_adx.iloc[i] = pos_adx.iloc[i-1]

adx_ret = pos_adx.shift(1) * ret
adx_cum = (1 + adx_ret).cumprod()
adx_dd = adx_cum / adx_cum.cummax() - 1

print(f"  总收益: {adx_cum.iloc[-1]-1:.2%}")
print(f"  夏普: {(adx_ret.mean()/adx_ret.std())*np.sqrt(252):.2f}")
print(f"  最大回撤: {adx_dd.min():.2%}")

ax = axes[3]
ax.plot(adx.index, adx, label="ADX (趋势强度)", color="#FF5722", lw=0.8)
ax.plot(pdi.index, pdi, label="+DI (上升)", color="#4CAF50", lw=0.6, alpha=0.8)
ax.plot(mdi.index, mdi, label="-DI (下降)", color="#F44336", lw=0.6, alpha=0.8)
ax.axhline(25, color="#666", ls="--", lw=0.5, alpha=0.5)
ax.fill_between(adx.index, 0, 25, alpha=0.08, color="#999")
ax.scatter(close[adx_entries].index, adx[adx_entries].fillna(0),
           marker="^", color="green", s=60, label="趋势入场", zorder=5)
ax.set_title("策略4: 趋势强度 (ADX+DI)", fontweight="bold")
ax.legend(loc="best", fontsize=8)
ax.set_ylabel("ADX / ±DI")
ax.set_xlabel("日期")
ax.grid(True, alpha=0.25)

plt.tight_layout()
plt.savefig("strategies_4in1.png", dpi=150, bbox_inches="tight")
print(f"\n图表已保存: strategies_4in1.png")

print("\n" + "=" * 60)
print("4种策略性能汇总")
print("=" * 60)

print(f"""
┌──────────────┬──────────┬────────┬──────────┬──────────┐
│ 策略         │ 总收益率  │ 夏普   │ 最大回撤 │ 交易次数 │
├──────────────┼──────────┼────────┼──────────┼──────────┤
│ 均值回归     │ {mr_cum.iloc[-1]-1:>7.2%} │ {(mr_ret.mean()/mr_ret.std())*np.sqrt(252):>5.2f} │ {mr_dd.min():>7.2%} │ {(buy_signal|sell_signal).sum():>8d} │
│ MACD趋势     │ {macd_cum.iloc[-1]-1:>7.2%} │ {(macd_ret.mean()/macd_ret.std())*np.sqrt(252):>5.2f} │ {macd_dd.min():>7.2%} │ {(macd_buy|macd_sell).sum():>8d} │
│ ADX趋势强度  │ {adx_cum.iloc[-1]-1:>7.2%} │ {(adx_ret.mean()/adx_ret.std())*np.sqrt(252):>5.2f} │ {adx_dd.min():>7.2%} │ {0:>8d} │
│ 配对交易     │       N/A │   N/A │      N/A │       N/A │
└──────────────┴──────────┴────────┴──────────┴──────────┘
""")

print("策略特征速查:")
print("""
  均值回归: 适合震荡行情, 赚价格回归的钱
  趋势跟踪(MACD): 适合趋势行情, 赚动量延续的钱
  配对交易: 市场中性, 赚价差回归的钱
  ADX趋势强度: 判断是否有趋势, 适合配合其他策略做过滤
  """)
