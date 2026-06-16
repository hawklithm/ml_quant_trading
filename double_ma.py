#!/usr/bin/env python3
"""
双均线策略完整回测 — 带参数优化 + 可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import sys

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.size"] = 11

TICKER = sys.argv[1] if len(sys.argv) > 1 else "000300.SS"

# 1. 获取数据
print(f"正在获取 {TICKER} 数据...")
df = yf.download(TICKER, period="3y", auto_adjust=True, progress=False)
df.columns = [c[0] for c in df.columns]
df["Returns"] = df["Close"].pct_change()

# 2. 参数扫描
print("参数扫描中...")
results = []
best_sharpe = -999
best_params = None

for short in range(5, 51, 5):        # 短期均线: 5~50
    for long in range(40, 201, 10):   # 长期均线: 40~200
        if short >= long:
            continue

        df["SMA_S"] = df["Close"].rolling(short).mean()
        df["SMA_L"] = df["Close"].rolling(long).mean()
        df["Pos"] = 0
        df.loc[df["SMA_S"] > df["SMA_L"], "Pos"] = 1
        strat_ret = df["Pos"].shift(1) * df["Returns"]

        sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) * 0.5  # 惩罚

        bh_sharpe = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252)

        cum = (1 + strat_ret).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()

        results.append({
            "short": short, "long": long,
            "sharpe": sharpe, "max_dd": max_dd,
        })

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = (short, long)

# 3. 最佳参数回测
print(f"\n最佳参数: SMA{best_params[0]} / SMA{best_params[1]}  (夏普={best_sharpe:.2f})")

short, long = best_params
df["SMA_S"] = df["Close"].rolling(short).mean()
df["SMA_L"] = df["Close"].rolling(long).mean()
df["Pos"] = 0
df.loc[df["SMA_S"] > df["SMA_L"], "Pos"] = 1
df["Strat_Ret"] = df["Pos"].shift(1) * df["Returns"]

# 4. 可视化
fig, axes = plt.subplots(3, 1, figsize=(14, 11), gridspec_kw={"height_ratios": [3, 1.5, 1.5]})

# 价格+均线
ax = axes[0]
ax.plot(df.index, df["Close"], label="收盘价", color="#2196F3", lw=1, alpha=0.7)
ax.plot(df.index, df["SMA_S"], label=f"SMA{short}", color="#FF9800", lw=1.2)
ax.plot(df.index, df["SMA_L"], label=f"SMA{long}", color="#F44336", lw=1.2)
# 标记买卖点
buy_signals = df[(df["Pos"] == 1) & (df["Pos"].shift(1) == 0)].index
sell_signals = df[(df["Pos"] == 0) & (df["Pos"].shift(1) == 1)].index
ax.scatter(buy_signals, df.loc[buy_signals, "Close"], marker="^", color="green", s=80, label="买入", zorder=5)
ax.scatter(sell_signals, df.loc[sell_signals, "Close"], marker="v", color="red", s=80, label="卖出", zorder=5)
ax.set_title(f"{TICKER} 双均线策略 (SMA{short}/{long})", fontsize=13, fontweight="bold")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
ax.set_ylabel("价格")

# 净值曲线
ax = axes[1]
bh = (1 + df["Returns"]).cumprod()
strat = (1 + df["Strat_Ret"]).cumprod()
ax.plot(df.index, bh, label="买入持有", color="#999", lw=1, alpha=0.6)
ax.plot(df.index, strat, label="策略收益", color="#4CAF50", lw=1.5)
ax.fill_between(df.index, 1, strat.values, alpha=0.1, color="#4CAF50")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
ax.set_ylabel("净值")
ax.set_title(f"策略总收益: {(strat.iloc[-1]-1)*100:.1f}%  vs  买入持有: {(bh.iloc[-1]-1)*100:.1f}%")

# 回撤
ax = axes[2]
cum = strat.cummax()
dd = (strat - cum) / cum
ax.fill_between(df.index, 0, dd * 100, color="#F44336", alpha=0.4)
ax.set_ylabel("回撤 (%)")
ax.set_title(f"最大回撤: {dd.min()*100:.1f}%")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("backtest_result.png", dpi=150, bbox_inches="tight")
print(f"回测图表已保存: backtest_result.png")
print()

# 5. 热力图
results_df = pd.DataFrame(results)
pivot = results_df.pivot_table(index="short", columns="long", values="sharpe", aggfunc="first")

fig2, ax2 = plt.subplots(figsize=(10, 7))
im = ax2.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0)
ax2.set_xticks(range(len(pivot.columns)))
ax2.set_yticks(range(len(pivot.index)))
ax2.set_xticklabels(pivot.columns, fontsize=8)
ax2.set_yticklabels(pivot.index, fontsize=8)
ax2.set_xlabel("长期均线参数")
ax2.set_ylabel("短期均线参数")
ax2.set_title("夏普比率参数热力图", fontsize=13, fontweight="bold")

# 标记最佳
r_idx = list(pivot.index).index(short)
c_idx = list(pivot.columns).index(long)
ax2.scatter(c_idx, r_idx, marker="*", color="black", s=200, zorder=5,
            label=f"最佳: SMA{short}/{long}")
ax2.legend(fontsize=11)

plt.colorbar(im, ax=ax2, label="夏普比率")
plt.tight_layout()
plt.savefig("sharpe_heatmap.png", dpi=150, bbox_inches="tight")
print(f"参数热力图已保存: sharpe_heatmap.png")
print()
print("完成! 打开 .png 文件查看结果。")
