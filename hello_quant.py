#!/usr/bin/env python3
"""
量化交易入门脚本 — 拉数据 + 画K线 + 均线策略回测
用法: python hello_quant.py [股票代码]

股票代码默认使用 A股 '000300.SS' (沪深300), 也可以换成:
  - 'AAPL'     苹果
  - 'BTC-USD'  比特币
  - '000001.SS' 上证指数
  - '600519.SS' 贵州茅台
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["font.size"] = 12

def main():
    # 1. 获取股票代码
    ticker = sys.argv[1] if len(sys.argv) > 1 else "000300.SS"

    print(f"=== 量化交易入门: {ticker} ===")
    print("正在获取数据...")

    # 2. 拉取最近 2 年数据
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    if df.empty:
        print(f"错误：无法获取 {ticker} 的数据。试试其他股票代码。")
        sys.exit(1)

    # yfinance 返回 MultiIndex columns，展平
    df.columns = [c[0] for c in df.columns]

    print(f"数据区间: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"交易日数: {len(df)}")
    print(f"最新收盘: {df['Close'].iloc[-1]:.2f}")
    print(f"区间涨幅: {(df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100:.1f}%")
    print()

    # 3. 统计特征
    df["Returns"] = df["Close"].pct_change()
    daily_vol = df["Returns"].std() * np.sqrt(252)  # 年化波动率
    sharpe = (df["Returns"].mean() / df["Returns"].std()) * np.sqrt(252)

    print("--- 基础统计 ---")
    print(f"年化波动率: {daily_vol*100:.1f}%")
    print(f"日夏普比率(年化): {sharpe:.2f}")
    print(f"最大单日涨幅: {df['Returns'].max()*100:.2f}%")
    print(f"最大单日跌幅: {df['Returns'].min()*100:.2f}%")

    # 4. 计算两条均线
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA60"] = df["Close"].rolling(60).mean()

    # 5. 绘制 K线(简化:收盘价) + 均线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df.index, df["Close"], label="收盘价", color="#2196F3", linewidth=1, alpha=0.8)
    ax1.plot(df.index, df["SMA20"], label="SMA20 (20日均线)", color="#FF9800", linewidth=1.2)
    ax1.plot(df.index, df["SMA60"], label="SMA60 (60日均线)", color="#F44336", linewidth=1.2)
    ax1.set_title(f"{ticker} — 简单均线策略演示", fontsize=14, fontweight="bold")
    ax1.set_ylabel("价格")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # 成交量
    ax2.bar(df.index, df["Volume"], color="#4CAF50", alpha=0.5, width=1)
    ax2.set_ylabel("成交量")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("chart.png", dpi=150, bbox_inches="tight")
    print(f"\n图表已保存: {ticker}_chart.png")

    # 6. 简单回测: SMA20 上穿 SMA60 做多，下穿平仓
    df["Position"] = 0
    df.loc[df["SMA20"] > df["SMA60"], "Position"] = 1
    df["Signal"] = df["Position"].diff()
    df["Strategy_Returns"] = df["Position"].shift(1) * df["Returns"]

    # 累计净值
    buy_hold = (1 + df["Returns"]).cumprod()
    strategy = (1 + df["Strategy_Returns"]).cumprod()

    # 回测指标
    strat_sharpe = (df["Strategy_Returns"].mean() / df["Strategy_Returns"].std()) * np.sqrt(252)
    total_return = strategy.iloc[-1] - 1
    bh_return = buy_hold.iloc[-1] - 1

    # 最大回撤
    cum_max = strategy.cummax()
    drawdown = (strategy - cum_max) / cum_max
    max_dd = drawdown.min()

    # 交易次数
    n_trades = int(df["Signal"].abs().sum())

    print()
    print("=== 均线策略回测结果 (SMA20/60 金叉死叉) ===")
    print(f"{'指标':<20} {'策略':>12} {'买入持有':>12}")
    print("-" * 44)
    print(f"{'总收益率':<20} {total_return*100:>10.1f}% {bh_return*100:>10.1f}%")
    print(f"{'年化夏普比率':<20} {strat_sharpe:>12.2f} {'N/A':>12}")
    print(f"{'最大回撤':<20} {max_dd*100:>10.1f}% {'N/A':>12}")
    print(f"{'交易次数':<20} {n_trades:>12} {'N/A':>12}")

    print()
    print("=== 下一步 ===")
    print("  python double_ma.py    — 双均线策略完整回测")
    print("  python explore.py      — 策略参数扫描")
    print("  source .venv/bin/activate && jupyter lab  — 交互式研究")
    print()

    # 显示图表（CLI环境无法弹窗，已保存文件）
    return df


if __name__ == "__main__":
    df = main()
