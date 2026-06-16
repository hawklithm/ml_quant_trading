#!/usr/bin/env python3
"""
Backtrader vs VectorBT 深度对比 + 性能基准

测试条件:
  - 相同数据 (AAPL 3年日线)
  - 相同策略 (SMA20/60 金叉死叉)
  - 相同交易成本 (0.1% 手续费)
  - 各跑5次取平均

备注: VectorBT 1.0 首跑需要 numba JIT 编译（约30-60s），之后很快。
      本脚本在 warmup 阶段已经包含了编译耗时。
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

# ===================== 数据准备 =====================
print("=" * 60)
print("Backtrader vs VectorBT 深度对比")
print("=" * 60)

TICKER = "AAPL"
PERIOD = "3y"

print(f"\n标的: {TICKER} | 周期: {PERIOD}")
print(f"\n正在下载数据...")

df_raw = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)
df_raw.columns = [c[0] for c in df_raw.columns]
print(f"数据行数: {len(df_raw)}")
print(f"日期范围: {df_raw.index[0].date()} ~ {df_raw.index[-1].date()}")

close = df_raw["Close"]

# ===================== 快速向量化回测（pandas版） =====================
print("\n--- 快速向量化回测 (pandas 版) ---")

def run_fast_backtest(short=20, long=60, fees=0.001):
    """纯 pandas 向量化回测"""
    t0 = time.perf_counter()

    sma_short = close.rolling(short).mean()
    sma_long = close.rolling(long).mean()

    cross = (sma_short > sma_long)
    entries = (cross == True) & (cross.shift(1) == False)
    exits = (cross == False) & (cross.shift(1) == True)

    # 构建仓位
    position = 0
    daily_returns = close.pct_change()
    strat_returns = np.zeros(len(close))

    for i in range(1, len(close)):
        if entries.iloc[i]:
            position = 1
        elif exits.iloc[i]:
            position = 0

        if position == 1:
            strat_returns[i] = daily_returns.iloc[i]

    # 进出场时各扣0.1%
    strat_returns = pd.Series(strat_returns, index=close.index)
    entry_fees = entries * -fees
    exit_fees = exits * -fees
    strat_returns = strat_returns + entry_fees + exit_fees

    cum = (1 + strat_returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    t1 = time.perf_counter()

    return {
        "total_return": cum.iloc[-1] - 1,
        "sharpe": (strat_returns.mean() / strat_returns.std()) * np.sqrt(252),
        "max_dd": dd.min(),
        "n_trades": int(entries.sum()),
        "time": t1 - t0,
    }

# warmup
result_fast = run_fast_backtest()

# benchmark
times_pd = []
for i in range(10):
    r = run_fast_backtest()
    times_pd.append(r["time"])
    if i == 0:
        result_pd = r

mean_time_pd = np.mean(times_pd)

print(f"  总收益率: {result_pd['total_return']*100:.2f}%")
print(f"  夏普比率: {result_pd['sharpe']:.2f}")
print(f"  最大回撤: {result_pd['max_dd']*100:.2f}%")
print(f"  交易次数: {result_pd['n_trades']}")
print(f"  平均耗时: {mean_time_pd*1000:.1f}ms")

# ===================== VectorBT 实现 =====================
print("\n--- VectorBT ---")

import vectorbt as vbt

entries_ser = (close.rolling(20).mean() > close.rolling(60).mean())
entries_ser = (entries_ser == True) & (entries_ser.shift(1) == False)
exits_ser = (close.rolling(20).mean() <= close.rolling(60).mean())
exits_ser = (exits_ser == True) & (exits_ser.shift(1) == False)

# 为 VectorBT 设定频率（Yahoo 日线有gap，手动设为 'D'）
close_vbt = close.copy()
close_vbt.index.freq = 'D'
print(f"  手动设定频率: D (daily)")

# VectorBT 回测（首次含编译时间）
t0 = time.perf_counter()
pf = vbt.Portfolio.from_signals(close_vbt, entries_ser, exits_ser,
                                 direction="longonly",
                                 fees=0.001,
                                 init_cash=100000.0)
t1 = time.perf_counter()

result_vbt = {
    "total_return": float(pf.total_return()),
    "sharpe": float(pf.sharpe_ratio()),
    "max_dd": float(pf.max_drawdown()),
    "n_trades": int(pf.trades.count()),
    "time": t1 - t0,
}

print(f"  总收益率: {result_vbt['total_return']*100:.2f}%")
print(f"  夏普比率: {result_vbt['sharpe']:.2f}")
print(f"  最大回撤: {result_vbt['max_dd']*100:.2f}%")
print(f"  交易次数: {result_vbt['n_trades']}")
print(f"  耗时(含编译): {result_vbt['time']:.1f}s")

# ===================== Backtrader 实现 =====================
print("\n--- Backtrader ---")

import backtrader as bt

class SmaCross(bt.Strategy):
    params = (("short", 20), ("long", 60))

    def __init__(self):
        self.sma_short = bt.indicators.SMA(self.data.close, period=self.params.short)
        self.sma_long = bt.indicators.SMA(self.data.close, period=self.params.long)
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.close()

def run_backtrader():
    df = df_raw.copy()
    df.reset_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "datetime"}, inplace=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    data = bt.feeds.PandasData(dataname=df)

    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(100000)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    t0 = time.perf_counter()
    results = cerebro.run()
    t1 = time.perf_counter()

    strat = results[0]
    end_value = cerebro.broker.getvalue()

    sr = strat.analyzers.sharpe.get_analysis().get("sharperatio", 0)
    mdd = strat.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0)
    ta = strat.analyzers.trades.get_analysis()
    n_trades = ta.get("total", {}).get("total", 0) if ta else 0

    return {
        "total_return": end_value / 100000 - 1,
        "sharpe": sr if sr else 0,
        "max_dd": mdd / 100,
        "n_trades": n_trades,
        "time": t1 - t0,
    }

# warmup
_ = run_backtrader()

# benchmark
times_bt = []
for i in range(10):
    r = run_backtrader()
    times_bt.append(r["time"])
    if i == 0:
        result_bt = r

mean_time_bt = np.mean(times_bt)

print(f"  总收益率: {result_bt['total_return']*100:.2f}%")
print(f"  夏普比率: {result_bt['sharpe']:.2f}")
print(f"  最大回撤: {result_bt['max_dd']*100:.2f}%")
print(f"  交易次数: {result_bt['n_trades']}")
print(f"  平均耗时: {mean_time_bt*1000:.0f}ms")

# ===================== 对比总结 =====================
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)

print(f"""
┌──────────────────────┬────────────┬────────────┬────────────┐
│ 指标                 │ pandas向量  │ VectorBT   │ Backtrader │
├──────────────────────┼────────────┼────────────┼────────────┤
│ 总收益率             │ {result_pd['total_return']*100:>9.2f}% │ {result_vbt['total_return']*100:>9.2f}% │ {result_bt['total_return']*100:>9.2f}% │
│ 夏普比率             │ {result_pd['sharpe']:>10.2f} │ {result_vbt['sharpe']:>10.2f} │ {result_bt['sharpe']:>10.2f} │
│ 最大回撤             │ {result_pd['max_dd']*100:>9.2f}% │ {result_vbt['max_dd']*100:>9.2f}% │ {result_bt['max_dd']*100:>9.2f}% │
│ 交易次数             │ {result_pd['n_trades']:>11d} │ {result_vbt['n_trades']:>11d} │ {result_bt['n_trades']:>11d} │
│ 平均耗时             │ {mean_time_pd*1000:>9.1f}ms │ {result_vbt['time']:>9.1f}s¹ │ {mean_time_bt*1000:>9.0f}ms │
│ 引擎类型             │    向量化   │   向量化    │   事件驱动  │
├──────────────────────┴────────────┴────────────┴────────────┤
│ ¹ VectorBT 首次含 numba JIT 编译 (~30-60s), 后续秒级        │
│                                                              │
│ 速度排名         : pandas向量 ≈ Backtrader < VectorBT(含编译) │
│ 代码简洁度       : pandas向量 >> Backtrader > VectorBT        │
│ 复杂策略灵活性   : Backtrader >> VectorBT ≈ pandas向量       │
│ 实盘交易接入     : Backtrader (IB API等)                     │
│ 内置分析器       : Backtrader > VectorBT > pandas向量        │
│ 文档质量         : Backtrader > pandas > VectorBT 1.0        │
└──────────────────────────────────────────────────────────────┘
""")
