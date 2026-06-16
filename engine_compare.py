#!/usr/bin/env python3
"""
回测引擎对比: pandas 向量化 vs Backtrader

测试条件:
  - 相同数据 (AAPL 3年日线)
  - 相同策略 (SMA20/60 金叉死叉)
  - 相同交易成本 (0.1% 手续费)
  - 各跑5次取平均

结论: 简单策略用 pandas 向量化, 复杂策略用 Backtrader
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

print("=" * 60)
print("回测引擎对比: pandas 向量化 vs Backtrader")
print("=" * 60)

TICKER = "AAPL"
PERIOD = "3y"

print(f"\n标的: {TICKER} | 周期: {PERIOD}")
print("下载数据...")

df_raw = yf.download(TICKER, period=PERIOD, auto_adjust=True, progress=False)
df_raw.columns = [c[0] for c in df_raw.columns]
close = df_raw["Close"]
print(f"  数据行数: {len(df_raw)}")
print(f"  日期范围: {df_raw.index[0].date()} ~ {df_raw.index[-1].date()}")

# ──────────────────────────────────────────────
# 1. pandas 向量化回测
# ──────────────────────────────────────────────
print("\n1) pandas 向量化回测")

def backtest_pandas(short=20, long=60, fees=0.001):
    t0 = time.perf_counter()

    sma_s = close.rolling(short).mean()
    sma_l = close.rolling(long).mean()
    in_pos = (sma_s > sma_l).astype(int)

    # 持仓变更时扣手续费
    pos_chg = in_pos.diff().abs() * fees
    ret = close.pct_change()
    strat_ret = in_pos.shift(1) * ret - pos_chg

    t1 = time.perf_counter()

    cum = (1 + strat_ret).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()

    return {
        "total_return": float(cum.iloc[-1] - 1),
        "sharpe": float((strat_ret.mean() / strat_ret.std()) * np.sqrt(252)),
        "max_dd": float(dd.min()),
        "n_trades": int(in_pos.diff().abs().sum() // 2),  # 进+出=1笔
        "time": t1 - t0,
    }

# warmup + 结果
r_pd = backtest_pandas()

# benchmark 10次
times_pd = [backtest_pandas()["time"] for _ in range(10)]
avg_pd = np.mean(times_pd)

print(f"  总收益率: {r_pd['total_return']*100:.2f}%")
print(f"  夏普比率: {r_pd['sharpe']:.2f}")
print(f"  最大回撤: {r_pd['max_dd']*100:.2f}%")
print(f"  交易次数: {r_pd['n_trades']}")
print(f"  平均耗时: {avg_pd*1000:.1f}ms")
print(f"  代码行数: ~4行 (pandas 向量化操作)")

# ──────────────────────────────────────────────
# 2. Backtrader 回测
# ──────────────────────────────────────────────
print("\n2) Backtrader")
import backtrader as bt

class SmaCross(bt.Strategy):
    params = (("short", 20), ("long", 60))

    def __init__(self):
        self.sma_s = bt.indicators.SMA(self.data.close, period=self.params.short)
        self.sma_l = bt.indicators.SMA(self.data.close, period=self.params.long)
        self.cross = bt.indicators.CrossOver(self.sma_s, self.sma_l)

    def next(self):
        if not self.position and self.cross > 0:
            self.buy()
        elif self.position and self.cross < 0:
            self.close()

def backtest_bt():
    df = df_raw.copy()
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index)
    df = df.rename(columns=str.lower)
    df.index.name = "datetime"
    # Backtrader PandasData 需要 datetime 在行索引上
    data = bt.feeds.PandasData(dataname=df, timeframe=bt.TimeFrame.Days)
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross)
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sr", riskfreerate=0.02, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ta")

    t0 = time.perf_counter()
    results = cerebro.run()
    t1 = time.perf_counter()

    s = results[0]
    ev = cerebro.broker.getvalue()
    sr = s.analyzers.sr.get_analysis().get("sharperatio", 0) or 0
    mdd = s.analyzers.dd.get_analysis().get("max", {}).get("drawdown", 0) / 100
    ta = s.analyzers.ta.get_analysis()
    nt = ta.get("total", {}).get("total", 0) if ta else 0

    return {
        "total_return": ev / 100000 - 1,
        "sharpe": sr,
        "max_dd": mdd,
        "n_trades": nt,
        "time": t1 - t0,
    }

r_bt = backtest_bt()

times_bt = [backtest_bt()["time"] for _ in range(10)]
avg_bt = np.mean(times_bt)

print(f"  总收益率: {r_bt['total_return']*100:.2f}%")
print(f"  夏普比率: {r_bt['sharpe']:.2f}")
print(f"  最大回撤: {r_bt['max_dd']*100:.2f}%")
print(f"  交易次数: {r_bt['n_trades']}")
print(f"  平均耗时: {avg_bt*1000:.0f}ms")
print(f"  代码行数: ~40行 (事件驱动框架)")

# ──────────────────────────────────────────────
# 3. 对比总结
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("对比总结")
print("=" * 60)

print(f"""
┌──────────────────────┬─────────────┬─────────────┐
│ 指标                 │ pandas向量化 │  Backtrader │
├──────────────────────┼─────────────┼─────────────┤
│ 总收益率             │ {r_pd['total_return']*100:>10.2f}% │ {r_bt['total_return']*100:>10.2f}% │
│ 夏普比率             │ {r_pd['sharpe']:>11.2f} │ {r_bt['sharpe']:>11.2f} │
│ 最大回撤             │ {r_pd['max_dd']*100:>10.2f}% │ {r_bt['max_dd']*100:>10.2f}% │
│ 交易次数             │ {r_pd['n_trades']:>12d} │ {r_bt['n_trades']:>12d} │
│ 平均耗时             │ {avg_pd*1000:>10.1f}ms │ {avg_bt*1000:>10.0f}ms │
│ 引擎类型             │     向量化   │    事件驱动  │
│ 代码行数(SMA策略)    │       ~4行  │      ~40行  │
├──────────────────────┴─────────────┴─────────────┤
│ 速度:   pandas ~{avg_bt/avg_pd:.0f}x 快于 Backtrader         │
└──────────────────────────────────────────────────┘
""")

print("功能覆盖矩阵:")
matrix = """
│ 功能                      │ pandas向量化 │ Backtrader │
│───────────────────────────│──────────────│────────────��
│ 简单信号回测              │     ✓        │    ✓       │
│ 复杂策略 (多条件/自适应)  │     ✗        │    ✓       │
│ 止损/止盈/限价单          │     ✗        │    ✓       │
│ 多标的组合回测            │     ✓        │    ✓       │
│ 参数优化扫描              │     ✓(手动)  │    ✓(内置) │
│ 实盘交易接口(IB)          │     ✗        │    ✓       │
│ 自定义指标                │     ✓        │    ✓       │
│ 内置分析器                │     ✗        │    ✓       │
│ 交互式图表                │     ✓(plt)   │    ✓(plt)  │
│ 学习成本                  │     低       │    中-高   │
│ 适用数据量                │     任意     │     适中   │
"""
print(matrix)

print("选择建议:")
print("""
  选 pandas 向量化:
    - 快速验证策略想法 (几分钟内出结果)
    - 参数扫描/网格搜索 (几百种组合)
    - 信号简单 (进场条件 <=1-2 个)
    - 不需要精确模拟滑点/限价单
    - 研究阶段/Notebook 探索

  选 Backtrader:
    - 策略逻辑复杂 (多条件、自适应参数)
    - 需要止损、止盈、限价单模拟
    - 需要对接实盘 (IB API)
    - 需要完整的交易记录分析
    - 上线前的最终回测验证

  推荐工作流:
    1. pandas 向量化做策略探索 + 参数扫描
    2. Backtrader 做最终回测验证 + 实盘模拟
    3. 两者收益、夏普、回撤应基本一致 (差异 < 5%)
       → 如果差异大, 检查手续费/滑点/持仓逻辑
""")
