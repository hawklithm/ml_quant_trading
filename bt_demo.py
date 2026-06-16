#!/usr/bin/env python3
"""
Backtrader 框架演示 — 双均线策略完整版
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class SmaCross(bt.Strategy):
    """双均线交叉策略"""
    params = (("short", 20), ("long", 60))

    def __init__(self):
        self.sma_short = bt.indicators.SMA(self.data.close, period=self.params.short)
        self.sma_long = bt.indicators.SMA(self.data.close, period=self.params.long)
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        if not self.position:  # 未持仓
            if self.crossover > 0:  # 金叉 -> 买入
                self.buy()
        elif self.crossover < 0:  # 死叉 -> 卖出
            self.close()


def run_backtest(ticker="AAPL", period="2y", short=20, long=60, cash=100000):
    # 拉数据
    print(f"获取 {ticker} 数据...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    df.columns = [c[0] for c in df.columns]

    df.reset_index(inplace=True)
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"date": "datetime"}, inplace=True)
    df = df[["datetime", "open", "high", "low", "close", "volume"]]

    # Backtrader 数据源
    data = bt.feeds.PandasData(dataname=df)

    # 引擎
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(SmaCross, short=short, long=long)
    cerebro.broker.setcash(cash)
    cerebro.broker.setcommission(commission=0.001)  # 千分之一手续费

    # 分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.02, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

    print(f"初始资金: ${cash:,.2f}")
    results = cerebro.run()
    strat = results[0]

    # 结果
    end_value = cerebro.broker.getvalue()

    sharpe = strat.analyzers.sharpe.get_analysis()
    dd = strat.analyzers.dd.get_analysis()
    ret = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    print(f"最终资金: ${end_value:,.2f}")
    print(f"总收益率: {(end_value/cash-1)*100:.2f}%")
    print(f"年化收益率: {ret.get('rnorm100', 0):.2f}%")
    print(f"夏普比率: {sharpe.get('sharperatio', 0):.2f}")
    print(f"最大回撤: {dd.get('max', {}).get('drawdown', 0):.2f}%")

    if trades:
        total_trades = trades.get("total", {}).get("total", 0)
        won = trades.get("won", {}).get("total", 0)
        lost = trades.get("lost", {}).get("total", 0)
        print(f"\n交易统计:")
        print(f"  总交易次数: {total_trades}")
        print(f"  盈利: {won} / 亏损: {lost}")
        if total_trades:
            print(f"  胜率: {won/total_trades*100:.1f}%")

    print(f"\n策略参数: SMA{short}/{long}")
    print(f"回测区间: {period}")
    print(f"交易成本: 0.1% 手续费")

    # 画图
    fig = cerebro.plot(style="candlestick", volume=True, figsize=(14, 9))
    print("\n图表已生成 (Backtrader 内嵌绘图)")

    return end_value


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    run_backtest(ticker)
