import pandas as pd

from quant.backtest.benchmarks import run_benchmark_suite
from quant.backtest.engine import run_long_only_backtest


def test_benchmark_suite_produces_comparable_metrics():
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame([
        {"date": date, "ticker": ticker, "close": 100 + i + (5 if ticker == "SPY" else 0)}
        for i, date in enumerate(dates) for ticker in ["AAA", "SPY"]
    ])
    signals = pd.DataFrame({"signal_date": dates[:4], "ticker": ["AAA"] * 4, "score": [1.0] * 4})
    strategy = run_long_only_backtest(signals, prices, top_n=1, hold_days=2)
    suite = run_benchmark_suite(prices, strategy, top_n=1, hold_days=2)
    assert {"strategy", "equal_weight", "random", "spy"}.issubset(suite["comparison"])
