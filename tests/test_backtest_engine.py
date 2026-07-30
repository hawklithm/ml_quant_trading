import pandas as pd
import pytest

from quant.backtest.costs import TransactionCostModel
from quant.backtest.engine import run_long_only_backtest


def test_backtest_uses_next_session_and_records_costs():
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    prices = pd.DataFrame(
        [(date, "AAA", 100.0 + i, 100.0 + i) for i, date in enumerate(dates)],
        columns=["date", "ticker", "open", "close"],
    )
    signals = pd.DataFrame(
        [{"signal_date": dates[0], "ticker": "AAA", "score": 1.0}]
    )
    result = run_long_only_backtest(
        signals,
        prices,
        initial_cash=10_000,
        top_n=1,
        costs=TransactionCostModel(commission_bps=10, slippage_bps=0),
    )
    assert result["trades"].iloc[0]["execution_date"] == dates[1]
    assert result["trades"].iloc[0]["cost"] > 0
    assert result["metrics"]["observations"] == len(dates)


def test_backtest_rejects_future_signal_columns():
    prices = pd.DataFrame(
        [
            {"date": "2024-01-01", "ticker": "AAA", "close": 100.0},
            {"date": "2024-01-02", "ticker": "AAA", "close": 101.0},
        ]
    )
    signals = pd.DataFrame(
        [{"signal_date": "2024-01-01", "ticker": "AAA", "score": 1.0, "future_return": 0.2}]
    )
    with pytest.raises(ValueError, match="future-looking"):
        run_long_only_backtest(signals, prices, top_n=1)
