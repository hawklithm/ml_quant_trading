import pandas as pd

from quant.execution.reconcile import reconcile_equity, reconcile_trades


def test_reconcile_matching_equity_and_trades():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    equity = pd.Series([100.0, 101.0], index=dates)
    assert reconcile_equity(equity, equity)["passed"]
    trades = pd.DataFrame([{
        "execution_date": "2024-01-02", "ticker": "AAA", "side": "BUY",
        "quantity": 1.0, "price": 100.0,
    }])
    assert reconcile_trades(trades, trades)["passed"]


def test_reconcile_detects_drift():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    result = reconcile_equity(pd.Series([100.0, 101.0], index=dates), pd.Series([100.0, 102.0], index=dates))
    assert not result["passed"]
    trades = pd.DataFrame([{
        "execution_date": "2024-01-02", "ticker": "AAA", "side": "BUY",
        "quantity": 1.0, "price": 100.0,
    }])
    changed = trades.assign(price=101.0)
    assert not reconcile_trades(trades, changed)["passed"]
