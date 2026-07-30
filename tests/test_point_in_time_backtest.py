import pandas as pd

from quant.backtest.engine import run_point_in_time_backtest
from quant.data.universe import build_universe_snapshot, save_universe_snapshot


def test_point_in_time_generator_receives_no_future_prices(tmp_path):
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    prices = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "open": 100.0 + i, "close": 100.0 + i}
            for i, date in enumerate(dates)
            for ticker in ("AAA", "BBB")
        ]
    )
    save_universe_snapshot(
        build_universe_snapshot(["AAA", "BBB"], "US", dates[0]), tmp_path
    )

    seen = []

    def generator(as_of, tickers, history):
        assert history["date"].max() <= as_of
        seen.append((as_of, history["date"].max()))
        return pd.DataFrame(
            [{"signal_date": as_of, "ticker": tickers[0], "score": 1.0}]
        )

    result = run_point_in_time_backtest(
        prices,
        generator,
        universe_root=tmp_path,
        market="US",
        signal_dates=dates[:3],
        top_n=1,
    )
    assert len(seen) == 3
    assert result["signals"]["signal_date"].max() <= dates[2]


def test_point_in_time_rejects_backdated_and_unknown_signals():
    dates = pd.date_range("2024-01-01", periods=4, freq="B")
    prices = pd.DataFrame([
        {"date": date, "ticker": "AAA", "close": 100.0 + i}
        for i, date in enumerate(dates)
    ])

    def backdated(as_of, tickers, history):
        return pd.DataFrame([{"signal_date": as_of - pd.Timedelta(days=1), "ticker": "AAA", "score": 1.0}])

    import pytest
    with pytest.raises(ValueError, match="exactly as_of"):
        run_point_in_time_backtest(prices, backdated, signal_dates=[dates[1]], top_n=1)

    def unknown(as_of, tickers, history):
        return pd.DataFrame([{"signal_date": as_of, "ticker": "FUTURE", "score": 1.0}])

    with pytest.raises(ValueError, match="outside"):
        run_point_in_time_backtest(prices, unknown, signal_dates=[dates[1]], top_n=1)
