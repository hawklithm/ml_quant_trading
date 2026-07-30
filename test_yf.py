import warnings

import pytest


def test_yfinance_smoke():
    """Optional live-data smoke test; never block offline unit tests."""
    yf = pytest.importorskip("yfinance")
    warnings.filterwarnings("ignore")
    try:
        df = yf.Ticker("AAPL").history(period="3mo")
    except Exception as exc:
        pytest.skip(f"live Yahoo Finance unavailable: {exc}")
    if df.empty:
        pytest.skip("Yahoo Finance returned no data")
    assert "Close" in df.columns
