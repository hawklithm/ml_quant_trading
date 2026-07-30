"""Deterministic benchmark portfolios for backtest reports."""

import numpy as np
import pandas as pd

from .engine import run_long_only_backtest


def make_benchmark_signals(prices: pd.DataFrame, tickers, mode: str = "equal_weight", seed: int = 7) -> pd.DataFrame:
    """Create point-in-time signals for equal-weight or seeded random ranking."""
    if mode not in {"equal_weight", "random"}:
        raise ValueError("mode must be equal_weight or random")
    dates = sorted(pd.to_datetime(prices["date"]).dt.normalize().unique())
    symbols = sorted(set(tickers))
    if not symbols:
        raise ValueError("tickers must not be empty")
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        values = rng.random(len(symbols)) if mode == "random" else np.ones(len(symbols))
        rows.extend({"signal_date": date, "ticker": ticker, "score": float(score)} for ticker, score in zip(symbols, values))
    return pd.DataFrame(rows)


def make_buy_hold_signals(prices: pd.DataFrame, ticker: str) -> pd.DataFrame:
    dates = sorted(pd.to_datetime(prices.loc[prices["ticker"] == ticker, "date"]).dt.normalize().unique())
    if not dates:
        raise ValueError(f"benchmark ticker not found: {ticker}")
    return pd.DataFrame({"signal_date": [dates[0]], "ticker": [ticker], "score": [1.0]})


def run_benchmark_suite(prices, strategy_result, market="US", costs=None, top_n=10, hold_days=5, seed=7):
    """Run buy-and-hold, equal-weight, random and market benchmarks available in prices."""
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    symbols = sorted(px["ticker"].dropna().unique())
    results = {}
    for name, signals, universe_size in [
        ("equal_weight", make_benchmark_signals(px, symbols, "equal_weight", seed), top_n),
        ("random", make_benchmark_signals(px, symbols, "random", seed), top_n),
    ]:
        results[name] = run_long_only_backtest(signals, px, top_n=min(universe_size, len(symbols)), hold_days=hold_days, costs=costs)
    benchmark_ticker = "SPY" if market.upper() == "US" else "HSI"
    aliases = {benchmark_ticker, "^HSI"} if benchmark_ticker == "HSI" else {benchmark_ticker}
    available = next((ticker for ticker in aliases if ticker in symbols), None)
    if available:
        results[benchmark_ticker.lower()] = run_long_only_backtest(
            make_buy_hold_signals(px, available), px, top_n=1, hold_days=10**9, costs=costs
        )
    strategy_end = float(strategy_result["equity"]["equity"].iloc[-1])
    strategy_start = float(strategy_result["equity"]["equity"].iloc[0])
    results["strategy"] = strategy_result
    comparison = {}
    for name, result in results.items():
        start = float(result["equity"]["equity"].iloc[0])
        end = float(result["equity"]["equity"].iloc[-1])
        comparison[name] = {
            **result["metrics"],
            "ending_equity": end,
            "total_return": end / start - 1 if start else None,
            "excess_return_vs_strategy": (end / start - 1) - (strategy_end / strategy_start - 1) if start and strategy_start else None,
        }
    return {"results": results, "comparison": comparison}
