"""Run a reproducible point-in-time historical report from CSV or yfinance."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.backtest.costs import TransactionCostModel
from quant.backtest.engine import run_point_in_time_backtest
from quant.backtest.benchmarks import run_benchmark_suite
from quant.signals.v5_replay import make_v5_signal_generator


def load_prices(args) -> pd.DataFrame:
    if args.prices:
        frame = pd.read_csv(args.prices)
    else:
        if not args.download:
            raise ValueError("provide --prices or pass --download")
        import yfinance as yf
        raw = yf.download(args.tickers, start=args.start, end=args.end, auto_adjust=True, progress=False, group_by="ticker")
        if raw is None or raw.empty:
            raise RuntimeError("historical provider returned no rows; retry later or provide --prices CSV")
        frame = _from_yfinance(raw, args.tickers)
    required = {"date", "ticker", "close"}
    missing = required - set(frame.columns.str.lower())
    if missing:
        raise ValueError(f"prices input requires columns {sorted(required)}; missing {sorted(missing)}")
    frame.columns = [str(column).lower() for column in frame.columns]
    for column in ("open", "high", "low"):
        if column not in frame:
            frame[column] = frame["close"]
    if "volume" not in frame:
        frame["volume"] = 0
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    return frame[["date", "ticker", "open", "high", "low", "close", "volume"]].dropna(subset=["date", "ticker", "close"])


def _from_yfinance(raw, tickers):
    if isinstance(raw.columns, pd.MultiIndex):
        rows = []
        for ticker in tickers.split(","):
            part = raw[ticker].reset_index()
            part["ticker"] = ticker
            rows.append(part)
        raw = pd.concat(rows, ignore_index=True)
    else:
        raw = raw.reset_index()
        raw["ticker"] = tickers.split(",")[0]
    return raw.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", help="CSV with date,ticker,open,high,low,close,volume")
    parser.add_argument("--tickers", default="SPY,QQQ,MSFT,AAPL", help="comma-separated tickers for --download")
    parser.add_argument("--download", action="store_true", help="download data with yfinance")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--market", default="US", choices=["US", "HK"])
    parser.add_argument("--universe-root")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--signal-step", type=int, default=21, help="sample every N available sessions to limit model retraining")
    parser.add_argument("--output", default="historical_report")
    args = parser.parse_args()
    prices = load_prices(args)
    dates = prices.loc[(prices.date >= pd.Timestamp(args.start)) & (prices.date < pd.Timestamp(args.end)), "date"].drop_duplicates().sort_values()
    dates = dates.iloc[::args.signal_step]
    costs = TransactionCostModel(commission_bps=3.0, slippage_bps=5.0, sell_tax_bps=10.0)
    result = run_point_in_time_backtest(
        prices, make_v5_signal_generator(args.market), universe_root=args.universe_root,
        market=args.market, signal_dates=dates, top_n=args.top_n, hold_days=args.hold_days,
        costs=costs,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result["equity"].to_csv(output.with_suffix(".equity.csv"))
    result["trades"].to_csv(output.with_suffix(".trades.csv"), index=False)
    suite = run_benchmark_suite(prices, result, market=args.market, costs=costs, top_n=args.top_n, hold_days=args.hold_days)
    for name, benchmark in suite["results"].items():
        if name != "strategy":
            benchmark["equity"].to_csv(output.with_name(f"{output.name}.{name}.equity.csv"))
            benchmark["trades"].to_csv(output.with_name(f"{output.name}.{name}.trades.csv"), index=False)
    config_path = Path("v5_config.json")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": args.start, "end": args.end, "market": args.market,
        "rows": len(prices), "signal_dates": len(dates),
        "config_sha256": config_hash, "metrics": result["metrics"],
        "benchmarks": suite["comparison"], "cost_model": result["cost_model"],
    }
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
