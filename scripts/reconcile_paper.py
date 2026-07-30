"""Reconcile a PaperBroker SQLite database against backtest CSV outputs."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.execution.paper_reconcile import reconcile_paper_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="PaperBroker SQLite path")
    parser.add_argument("--backtest-equity", required=True, help="backtest equity CSV with date,equity")
    parser.add_argument("--backtest-trades", required=True, help="backtest trades CSV")
    parser.add_argument("--prices", required=True, help="prices CSV with date,ticker,close")
    parser.add_argument("--output", default="paper_reconciliation.json")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    equity = pd.read_csv(args.backtest_equity, index_col=0, parse_dates=True)
    trades = pd.read_csv(args.backtest_trades)
    prices = pd.read_csv(args.prices)
    result = reconcile_paper_database(args.db, trades, equity, prices, args.tolerance)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
