"""Load PaperBroker SQLite state and build an auditable reconciliation report."""

import sqlite3
from pathlib import Path

import pandas as pd

from .reconcile import reconcile_trades


def read_paper_database(db_path):
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as conn:
        trades = pd.read_sql_query(
            "SELECT ticker, side, price, quantity, timestamp, order_status FROM trades WHERE order_status IS NULL OR order_status='FILLED'",
            conn,
        )
        positions = pd.read_sql_query("SELECT ticker, quantity, avg_cost FROM positions WHERE quantity > 0", conn)
        row = conn.execute("SELECT value FROM config WHERE key='cash'").fetchone()
    return {"trades": trades, "positions": positions, "cash": float(row[0]) if row else None}


def paper_equity_snapshot(paper_state, prices, as_of=None):
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    date = pd.Timestamp(as_of).normalize() if as_of is not None else px["date"].max()
    latest = px[px["date"] <= date].sort_values("date").groupby("ticker").tail(1).set_index("ticker")
    market_value = 0.0
    missing_prices = []
    for row in paper_state["positions"].itertuples(index=False):
        if row.ticker not in latest.index:
            missing_prices.append(row.ticker)
            continue
        market_value += float(row.quantity) * float(latest.loc[row.ticker, "close"])
    return {
        "as_of": date.date().isoformat(),
        "cash": paper_state["cash"],
        "positions_value": market_value,
        "equity": (paper_state["cash"] or 0.0) + market_value,
        "missing_prices": missing_prices,
    }


def reconcile_paper_database(db_path, backtest_trades, backtest_equity, prices, tolerance=1e-6):
    paper = read_paper_database(db_path)
    paper_trades = paper["trades"].rename(columns={"timestamp": "execution_date"})
    paper_trades["quantity"] = paper_trades["quantity"].abs()
    expected_trades = pd.DataFrame(backtest_trades).copy()
    if not expected_trades.empty:
        expected_trades["quantity"] = expected_trades["quantity"].abs()
    trade_result = reconcile_trades(expected_trades, paper_trades, tolerance=tolerance)
    snapshot = paper_equity_snapshot(paper, prices)
    expected_last = float(pd.DataFrame(backtest_equity)["equity"].iloc[-1])
    snapshot["backtest_ending_equity"] = expected_last
    snapshot["ending_equity_diff"] = snapshot["equity"] - expected_last
    snapshot["equity_passed"] = abs(snapshot["ending_equity_diff"]) <= tolerance
    return {"trades": trade_result, "equity": snapshot, "passed": bool(trade_result["passed"] and snapshot["equity_passed"])}
