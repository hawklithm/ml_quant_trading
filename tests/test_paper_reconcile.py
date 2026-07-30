import sqlite3

import pandas as pd

from quant.execution.paper_reconcile import reconcile_paper_database


def test_paper_database_reconciliation(tmp_path):
    db = tmp_path / "paper.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
        CREATE TABLE config (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE positions (ticker TEXT PRIMARY KEY, quantity INTEGER, avg_cost REAL);
        CREATE TABLE trades (ticker TEXT, side TEXT, price REAL, quantity REAL, timestamp TEXT, order_status TEXT);
        INSERT INTO config VALUES ('cash', '900.0');
        INSERT INTO positions VALUES ('AAA', 1, 100.0);
        INSERT INTO trades VALUES ('AAA', 'BUY', 100.0, 1, '2024-01-02T00:00:00', 'FILLED');
        """)
    prices = pd.DataFrame({"date": ["2024-01-02"], "ticker": ["AAA"], "close": [100.0]})
    trades = pd.DataFrame({"execution_date": ["2024-01-02"], "ticker": ["AAA"], "side": ["BUY"], "quantity": [1.0], "price": [100.0]})
    equity = pd.DataFrame({"equity": [1000.0]}, index=pd.to_datetime(["2024-01-02"]))
    result = reconcile_paper_database(db, trades, equity, prices)
    assert result["passed"]
