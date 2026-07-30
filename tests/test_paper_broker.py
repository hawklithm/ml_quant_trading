import live_pipeline
from quant.backtest.costs import TransactionCostModel


def test_paper_broker_applies_costs_and_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "paper.db"
    monkeypatch.setattr(live_pipeline, "DB_PATH", str(db_path))
    live_pipeline.init_db()
    broker = live_pipeline.PaperBroker(
        initial_cash=1_000.0,
        db_path=str(db_path),
        costs=TransactionCostModel(commission_bps=10, slippage_bps=0, sell_tax_bps=0),
    )
    filled = broker.execute("AAA", "BUY", 100.0, quantity=5, signal_id="s1", model_version="test")
    assert filled["status"] == "FILLED"
    assert filled["fee"] == 0.5
    assert broker.get_cash() == 499.5
    duplicate = broker.execute("AAA", "BUY", 100.0, quantity=5, signal_id="s1", model_version="test")
    assert duplicate["status"] == "DUPLICATE"
