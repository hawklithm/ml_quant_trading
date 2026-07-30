import pandas as pd

from quant.backtest.metrics import quantile_spread, rank_ic, turnover
from quant.data.news_archive import append_news_archive, news_as_of
from quant.data.universe import build_universe_snapshot, load_universe_as_of, save_universe_snapshot
from quant.models.calibration import calibrate_expected_returns


def test_universe_snapshot_is_point_in_time(tmp_path):
    first = build_universe_snapshot(["AAA", "BBB"], "US", "2024-01-01")
    second = build_universe_snapshot(["AAA", "CCC"], "US", "2024-02-01")
    save_universe_snapshot(pd.concat([first, second]), tmp_path)
    assert load_universe_as_of(tmp_path, "2024-01-15", "US") == ["AAA", "BBB"]
    assert load_universe_as_of(tmp_path, "2024-02-15", "US") == ["AAA", "CCC"]


def test_news_as_of_excludes_future_publications(tmp_path):
    append_news_archive(
        [
            {"news_id": "old", "ticker": "AAA", "published_at": "2024-01-01T10:00:00Z", "title": "old"},
            {"news_id": "new", "ticker": "AAA", "published_at": "2024-01-03T10:00:00Z", "title": "new"},
        ],
        tmp_path,
    )
    visible = news_as_of(tmp_path, ["AAA"], "2024-01-02T00:00:00Z")
    assert visible["news_id"].tolist() == ["old"]


def test_cross_sectional_metrics_and_calibration():
    scores = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    returns = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    assert rank_ic(scores, returns) > 0.99
    assert quantile_spread(scores, returns, quantiles=5)["spread"] > 0
    assert turnover(pd.Series({"AAA": 1.0}), pd.Series({"AAA": 0.5, "BBB": 0.5})) == 0.5
    calibrated = calibrate_expected_returns(scores, returns, buckets=5)
    assert calibrated["monotonic"] is True
