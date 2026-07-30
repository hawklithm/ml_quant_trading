import json

import numpy as np
import pandas as pd

from ml_optimized_picker_v5 import (
    BENCHMARK_BY_MARKET,
    PURGE_GAP_DAYS,
    build_features_v5,
    make_purged_time_splits,
)


def test_config_declares_market_benchmarks_and_purge_gap():
    with open("v5_config.json", encoding="utf-8") as handle:
        config = json.load(handle)["ml_scoring"]
    assert config["benchmark_by_market"] == {"US": "spy", "HK": "hsi"}
    assert config["purge_gap_days"] >= max(config["prediction_horizons"])
    assert BENCHMARK_BY_MARKET["HK"] == "hsi"
    assert PURGE_GAP_DAYS >= 21


def test_purged_splits_leave_gap_before_test_data():
    splits = make_purged_time_splits(120, n_splits=3, gap=21)
    assert len(splits) == 3
    for train_idx, test_idx in splits:
        assert train_idx[-1] + 21 < test_idx[0]
        assert train_idx[-1] < test_idx[0]


def test_hk_target_uses_hsi_not_spy():
    dates = pd.date_range("2020-01-01", periods=140, freq="B")
    close = np.linspace(100, 180, len(dates))
    df = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(dates), 100000.0),
        },
        index=dates,
    )
    macro = {
        "spy": pd.Series(np.linspace(100, 250, len(dates)), index=dates),
        "hsi": pd.Series(np.linspace(100, 120, len(dates)), index=dates),
    }
    _, _, target_21d, _ = build_features_v5(
        df, macro_data=macro, ticker="0700.HK", market="HK", cross_section_rank=False
    )
    expected = (
        df["Close"].pct_change(21)
        - macro["hsi"].pct_change(21)
    ).shift(-21)
    common = target_21d.dropna().index.intersection(expected.dropna().index)
    pd.testing.assert_series_equal(
        target_21d.loc[common], expected.loc[common], check_names=False
    )
