import json
from pathlib import Path

import cron_market_job


def test_reset_bias_matches_config_defaults():
    config = json.loads(Path("v5_config.json").read_text(encoding="utf-8"))
    expected = config["ml_scoring"]["direction_thresholds"]
    assert cron_market_job.TUNE_PRESETS["reset_bias"]["steps"] == [{
        "bullish": expected["bullish"],
        "bearish": expected["bearish"],
    }]


def test_benchmark_and_sentiment_helpers_are_market_aware_and_idempotent():
    assert cron_market_job._benchmark_ticker("US") == "SPY"
    assert cron_market_job._benchmark_ticker("HK") == "^HSI"
    predictions = [{"ticker": "AAA", "sentiment_applied": True}, {"ticker": "BBB"}]
    assert [p["ticker"] for p in cron_market_job._pending_sentiment_predictions(predictions)] == ["BBB"]
