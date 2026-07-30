import pandas as pd
import pytest

from quant.data.validation import validate_price_frame, validate_signal_frame


def test_validation_rejects_duplicate_prices_and_nonfinite_scores():
    prices = pd.DataFrame({"date": ["2024-01-01", "2024-01-01"], "ticker": ["AAA", "AAA"], "close": [1, 1]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_price_frame(prices)
    signals = pd.DataFrame({"signal_date": ["2024-01-01"], "ticker": ["AAA"], "score": [float("nan")]})
    with pytest.raises(ValueError, match="finite"):
        validate_signal_frame(signals)
