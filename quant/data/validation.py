"""Fail-fast validation for market and signal inputs."""

import numpy as np
import pandas as pd


def validate_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices requires {sorted(required)}; missing {sorted(missing)}")
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    if frame["ticker"].eq("").any() or frame["date"].isna().any():
        raise ValueError("prices contains empty tickers or invalid dates")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("prices contains duplicate date/ticker rows")
    numeric = [column for column in ("open", "high", "low", "close", "volume") if column in frame]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"prices contains non-finite values in {column}")
    if (frame["close"] <= 0).any():
        raise ValueError("prices close values must be positive")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def validate_signal_frame(signals: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_date", "ticker", "score"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"signals requires {sorted(required)}; missing {sorted(missing)}")
    frame = signals.copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str).str.strip()
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    if frame["ticker"].eq("").any() or frame["signal_date"].isna().any():
        raise ValueError("signals contains empty tickers or invalid dates")
    if not np.isfinite(frame["score"].to_numpy(dtype=float)).all():
        raise ValueError("signals score values must be finite")
    return frame.sort_values(["signal_date", "score"], ascending=[True, False]).reset_index(drop=True)
