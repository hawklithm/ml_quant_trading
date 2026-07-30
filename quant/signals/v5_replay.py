import pandas as pd


def make_v5_signal_generator(market="US", macro_data=None):
    """Build a point-in-time callback around the v5 scorer.

    The callback accepts only the truncated history supplied by the replay
    engine and passes it through score_stock_v5(data=...).
    """
    from ml_optimized_picker_v5 import score_stock_v5

    def generate(as_of, tickers, history):
        rows = []
        for ticker in tickers:
            frame = history[history["ticker"] == ticker].copy()
            if frame.empty:
                continue
            rename = {
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
            frame = frame.rename(columns=rename)
            frame = frame.set_index("Date").sort_index()
            required = {"Open", "High", "Low", "Close", "Volume"}
            if not required.issubset(frame.columns):
                continue
            result = score_stock_v5(
                ticker,
                macro_data=macro_data,
                market=market,
                data=frame[["Open", "High", "Low", "Close", "Volume"]],
            )
            if result is not None:
                rows.append({
                    "signal_date": as_of,
                    "ticker": ticker,
                    "score": result["score"],
                    "direction": result["direction"],
                    "data_asof": result.get("data_asof"),
                    "benchmark": result.get("benchmark"),
                })
        return pd.DataFrame(rows)

    return generate
