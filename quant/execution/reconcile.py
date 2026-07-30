"""Compare paper execution with a deterministic backtest."""

import pandas as pd


def reconcile_equity(backtest_equity, paper_equity, tolerance: float = 1e-6) -> dict:
    """Compare two equity series after normalizing their date indexes."""
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    left = _equity_series(backtest_equity, "backtest")
    right = _equity_series(paper_equity, "paper")
    joined = pd.concat([left, right], axis=1, join="inner").dropna()
    if joined.empty:
        return {"passed": False, "observations": 0, "reason": "no overlapping equity dates"}
    difference = (joined.iloc[:, 0] - joined.iloc[:, 1]).abs()
    return {
        "passed": bool(float(difference.max()) <= tolerance),
        "observations": int(len(joined)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "tolerance": float(tolerance),
        "first_date": joined.index.min().date().isoformat(),
        "last_date": joined.index.max().date().isoformat(),
    }


def reconcile_trades(backtest_trades, paper_trades, tolerance: float = 1e-6) -> dict:
    """Compare the execution fields that affect portfolio state."""
    required = {"execution_date", "ticker", "side", "quantity", "price"}
    left = _trade_frame(backtest_trades, required)
    right = _trade_frame(paper_trades, required)
    if len(left) != len(right):
        return {"passed": False, "backtest_count": len(left), "paper_count": len(right), "reason": "trade count differs"}
    if left.empty:
        return {"passed": True, "backtest_count": 0, "paper_count": 0}
    left = left.sort_values(list(required)).reset_index(drop=True)
    right = right.sort_values(list(required)).reset_index(drop=True)
    exact = (left[["execution_date", "ticker", "side"]] == right[["execution_date", "ticker", "side"]]).all().all()
    quantity_diff = (left["quantity"] - right["quantity"]).abs().max()
    price_diff = (left["price"] - right["price"]).abs().max()
    return {
        "passed": bool(exact and quantity_diff <= tolerance and price_diff <= tolerance),
        "backtest_count": len(left),
        "paper_count": len(right),
        "max_quantity_diff": float(quantity_diff),
        "max_price_diff": float(price_diff),
        "tolerance": float(tolerance),
    }


def _equity_series(value, label: str) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if "equity" not in value.columns:
            raise ValueError(f"{label} equity DataFrame requires an equity column")
        series = value["equity"]
    else:
        series = pd.Series(value)
    series = series.copy()
    series.index = pd.to_datetime(series.index).normalize()
    return pd.to_numeric(series, errors="raise").rename(label)


def _trade_frame(value, required: set[str]) -> pd.DataFrame:
    frame = pd.DataFrame(value).copy()
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"trade frame requires {sorted(required)}; missing {sorted(missing)}")
    frame["execution_date"] = pd.to_datetime(frame["execution_date"]).dt.normalize()
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="raise")
    frame["price"] = pd.to_numeric(frame["price"], errors="raise")
    return frame
