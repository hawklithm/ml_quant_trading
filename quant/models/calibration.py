import pandas as pd


def score_bucket_report(scores, realized_returns, buckets: int = 5) -> pd.DataFrame:
    """Create an OOS-only score bucket report for calibration review."""
    if buckets < 2:
        raise ValueError("buckets must be at least 2")
    frame = pd.DataFrame({"score": scores, "realized_return": realized_returns}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bucket", "count", "mean_return", "median_return"])
    frame["bucket"] = pd.qcut(
        frame["score"].rank(method="first"), buckets, labels=False, duplicates="drop"
    ) + 1
    return (
        frame.groupby("bucket", as_index=False)
        .agg(
            count=("realized_return", "size"),
            mean_return=("realized_return", "mean"),
            median_return=("realized_return", "median"),
        )
        .sort_values("bucket")
    )


def calibrate_expected_returns(scores, realized_returns, buckets: int = 5) -> dict:
    """Map scores to empirical OOS returns; reject non-monotonic mappings."""
    report = score_bucket_report(scores, realized_returns, buckets=buckets)
    if report.empty:
        raise ValueError("at least one valid score/return observation is required")
    means = report["mean_return"].tolist()
    monotonic = all(left <= right for left, right in zip(means, means[1:]))
    return {
        "report": report,
        "monotonic": monotonic,
        "min_score": float(pd.Series(scores).dropna().min()),
        "max_score": float(pd.Series(scores).dropna().max()),
    }


def map_scores_to_expected_returns(current_scores, calibration_scores, calibration_returns, buckets: int = 5):
    """Map current scores to empirical OOS bucket returns.

    Bucket boundaries are fitted only from historical calibration scores.
    """
    history = pd.DataFrame({"score": calibration_scores, "return": calibration_returns}).dropna()
    if len(history) < buckets:
        raise ValueError("not enough OOS observations for calibration")
    boundaries = history["score"].quantile([i / buckets for i in range(buckets + 1)]).to_numpy()
    boundaries[0] = float("-inf")
    boundaries[-1] = float("inf")
    history["bucket"] = pd.cut(history["score"], bins=boundaries, labels=False, include_lowest=True)
    means = history.groupby("bucket")["return"].mean()
    current = pd.Series(current_scores, dtype=float)
    current_bucket = pd.cut(current, bins=boundaries, labels=False, include_lowest=True)
    mapped = current_bucket.map(means).fillna(history["return"].mean())
    return mapped.to_numpy(dtype=float)
