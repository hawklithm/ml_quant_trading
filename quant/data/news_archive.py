from pathlib import Path

import pandas as pd


NEWS_COLUMNS = [
    "news_id",
    "ticker",
    "published_at",
    "fetched_at",
    "source",
    "title",
    "url",
    "body",
]


def normalize_news(records) -> pd.DataFrame:
    """Normalize provider records and reject records without publication time."""
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)
    missing = {"ticker", "published_at", "title"} - set(frame.columns)
    if missing:
        raise ValueError(f"news records missing columns: {sorted(missing)}")
    for column in NEWS_COLUMNS:
        if column not in frame:
            frame[column] = ""
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], utc=True, errors="coerce")
    if frame["published_at"].isna().any():
        raise ValueError("every news record requires a valid published_at")
    frame["news_id"] = frame["news_id"].where(frame["news_id"].astype(str).str.len() > 0, frame["url"])
    frame["news_id"] = frame["news_id"].where(frame["news_id"].astype(str).str.len() > 0, frame["title"])
    return frame[NEWS_COLUMNS].drop_duplicates("news_id", keep="last")


def append_news_archive(records, root) -> Path:
    """Append normalized news to a CSV archive with stable deduplication."""
    frame = normalize_news(records)
    path = Path(root) / "news_archive.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["published_at", "fetched_at"])
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.drop_duplicates("news_id", keep="last")
    frame.sort_values(["published_at", "ticker", "news_id"]).to_csv(path, index=False)
    return path


def news_as_of(root, tickers, signal_time) -> pd.DataFrame:
    """Return only news published no later than signal_time."""
    path = Path(root) / "news_archive.csv"
    if not path.exists():
        return pd.DataFrame(columns=NEWS_COLUMNS)
    frame = pd.read_csv(path, parse_dates=["published_at", "fetched_at"])
    cutoff = pd.Timestamp(signal_time)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    selected = frame[
        frame["ticker"].isin(list(tickers))
        & (pd.to_datetime(frame["published_at"], utc=True) <= cutoff)
    ]
    return selected[NEWS_COLUMNS].sort_values(["ticker", "published_at", "news_id"])
