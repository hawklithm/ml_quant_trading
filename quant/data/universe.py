from pathlib import Path

import pandas as pd


UNIVERSE_COLUMNS = [
    "snapshot_date",
    "market",
    "ticker",
    "status",
    "reason",
    "valid_from",
    "valid_to",
]


def build_universe_snapshot(
    tickers,
    market: str,
    snapshot_date,
    reason: str = "configured",
    status: str = "active",
) -> pd.DataFrame:
    """Create a normalized universe snapshot for one market and date."""
    date = pd.Timestamp(snapshot_date).normalize()
    market = str(market).upper()
    if market not in {"US", "HK"}:
        raise ValueError("market must be US or HK")
    unique = sorted({str(t).strip().upper() for t in tickers if str(t).strip()})
    if not unique:
        raise ValueError("tickers must contain at least one symbol")
    return pd.DataFrame(
        [
            {
                "snapshot_date": date,
                "market": market,
                "ticker": ticker,
                "status": status,
                "reason": reason,
                "valid_from": date,
                "valid_to": pd.NaT,
            }
            for ticker in unique
        ],
        columns=UNIVERSE_COLUMNS,
    )


def save_universe_snapshot(snapshot: pd.DataFrame, root) -> Path:
    """Append a snapshot to a CSV store, deduplicated by date/market/ticker."""
    missing = set(UNIVERSE_COLUMNS) - set(snapshot.columns)
    if missing:
        raise ValueError(f"snapshot missing columns: {sorted(missing)}")
    frame = snapshot[UNIVERSE_COLUMNS].copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"]).dt.normalize()
    frame["valid_from"] = pd.to_datetime(frame["valid_from"]).dt.normalize()
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    for (snapshot_date, market), current in frame.groupby(["snapshot_date", "market"]):
        current_tickers = set(current["ticker"])
        old_mask = (
            (frame["market"] == market)
            & (frame["snapshot_date"] < snapshot_date)
            & frame["valid_to"].isna()
            & ~frame["ticker"].isin(current_tickers)
        )
        frame.loc[old_mask, "valid_to"] = snapshot_date - pd.Timedelta(days=1)
    path = Path(root) / "universe_snapshots.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_csv(path, parse_dates=["snapshot_date", "valid_from", "valid_to"])
        # Close symbols omitted from a newer snapshot while preserving their
        # historical availability before the change date.
        for (snapshot_date, market), current in frame.groupby(["snapshot_date", "market"]):
            current_tickers = set(current["ticker"])
            old_mask = (
                (existing["market"] == market)
                & (existing["snapshot_date"] < snapshot_date)
                & existing["valid_to"].isna()
                & ~existing["ticker"].isin(current_tickers)
            )
            existing.loc[old_mask, "valid_to"] = snapshot_date - pd.Timedelta(days=1)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame = frame.drop_duplicates(["snapshot_date", "market", "ticker"], keep="last")
    frame.sort_values(["snapshot_date", "market", "ticker"]).to_csv(path, index=False)
    return path


def load_universe_as_of(root, as_of, market: str) -> list[str]:
    """Return symbols active on the requested historical date."""
    path = Path(root) / "universe_snapshots.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    date = pd.Timestamp(as_of).normalize()
    market = str(market).upper()
    frame = pd.read_csv(path, parse_dates=["snapshot_date", "valid_from", "valid_to"])
    frame = frame[(frame["market"] == market) & (frame["status"] == "active")]
    frame = frame[frame["snapshot_date"] <= date]
    frame = frame[(frame["valid_from"].isna()) | (frame["valid_from"] <= date)]
    frame = frame[(frame["valid_to"].isna()) | (frame["valid_to"] >= date)]
    if frame.empty:
        return []
    latest = frame.sort_values("snapshot_date").groupby("ticker", as_index=False).tail(1)
    return sorted(latest["ticker"].tolist())
