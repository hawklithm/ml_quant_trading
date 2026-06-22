#!/usr/bin/env python3
"""Seed crypto cache from CoinGecko (free, no auth needed)"""
import json, os, time, urllib.request, urllib.error
import pandas as pd
from datetime import datetime, timedelta

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant/crypto")
os.makedirs(CACHE_DIR, exist_ok=True)

# CoinGecko API: market_chart/range returns {prices: [[ts, price], ...]}
# Free rate limit: 10-30 calls/min, no API key needed
COINS = {
    "BTC-USD": "bitcoin",
    "ETH-USD": "ethereum",
    "SOL-USD": "solana",
}

now_ts = int(datetime.now().timestamp())
start_ts = now_ts - 200 * 86400  # ~6 months back

def fetch_coin_history(coin_id):
    """Fetch daily OHLCV from CoinGecko"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?vs_currency=usd&from={start_ts}&to={now_ts}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        data = json.loads(resp.read().decode())
        # prices: [[ts, price], ...]
        prices = data.get("prices", [])
        if not prices:
            return None

        # Build DataFrame with daily close
        records = []
        for ts, price in prices:
            d = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            records.append({"Date": d, "Close": price})

        # Deduplicate by date (keep last entry per day)
        seen = {}
        for r in records:
            seen[r["Date"]] = r["Close"]
        dates = sorted(seen.keys())
        closes = [seen[d] for d in dates]

        df = pd.DataFrame({"Close": closes}, index=pd.to_datetime(dates))
        # Also get high/low/volume if possible
        return df
    except urllib.error.HTTPError as e:
        print(f"  HTTP {e.code}: {e.reason}")
        return None
    except Exception as e:
        print(f"  error: {e}")
        return None

for ticker, coin_id in COINS.items():
    safe = ticker.replace("-", "_")
    path = os.path.join(CACHE_DIR, f"{safe}_6mo.pkl")
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"{ticker}: already cached ({os.path.getsize(path)} bytes)")
        continue

    print(f"{ticker} ({coin_id}): fetching from CoinGecko...")
    df = fetch_coin_history(coin_id)
    if df is not None and len(df) > 10:
        df.to_pickle(path)
        print(f"  OK: {len(df)} rows, ${float(df['Close'].iloc[-1]):,.0f}")
    else:
        print(f"  FAILED: no data")
    time.sleep(3)  # Rate limit: 10-30 calls/min

print("\n--- Cache contents ---")
import glob
for f in sorted(glob.glob(os.path.join(CACHE_DIR, "*"))):
    size = os.path.getsize(f)
    if size > 50:
        try:
            df = pd.read_pickle(f)
            print(f"  {os.path.basename(f)}: {len(df)} rows, latest=${float(df['Close'].iloc[-1]):,.0f} ({size} bytes)")
        except:
            print(f"  {os.path.basename(f)}: {size} bytes (unreadable)")
