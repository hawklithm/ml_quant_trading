#!/usr/bin/env python3
"""
Crypto ML Daily Prediction
Same v5 framework as stock system:
- 7-dimension scoring (momentum, volatility, RSI, trend, volume, drawdown, vol trend)
- Fear & Greed index
- Fibonacci support/resistance
- Cross-currency correlation
"""
import yfinance as yf
import numpy as np
import pandas as pd
import json, os, sys, time, pickle
from datetime import datetime, timedelta

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
STATE_DIR = os.path.join(CACHE_DIR, "crypto_jobs")

now = datetime.now()
today = now.strftime("%Y-%m-%d")

# ─── Load / init state ───
os.makedirs(STATE_DIR, exist_ok=True)
state_file = os.path.join(STATE_DIR, "crypto_state.json")
state = {}
if os.path.exists(state_file):
    with open(state_file) as f:
        state = json.load(f)

if state.get("last_pred_date") == today:
    print(f"今天 {today} 已运行过加密预测，跳过。")
    sys.exit(0)

os.makedirs(os.path.join(CACHE_DIR, "crypto"), exist_ok=True)

# ─── Download data ───
tickers = ['BTC-USD', 'ETH-USD', 'SOL-USD']
names = {'BTC-USD': 'BTC', 'ETH-USD': 'ETH', 'SOL-USD': 'SOL'}
CRYPTO_CACHE_DIR = os.path.join(CACHE_DIR, "crypto")
os.makedirs(CRYPTO_CACHE_DIR, exist_ok=True)
CACHE_TTL = 12 * 3600  # 12h硬缓存

def get_crypto_data(ticker, period="6mo"):
    """带缓存回退的加密数据获取（与v5系统相同逻辑）"""
    safe = ticker.replace("-", "_")
    path = os.path.join(CRYPTO_CACHE_DIR, f"{safe}_{period}.pkl")
    now_ts = time.time()

    # 缓存有效 → 直接返回
    if os.path.exists(path):
        mtime = os.path.getmtime(path)
        age = now_ts - mtime
        if age < CACHE_TTL:
            try:
                df = pd.read_pickle(path)
                # 尝试增量更新 (每4h补充最新数据)
                if age > 4 * 3600:
                    try:
                        last_date = df.index[-1]
                        delta = yf.download(ticker, start=last_date - timedelta(2),
                                            auto_adjust=True, progress=False)
                        if not delta.empty:
                            if delta.columns.nlevels > 1:
                                delta.columns = [c[0] for c in delta.columns]
                            new_rows = delta.index.difference(df.index)
                            if len(new_rows) > 0:
                                df = pd.concat([df, delta.loc[new_rows]])
                                df.to_pickle(path)
                    except:
                        pass
                return df
            except:
                pass

    # 实时拉取
    for attempt in range(3):
        try:
            time.sleep(0.3)  # 防限流: 每次尝试前等待
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if df.empty:
                df = yf.download(ticker, period="3mo", auto_adjust=True, progress=False)
            if not df.empty:
                if df.columns.nlevels > 1:
                    df.columns = [c[0] for c in df.columns]
                df.to_pickle(path)
                return df
            time.sleep(3)
        except Exception:
            time.sleep(3)

    # 回退到过期缓存
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except:
            pass
    return None

all_data = {}
for t in tickers:
    df = get_crypto_data(t)
    if df is not None:
        all_data[t] = df

if not all_data:
    print("无法获取加密数据，跳过。")
    sys.exit(1)

# ═══ Fear & Greed Index ═══
try:
    import urllib.request
    fng_resp = urllib.request.urlopen("https://api.alternative.me/fng/?limit=1", timeout=5).read()
    fng_data = json.loads(fng_resp)
    fng_value = int(fng_data['data'][0]['value'])
    fng_class = fng_data['data'][0]['value_classification']
except:
    fng_value = 50
    fng_class = "Neutral"

# ═══ Score each crypto ═══
def score_crypto(c, v, name):
    ret = np.diff(np.log(c))
    price = float(c[-1])

    mom_21d = (c[-1] / c[-22] - 1) * 100 if len(c) >= 22 else 0
    mom_63d = (c[-1] / c[-64] - 1) * 100 if len(c) >= 64 else 0
    mom_126d = (c[-1] / c[-127] - 1) * 100 if len(c) >= 127 else 0

    vol_21d = float(np.std(ret[-21:]) * np.sqrt(365) * 100) if len(ret) >= 21 else 0
    vol_63d = float(np.std(ret[-63:]) * np.sqrt(365) * 100) if len(ret) >= 63 else vol_21d
    vol_7d = float(np.std(ret[-7:]) * np.sqrt(365) * 100) if len(ret) >= 7 else vol_21d

    peak = np.maximum.accumulate(c[-126:])
    max_dd = float(((c[-126:] - peak) / peak).min() * 100)

    gains = ret[-14:].copy()
    losses = -gains.copy()
    losses[losses < 0] = 0
    gains[gains < 0] = 0
    avg_g = float(np.mean(gains))
    avg_l = float(np.mean(losses)) if np.mean(losses) > 0 else 0.001
    rsi = float(100 - 100 / (1 + avg_g / avg_l))

    vol_ratio = float(np.mean(v[-21:])) / float(np.mean(v[-63:])) if np.mean(v[-63:]) > 0 else 1.0
    vol_trend_r = vol_21d / vol_63d if vol_63d > 0 else 1.0

    # Sub-scores
    mom_s = np.clip(0.6 * max(0, (mom_21d + 25) / 50) + 0.4 * max(0, (mom_63d + 35) / 70), 0, 1)
    vol_s = max(0, 1 - vol_21d / 100)
    rsi_s = 1 - abs(rsi - 55) / 45
    vol2_s = max(0, 1 - abs(vol_ratio - 1.15) / 0.5)
    dd_s = max(0, 1 - abs(max_dd) / 50)
    vt_s = max(0, 1 - abs(vol_trend_r - 1) / 0.5)

    ema10 = float(np.mean(c[-10:]))
    ema30 = float(np.mean(c[-30:])) if len(c) >= 30 else ema10
    ema60 = float(np.mean(c[-60:])) if len(c) >= 60 else ema30
    if c[-1] > ema10 > ema30 > ema60:
        trend = 0.8
    elif c[-1] > ema10 and ema10 > ema30:
        trend = 0.5
    elif c[-1] < ema10 < ema30 < ema60:
        trend = 0.1
    elif c[-1] > ema10:
        trend = 0.3
    else:
        trend = 0.2

    score = (0.30 * mom_s + 0.15 * vol_s + 0.15 * rsi_s +
             0.10 * vol2_s + 0.20 * trend + 0.05 * dd_s + 0.05 * vt_s)

    if score > 0.55:
        direction = "看涨"
    elif score < 0.40:
        direction = "看跌"
    else:
        direction = "震荡"

    # Fibonacci
    hi = float(np.max(c[-126:]))
    lo = float(np.min(c[-126:]))
    rng = hi - lo
    fib = {
        "0.236": hi - 0.236 * rng, "0.382": hi - 0.382 * rng,
        "0.500": hi - 0.500 * rng, "0.618": hi - 0.618 * rng,
        "0.786": hi - 0.786 * rng,
    }

    # MA
    ma20 = float(np.mean(c[-20:]))
    ma50 = float(np.mean(c[-50:])) if len(c) >= 50 else 0
    ma200 = float(np.mean(c[-200:])) if len(c) >= 200 else 0

    return {
        "name": name, "price": round(price, 2), "score": round(score, 4),
        "direction": direction,
        "mom_21d": round(mom_21d, 2), "mom_63d": round(mom_63d, 2),
        "mom_126d": round(mom_126d, 2),
        "vol_21d": round(vol_21d, 1), "vol_7d": round(vol_7d, 1), "vol_63d": round(vol_63d, 1),
        "rsi": round(rsi, 1), "max_dd": round(max_dd, 1),
        "vol_ratio": round(vol_ratio, 2), "vol_trend": round(vol_trend_r, 2),
        "sub_mom": round(mom_s, 4), "sub_vol": round(vol_s, 4),
        "sub_rsi": round(rsi_s, 4), "sub_trend": float(trend),
        "sub_vol2": round(vol2_s, 4), "sub_dd": round(dd_s, 4), "sub_vt": round(vt_s, 4),
        "fib": {k: round(v, 0) for k, v in fib.items()},
        "fib_hi": round(hi, 0), "fib_lo": round(lo, 0),
        "ma20": round(ma20, 0), "ma50": round(ma50, 0), "ma200": round(ma200, 0),
        "high_6mo": round(hi, 0), "low_6mo": round(lo, 0),
    }

results = []
for t in tickers:
    if t not in all_data:
        continue
    df = all_data[t]
    c = df['Close'].values.astype(float)
    v = df['Volume'].values.astype(float)
    r = score_crypto(c, v, names[t])
    results.append(r)

results.sort(key=lambda x: -x['score'])

# ═══ Save to state ═══
state["last_pred_date"] = today
state["last_pred_time"] = now.strftime("%H:%M")
state["last_predictions"] = results
state["fng_value"] = fng_value
state["fng_class"] = fng_class
with open(state_file, "w") as f:
    json.dump(state, f, indent=2)

# ═══ Print report ═══
print(f"📊 加密币 ML 评分预测 — {today} {now.strftime('%H:%M')} CST")
print(f"======================================")
print(f"🌐 恐惧贪婪指数: {fng_value}/100 ({fng_class})")
print()

print(f"{'#':>2} {'币种':>4} {'评分':>7} {'方向':>5} {'价格':>10} {'21d':>8} {'63d':>8} {'波动率':>7} {'RSI':>5}")
print(f"{'-'*60}")
for i, r in enumerate(results):
    print(f"{i+1:>2} {r['name']:>4} {r['score']:.4f} {r['direction']:>4} ${r['price']:>8,.0f} {r['mom_21d']:>+6.1f}% {r['mom_63d']:>+6.1f}% {r['vol_21d']:>5.0f}% {r['rsi']:>4.0f}")

print()
print("── 技术面拆解 ──")
for r in results:
    ma_status = "死叉" if r.get('ma50', 0) and r.get('ma200', 0) and r['ma50'] < r['ma200'] else ("金叉" if r.get('ma50', 0) and r.get('ma200', 0) else "—")
    print(f"{r['name']} @ ${r['price']:>,.0f}")
    print(f"  子分: 动量{r['sub_mom']:.3f} 波动率{r['sub_vol']:.3f} RSI{r['sub_rsi']:.3f} 趋势{r['sub_trend']:.1f} 量能{r['sub_vol2']:.3f}")
    print(f"  MA20=${r.get('ma20', 0):>,.0f} MA50=${r.get('ma50', 0):>,.0f} MA200=${r.get('ma200', 0):>,.0f} ({ma_status})")
    print(f"  斐波那契: 阻0.236=${r['fib']['0.236']:>,.0f} 支0.618=${r['fib']['0.618']:>,.0f}")
    print()

# 相关性
if len([t for t in tickers if t in all_data]) >= 2:
    print("── 相关性 (60d) ──")
    valid = [t for t in tickers if t in all_data]
    for i in range(len(valid)):
        for j in range(i+1, len(valid)):
            r1 = np.diff(np.log(all_data[valid[i]]['Close'].values.astype(float)[-61:]))
            r2 = np.diff(np.log(all_data[valid[j]]['Close'].values.astype(float)[-61:]))
            corr = np.corrcoef(r1, r2)[0, 1]
            print(f"  {names[valid[i]]}↔{names[valid[j]]}: {corr:.3f}")
    print()

# 预测方向摘要
print("── 今日方向判断 ──")
for r in results:
    print(f"  {r['name']}: {r['direction']} (评分{r['score']:.4f})")

print()
print("⚠️ 加密市场24h交易，无明显开盘/收盘。上述评分基于日线数据，仅供参考。")
