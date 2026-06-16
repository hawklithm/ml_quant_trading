#!/usr/bin/env python3
"""
市场状态检测模块 (P2.1)
=====================
检测当前市场状态 (BULL/BEAR/SIDEWAYS/HIGH_VOL/LOW_VOL)
并根据状态返回自适应因子权重建议。

用法:
  python market_state.py                        # 打印当前市场状态
"""

import numpy as np
import warnings, sys, os
from datetime import datetime

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

LOOKBACK = 252  # 1年回看


def get_market_state(spy_data=None, period="6mo"):
    """
    检测市场状态。

    Args:
        spy_data: SPY OHLCV DataFrame，None 则自动下载
        period: 数据周期

    Returns:
        dict: {state, confidence, weights, features}
    """
    if spy_data is None:
        import pandas as pd
        import yfinance as yf
        spy = yf.download("SPY", period=period, auto_adjust=True, progress=False)
        if spy.empty:
            return _fallback_state()
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = [c[0] for c in spy.columns]
        spy_data = spy

    import pandas as pd
    import numpy as np
    close = spy_data["Close"].values.astype(float).ravel()
    high = spy_data["High"].values.astype(float).ravel()
    low = spy_data["Low"].values.astype(float).ravel()
    n = len(close)

    ret = pd.Series(close).pct_change()

    # --- 特征计算 ---
    features = {}

    # 1. 近期动量 (21天)
    mom_21 = close[-1] / close[-22] - 1 if n >= 22 else 0
    features["momentum_21d"] = round(float(mom_21 * 100), 2)

    # 2. 中期趋势 (63天)
    mom_63 = close[-1] / close[-64] - 1 if n >= 64 else 0
    features["momentum_63d"] = round(float(mom_63 * 100), 2)

    # 3. 200日均线偏离
    if n >= 200:
        sma200 = np.mean(close[-200:])
        sma_dev = close[-1] / sma200 - 1
    else:
        sma_dev = 0
    features["sma200_dev"] = round(float(sma_dev * 100), 2)

    # 4. 波动率
    vol_21 = ret.tail(21).std() * np.sqrt(252) if n >= 22 else 0
    vol_63 = ret.tail(63).std() * np.sqrt(252) if n >= 64 else 0
    vol_252 = ret.tail(252).std() * np.sqrt(252) if n >= 253 else vol_63
    features["vol_21d"] = round(float(vol_21 * 100), 2)
    features["vol_63d"] = round(float(vol_63 * 100), 2)
    features["vol_252d"] = round(float(vol_252 * 100), 2)

    # 5. 波动率比 (短期/长期)
    vol_ratio = vol_21 / vol_63 if vol_63 > 0 else 1.0
    features["vol_ratio_21_63"] = round(float(vol_ratio), 4)

    # 6. 最大回撤
    cummax = np.maximum.accumulate(close)
    dd = (close - cummax) / cummax
    max_dd = np.min(dd)
    features["max_dd_252d"] = round(float(max_dd * 100), 2)

    # 7. 波动率历史百分位 (过去 252 天波动率在 5 年中的位置)
    # 用滚动 63d 波动率的分位数
    if n >= 252:
        rolling_vol = ret.rolling(63).std() * np.sqrt(252) * 100
        current_vol = rolling_vol.iloc[-1]
        vol_percentile = (rolling_vol.dropna() < current_vol).mean()
    else:
        vol_percentile = 0.5
    features["vol_percentile"] = round(float(vol_percentile), 4)

    # --- 状态判定 ---
    # 趋势状态
    trend_signals = []

    # 长期动量 > 10% → BULL
    if mom_63 > 0.10:
        trend_signals.append("BULL")
    elif mom_63 < -0.10:
        trend_signals.append("BEAR")
    else:
        trend_signals.append("SIDEWAYS")

    # 200日均线偏离 > 5% → BULL
    if sma_dev > 0.05:
        trend_signals.append("BULL")
    elif sma_dev < -0.05:
        trend_signals.append("BEAR")
    else:
        trend_signals.append("SIDEWAYS")

    # 21天动量方向
    if mom_21 > 0.03:
        trend_signals.append("BULL")
    elif mom_21 < -0.03:
        trend_signals.append("BEAR")
    else:
        trend_signals.append("SIDEWAYS")

    # 投票
    bull_votes = trend_signals.count("BULL")
    bear_votes = trend_signals.count("BEAR")
    side_votes = trend_signals.count("SIDEWAYS")

    if bull_votes >= 2:
        state = "BULL"
        confidence = bull_votes / len(trend_signals)
    elif bear_votes >= 2:
        state = "BEAR"
        confidence = bear_votes / len(trend_signals)
    else:
        state = "SIDEWAYS"
        confidence = side_votes / len(trend_signals)

    # 波动率状态修正
    vol_state = "NORMAL"
    if vol_percentile > 0.8:
        vol_state = "HIGH_VOL"
        state = f"{state}_HIGH_VOL"
    elif vol_percentile < 0.2:
        vol_state = "LOW_VOL"
        if state == "SIDEWAYS":
            state = "LOW_VOL_SIDEWAYS"

    # --- 因子权重 (根据不同状态) ---
    weights = _factor_weights_for_state(state, vol_state)

    features["vol_state"] = vol_state
    features["trend_state"] = state.split("_")[0]
    features["bull_votes"] = bull_votes
    features["bear_votes"] = bear_votes
    features["side_votes"] = side_votes

    return {
        "state": state,
        "confidence": round(float(confidence), 4),
        "vol_state": vol_state,
        "weights": weights,
        "features": features,
        "timestamp": datetime.now().isoformat(),
    }


def _factor_weights_for_state(state, vol_state="NORMAL"):
    """根据市场状态返回推荐的因子权重"""
    base_state = state.split("_")[0]

    if base_state == "BULL":
        # 牛市: 动量 > 价值 > 低波 > 质量 > 成长
        weights = {
            "momentum": 0.35,
            "value": 0.20,
            "low_vol": 0.10,
            "quality": 0.15,
            "growth": 0.20,
        }
    elif base_state == "BEAR":
        # 熊市: 低波 > 质量 > 价值 > 成长 > 动量
        weights = {
            "momentum": 0.05,
            "value": 0.20,
            "low_vol": 0.40,
            "quality": 0.25,
            "growth": 0.10,
        }
    else:  # SIDEWAYS
        # 震荡: 低波 > 价值 > 质量 > 动量 > 成长
        weights = {
            "momentum": 0.15,
            "value": 0.25,
            "low_vol": 0.30,
            "quality": 0.20,
            "growth": 0.10,
        }

    # 高波动环境 → 进一步偏向低波
    if vol_state == "HIGH_VOL":
        weights["momentum"] *= 0.7
        weights["low_vol"] *= 1.5
        weights["growth"] *= 0.7
        # 归一化
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total
    # 低波动环境 → 增加动量暴露
    elif vol_state == "LOW_VOL":
        weights["momentum"] *= 1.3
        weights["low_vol"] *= 0.7
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

    # 四舍五入
    for k in weights:
        weights[k] = round(weights[k], 4)

    return weights


def adaptive_factor_weights(spy_data=None):
    """快捷函数: 只返回自适应因子权重"""
    state_info = get_market_state(spy_data)
    return state_info["weights"], state_info["state"]


def _fallback_state():
    """数据不足时的兜底状态"""
    return {
        "state": "UNKNOWN",
        "confidence": 0.0,
        "vol_state": "NORMAL",
        "weights": {
            "momentum": 0.20,
            "value": 0.20,
            "low_vol": 0.20,
            "quality": 0.20,
            "growth": 0.20,
        },
        "features": {},
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
if __name__ == "__main__":
    import pandas as pd

    state_info = get_market_state()

    print(f"\n{'='*60}")
    print(f"  市场状态检测 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")

    s = state_info["state"]
    c = state_info["confidence"]
    vs = state_info["vol_state"]

    emoji_map = {
        "BULL": "🟢", "BEAR": "🔴", "SIDEWAYS": "🟡",
        "BULL_HIGH_VOL": "🟠", "BEAR_HIGH_VOL": "🔴⚠️",
        "LOW_VOL_SIDEWAYS": "🟢",
    }
    emoji = emoji_map.get(s, "⚪")

    print(f"\n  状态:     {emoji} {s}")
    print(f"  可信度:   {c:.1%}")
    print(f"  波动率:   {vs}")

    feats = state_info["features"]
    if feats:
        print(f"\n  特征:")
        print(f"    21天动量:    {feats.get('momentum_21d', 0):+.2f}%")
        print(f"    63天动量:    {feats.get('momentum_63d', 0):+.2f}%")
        print(f"    200SMA偏离:  {feats.get('sma200_dev', 0):+.2f}%")
        print(f"    21天波动率:  {feats.get('vol_21d', 0):.2f}%")
        print(f"    63天波动率:  {feats.get('vol_63d', 0):.2f}%")
        print(f"    波动百分位:  {feats.get('vol_percentile', 0):.1%}")
        print(f"    最大回撤:    {feats.get('max_dd_252d', 0):.2f}%")

    w = state_info["weights"]
    print(f"\n  推荐因子权重:")
    for k in ["momentum", "value", "low_vol", "quality", "growth"]:
        bar = "█" * int(w.get(k, 0) * 20)
        print(f"    {k:<10}: {w.get(k, 0):>6.1%} {bar}")

    print()
