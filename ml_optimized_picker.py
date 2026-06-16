#!/usr/bin/env python3
"""
ML 优化版选股系统 v4
====================
新增 (vs v3):
  P1.1 自适应训练窗口 — 锦标赛选择自动挑最佳窗口长度
  P1.2 横截面排名 — 同板块股票之间排名替代绝对预测值

用法:
  python ml_optimized_picker.py                              # 运行完整 ML 选股
  python ml_optimized_picker.py --top 30                     # Top 30 结果
  python ml_optimized_picker.py --models rf,xgb,lgb,ensemble # 指定模型
  python ml_optimized_picker.py --quick                      # 单模型快速模式
  python ml_optimized_picker.py --adaptive                   # 启用自适应窗口 (默认)
  python ml_optimized_picker.py --cross-section              # 启用横截面排名
"""

import numpy as np
import pandas as pd
import warnings, sys, os, json
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ──── ML 模型 ────
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import joblib

# ── 可选模型 ──
try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    import networkx as nx
    HAS_NX = True
except:
    HAS_NX = False

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

TOP_N = 20
MIN_TRADING_DAYS = 252
TEST_SPLITS = 5
FORECAST_HORIZON = 5

# 自适应窗口
ADAPTIVE_WINDOWS = [504, 756, 1008]  # 2年, 3年, 4年 (交易日)
ADAPTIVE_WARMUP = 20                 # 最少测试样本

# 板块分组 (用于横截面排名)
SECTOR_MAP = {
    "tech": {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","ORCL",
             "AMD","INTC","QCOM","TXN","MU","ASML","ARM","TSM","PLTR"},
    "finance": {"JPM","BAC","GS","MS","V","MA","BLK","SOFI","HOOD","COIN","MSTR"},
    "consumer": {"WMT","COST","HD","LOW","PG","KO","PEP","MCD","SBUX","NKE"},
    "healthcare": {"UNH","JNJ","PFE","ABBV","MRK","LLY","TMO"},
    "energy": {"XOM","CVX","XLE"},
    "industrial": {"CAT","GE","BA","HON"},
    "etf": {"SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLI","XLU","XLRE","XLC"},
    "other": {"RDDT"},
}

MODELS = {
    "rf":  lambda: RandomForestRegressor(n_estimators=200, max_depth=6,
                                         min_samples_leaf=10, n_jobs=-1, random_state=42),
    "xgb": lambda: xgb.XGBRegressor(n_estimators=200, max_depth=5,
                                     learning_rate=0.05, n_jobs=-1, random_state=42) if HAS_XGB else None,
    "lgb": lambda: lgb.LGBMRegressor(n_estimators=200, max_depth=5,
                                      learning_rate=0.05, n_jobs=-1, random_state=42,
                                      verbose=-1) if HAS_LGB else None,
    "lasso": lambda: LassoCV(cv=5, max_iter=5000, random_state=42),
}

US_WATCHLIST = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AVGO","ORCL",
    "AMD","INTC","QCOM","TXN","MU","ASML","ARM",
    "JPM","BAC","GS","MS","V","MA","BLK",
    "WMT","COST","HD","LOW","PG","KO","PEP","MCD","SBUX","NKE",
    "UNH","JNJ","PFE","ABBV","MRK","LLY","TMO",
    "XOM","CVX","CAT","GE","BA","HON",
    "SPY","QQQ","IWM","DIA",
    "XLF","XLK","XLV","XLE","XLI","XLU","XLRE","XLC",
    "PLTR","SOFI","RDDT","HOOD","MSTR","COIN","TSM",
]


def get_ticker_sector(ticker):
    """获取股票所属板块"""
    for sector, stocks in SECTOR_MAP.items():
        if ticker in stocks:
            return sector
    return "other"


# ═══════════════════════════════════════
# 1. 特征工程（与 v3 一致）
# ═══════════════════════════════════════
def build_features_v3(df):
    """从历史 OHLCV DataFrame 构建扩展特征集"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    close = df["Close"].values.astype(float).ravel()
    high = df["High"].values.astype(float).ravel()
    low = df["Low"].values.astype(float).ravel()
    volume = df["Volume"].values.astype(float).ravel()
    idx = df.index

    ret = pd.Series(close, index=idx).pct_change()
    f = pd.DataFrame(index=idx)

    # A. 动量特征
    for p in [5, 10, 21, 42, 63, 126, 252]:
        f[f"mom_{p}d"] = pd.Series(close, index=idx).pct_change(p)
    mom_21 = pd.Series(close, index=idx).pct_change(21)
    mom_63 = pd.Series(close, index=idx).pct_change(63)
    f["mom_accel"] = mom_21 - mom_63.shift(63)

    # B. 均线偏离
    for p in [10, 20, 50, 100, 200]:
        sma = pd.Series(close, index=idx).rolling(p).mean()
        f[f"sma{p}_dev"] = pd.Series(close, index=idx) / sma - 1
    for p in [20, 50]:
        sma_series = pd.Series(close, index=idx).rolling(p).mean()
        f[f"sma{p}_slope"] = sma_series.pct_change(5) * 100

    # C. 波动率特征
    for p in [5, 10, 21, 63]:
        f[f"vol_{p}d"] = ret.rolling(p).std() * np.sqrt(252)
    f["vol_ratio_21_63"] = f["vol_21d"] / f["vol_63d"]
    f["vol_ratio_5_21"] = f["vol_5d"] / f["vol_21d"]
    f["vol_change_21"] = f["vol_21d"].pct_change(21)

    # D. RSI 多周期
    for p in [7, 14, 21]:
        delta = ret
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss
        f[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # E. 价格位置与形态
    for p in [20, 60, 120]:
        h = pd.Series(high, index=idx).rolling(p).max()
        l = pd.Series(low, index=idx).rolling(p).min()
        f[f"price_pos_{p}"] = (pd.Series(close, index=idx) - l) / (h - l)
    sma20 = pd.Series(close, index=idx).rolling(20).mean()
    std20 = pd.Series(close, index=idx).rolling(20).std()
    f["bb_width"] = (2 * std20) / sma20
    f["bb_position"] = (pd.Series(close, index=idx) - sma20) / (2 * std20)
    for p in [10, 20, 63]:
        h_p = pd.Series(high, index=idx).rolling(p).max()
        l_p = pd.Series(low, index=idx).rolling(p).min()
        f[f"hl_ratio_{p}"] = h_p / l_p - 1

    # F. 成交量特征
    vol_s = pd.Series(volume, index=idx)
    f["volume_ratio"] = vol_s / vol_s.rolling(20).mean()
    f["volume_trend"] = vol_s.pct_change(5)
    f["volume_std"] = vol_s.rolling(20).std() / vol_s.rolling(20).mean()
    obv = (np.sign(ret) * vol_s).cumsum()
    f["obv_trend"] = obv.pct_change(21)
    f["vol_price_corr_20"] = ret.rolling(20).corr(vol_s.pct_change())

    # G. 风险调整指标
    cum = (1 + ret).cumsum()
    dd = cum / cum.cummax() - 1
    f["calmar_60"] = ret.rolling(60).mean() * 252 / (-dd.rolling(60).min())
    f["skew_21"] = ret.rolling(21).skew()
    f["kurt_21"] = ret.rolling(21).kurt()

    # 目标
    target = pd.Series(close, index=idx).pct_change(FORECAST_HORIZON).shift(-FORECAST_HORIZON)

    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[:, f.notna().any()]
    valid = f.dropna(thresh=len(f.columns) * 0.7).index
    f = f.loc[valid]
    target = target.loc[valid]

    return f.astype(np.float32), target.rename("target")


# ═══════════════════════════════════════
# P1.1: 自适应训练窗口 (锦标赛选择)
# ═══════════════════════════════════════
def tournament_window_selection(stock_data, ticker, model_builder, windows=ADAPTIVE_WINDOWS):
    """
    锦标赛选择最佳训练窗口长度。

    对每个窗口长度:
      1. 取最近 N 天数据
      2. 用 walk-forward 的最后 1 折做验证
      3. 选 Walk-Forward R² 最高的窗口

    Returns:
      best_window: int (交易日数)
      best_r2: float
      window_scores: {window: r2}
    """
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data = stock_data.copy()
        stock_data.columns = [c[0] for c in stock_data.columns]

    close_raw = stock_data["Close"].values.astype(float).ravel()

    if len(close_raw) < min(windows) + ADAPTIVE_WARMUP:
        # 数据不够，返回最长可用窗口
        return max(ADAPTIVE_WINDOWS) if max(ADAPTIVE_WINDOWS) <= len(close_raw) else len(close_raw), 0, {}

    window_scores = {}
    tscv = TimeSeriesSplit(n_splits=TEST_SPLITS)
    scaler = StandardScaler()

    for w in windows:
        if w >= len(close_raw):
            continue
        # 截取最近 w 天
        df_w = stock_data.iloc[-w:].copy()
        try:
            features, target = build_features_v3(df_w)
            if len(features) < 100:
                continue

            X = features.dropna()
            y = target.loc[X.index].dropna()
            common = X.index.intersection(y.index)
            X = X.loc[common]
            y = y.loc[common]

            if len(X) < 80:
                continue

            # 只用最后 1 折来评估
            splits = list(tscv.split(X))
            if len(splits) < 2:
                continue

            # 最后 2 折作为验证
            tr_idx, te_idx = splits[-2]
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

            if len(X_te) < ADAPTIVE_WARMUP:
                continue

            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            m = model_builder()
            m.fit(X_tr_s, y_tr)
            y_pred = m.predict(X_te_s)
            r2_te = r2_score(y_te, y_pred)
            window_scores[w] = round(r2_te, 4)
        except Exception:
            continue

    if not window_scores:
        return max(ADAPTIVE_WINDOWS) if max(ADAPTIVE_WINDOWS) <= len(close_raw) else len(close_raw), 0, {}

    best_w = max(window_scores, key=window_scores.get)
    return best_w, window_scores[best_w], window_scores


# ═══════════════════════════════════════
# P1.2: 横截面排名 (Cross-Sectional Rank)
# ═══════════════════════════════════════
def compute_cross_sectional_ranks(all_stock_scores):
    """
    对所有股票的 ML 预测值做横截面排名。

    全局排名: 所有股票一起排名
    板块内排名: 同板块股票排名

    返回:
      ranked_list: [ticker, score, global_rank, sector, sector_rank, final_score]
    """
    if not all_stock_scores:
        return []

    tickers = []
    scores = []
    sectors = []

    for sr in all_stock_scores:
        tickers.append(sr["ticker"])
        scores.append(sr["score"])
        sectors.append(get_ticker_sector(sr["ticker"]))

    scores_arr = np.array(scores)

    # 全局百分位排名 (0~1)
    from scipy.stats import rankdata
    global_ranks = (rankdata(scores_arr) - 1) / (len(scores_arr) - 1)

    # 板块内排名
    sector_groups = {}
    for i, t in enumerate(tickers):
        sec = sectors[i]
        if sec not in sector_groups:
            sector_groups[sec] = []
        sector_groups[sec].append((i, scores_arr[i]))

    sector_ranks = np.zeros(len(tickers))
    for sec, members in sector_groups.items():
        if len(members) < 2:
            # 孤板块，给 0.5
            for idx, _ in members:
                sector_ranks[idx] = 0.5
        else:
            sec_scores = np.array([m[1] for m in members])
            sec_ranks = (rankdata(sec_scores) - 1) / (len(sec_scores) - 1)
            for (idx, _), r in zip(members, sec_ranks):
                sector_ranks[idx] = r

    # 最终评分: 全球排 * 0.5 + 板块排 * 0.4 + 原始分 * 0.1
    results = []
    for i in range(len(tickers)):
        final_score = global_ranks[i] * 0.5 + sector_ranks[i] * 0.4 + scores_arr[i] * 0.1
        results.append({
            "ticker": tickers[i],
            "score_original": round(float(scores_arr[i]), 4),
            "global_rank": round(float(global_ranks[i]), 4),
            "sector": sectors[i],
            "sector_rank": round(float(sector_ranks[i]), 4),
            "score": round(float(final_score), 4),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ═══════════════════════════════════════
# 2. Walk-Forward 训练与评估
# ═══════════════════════════════════════
def train_model_walk_forward(X, y, model_builder, n_splits=TEST_SPLITS, adaptive_window_data=None):
    """Walk-Forward 滚动验证，支持自适应窗口"""
    if adaptive_window_data is not None and len(X) > 0:
        # 用自适应窗口训练
        pass  # 窗口选择在 score_stock_by_ml 中完成

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()

    predictions = pd.Series(index=y.index, dtype=np.float32)
    feature_importances = []
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = model_builder()
        model.fit(X_tr_s, y_tr)
        y_pred = model.predict(X_te_s)
        predictions.iloc[test_idx] = y_pred

        fold_r2 = r2_score(y_te, y_pred)
        fold_mae = mean_absolute_error(y_te, y_pred)
        fold_metrics.append({"fold": fold, "r2": fold_r2, "mae": fold_mae})

        if hasattr(model, "feature_importances_"):
            feature_importances.append(model.feature_importances_)
        elif hasattr(model, "coef_"):
            feature_importances.append(np.abs(model.coef_))

    X_s = scaler.fit_transform(X)
    final_model = model_builder()
    final_model.fit(X_s, y)

    avg_importance = None
    if feature_importances:
        avg_importance = np.mean(feature_importances, axis=0)

    metrics = {
        "walk_forward_r2": r2_score(y.dropna(), predictions.dropna()),
        "walk_forward_mae": mean_absolute_error(y.dropna(), predictions.dropna()),
        "fold_metrics": fold_metrics,
        "feature_names": X.columns.tolist(),
    }
    if avg_importance is not None:
        metrics["feature_importance"] = {
            X.columns[i]: float(avg_importance[i])
            for i in np.argsort(avg_importance)[::-1][:20]
        }

    return final_model, scaler, metrics, predictions


# ═══════════════════════════════════════
# 3. 个股 ML 评分 (自适应窗口版)
# ═══════════════════════════════════════
def score_stock_by_ml(ticker, period="2y", models_to_use=None, use_adaptive=False):
    """用 ML 模型对单只股票评分"""
    import yfinance as yf

    if models_to_use is None:
        models_to_use = ["rf", "xgb", "lgb"]

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty or len(df) < MIN_TRADING_DAYS:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # ── P1.1: 自适应窗口选择 ──
    if use_adaptive and len(df) > min(ADAPTIVE_WINDOWS):
        rf_builder = MODELS["rf"]
        best_w, best_r2, _ = tournament_window_selection(df, ticker, rf_builder)
        # 截取最佳窗口
        df_used = df.iloc[-best_w:].copy()
    else:
        df_used = df
        best_w = len(df_used)
        best_r2 = 0

    features, target = build_features_v3(df_used)
    if len(features) < 100:
        return None

    X = features.dropna()
    y = target.loc[X.index].dropna()
    common = X.index.intersection(y.index)
    X = X.loc[common]
    y = y.loc[common]

    if len(X) < 100:
        return None

    model_results = {}
    all_preds = []

    for name in models_to_use:
        builder_fn = MODELS.get(name)
        if builder_fn is None:
            continue
        try:
            # 检查 builder 返回是否非 None
            m = builder_fn()
            if m is None:
                continue
            model, scaler, metrics, preds = train_model_walk_forward(X, y, builder_fn)
            model_results[name] = {"metrics": metrics, "predictions": preds}
            all_preds.append(preds)
        except Exception:
            continue

    if not model_results:
        try:
            model, scaler, metrics, preds = train_model_walk_forward(X, y, MODELS["rf"])
            model_results["rf"] = {"metrics": metrics, "predictions": preds}
            all_preds = [preds]
        except:
            return None

    ensemble_pred = pd.concat(all_preds, axis=1).mean(axis=1)
    latest_pred = float(ensemble_pred.iloc[-1]) if len(ensemble_pred) > 0 else 0
    latest_actual = float(y.iloc[-1]) if len(y) > 0 else 0

    weights = {}
    total_w = 0
    for name, res in model_results.items():
        w = max(res["metrics"]["walk_forward_r2"], 0)
        weights[name] = w
        total_w += w

    if total_w > 0:
        weighted_r2 = sum(
            model_results[n]["metrics"]["walk_forward_r2"] * weights[n] / total_w
            for n in model_results
        )
    else:
        weighted_r2 = 0

    # 综合评分
    ref_returns = y.values
    if len(ref_returns) > 50:
        pctl = np.searchsorted(np.sort(ref_returns), latest_pred) / len(ref_returns)
        ml_score = max(0, min(1, (pctl - 0.4) / 0.5))
    else:
        ml_score = 0.5

    confidence = min(max(weighted_r2 * 3, 0.1), 1.0)
    final_score = ml_score * confidence

    closes = df_used["Close"].values.astype(float)
    cur_price = float(closes[-1])
    chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
    chg_5d = (closes[-1] / closes[-min(6, len(closes))] - 1) * 100
    mom_1m = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
    mom_3m = (closes[-1] / closes[-63] - 1) * 100 if len(closes) >= 63 else 0

    return {
        "ticker": ticker,
        "price": round(cur_price, 2),
        "score": round(final_score, 4),
        "ml_score": round(ml_score, 4),
        "confidence": round(confidence, 4),
        "pred_return": round(latest_pred * 100, 2),
        "actual_return": round(latest_actual * 100, 2),
        "walk_forward_r2": round(weighted_r2, 4),
        "adaptive_window": best_w,
        "adaptive_r2": round(best_r2, 4),
        "chg_1d_pct": round(chg_1d, 2),
        "chg_5d_pct": round(chg_5d, 2),
        "mom_1m": round(mom_1m, 2),
        "mom_3m": round(mom_3m, 2),
        "models_used": list(model_results.keys()),
        "models_r2": {n: round(res["metrics"]["walk_forward_r2"], 4)
                      for n, res in model_results.items()},
        "top_features": list(model_results[list(model_results.keys())[0]]
                             ["metrics"].get("feature_importance", {}).keys())[:10],
    }


# ═══════════════════════════════════════
# 4. 批量选股
# ═══════════════════════════════════════
def run_ml_picking(tickers=None, top_n=TOP_N, models_to_use=None, quick=False,
                   use_adaptive=False, use_cross_section=False):
    """批量 ML 选股"""
    if tickers is None:
        tickers = US_WATCHLIST
    if models_to_use is None:
        models_to_use = ["xgb", "lgb", "rf"] if not quick else ["rf"]

    features_str = "50+ (动量/均线/波动率/RSI/形态/成交量/风险)"
    if use_adaptive:
        features_str += " + 自适应窗口"
    if use_cross_section:
        features_str += " + 横截面排名"

    print(f"{'='*80}")
    print(f"  ML 优化选股系统 v4 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"  股票池: {len(tickers)} 只")
    print(f"  模型:   {', '.join(models_to_use)}")
    print(f"  特征:   {features_str}")
    print(f"  验证:   Walk-Forward ({TEST_SPLITS}-折)")
    print(f"  目标:   未来 {FORECAST_HORIZON} 日收益")
    if quick:
        print(f"  模式:   快速 (单模型 RF)")
    print(f"{'='*80}")

    results = []
    errors = []
    total = len(tickers)

    for i, t in enumerate(tickers):
        try:
            print(f"  [{i+1}/{total}] {t} ... ", end="", flush=True)
            sr = score_stock_by_ml(t, models_to_use=models_to_use, use_adaptive=use_adaptive)
            if sr is not None:
                results.append(sr)
                window_info = f" w={sr['adaptive_window']}" if use_adaptive else ""
                print(f"评分={sr['score']:.3f} R²={sr['walk_forward_r2']:.3f}{window_info}")
            else:
                errors.append(t)
                print(f"数据不足")
        except Exception as e:
            errors.append(t)
            print(f"失败: {e}")

    # ── P1.2: 横截面排名 ──
    if use_cross_section and results:
        ranked = compute_cross_sectional_ranks(results)
        # 把排名字段合并回去
        rank_map = {r["ticker"]: r for r in ranked}
        for r in results:
            if r["ticker"] in rank_map:
                rr = rank_map[r["ticker"]]
                r["score"] = rr["score"]
                r["global_rank"] = rr["global_rank"]
                r["sector"] = rr["sector"]
                r["sector_rank"] = rr["sector_rank"]
        results.sort(key=lambda x: x["score"], reverse=True)
    else:
        results.sort(key=lambda x: x["score"], reverse=True)

    return results, errors


# ═══════════════════════════════════════
# 5. 报告输出
# ═══════════════════════════════════════
def print_ml_report(results, top_n=TOP_N):
    """打印 ML 选股报告"""
    top = results[:top_n]
    print(f"\n{'='*120}")
    print(f"  ML 选股结果 Top {len(top)}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*120}")
    hdr = (f" {'#':>3} {'代码':>6} {'ML评分':>7} {'可信':>5} {'R²':>6} "
           f"{'预测5d':>8} {'价格':>8} {'1日':>7} {'5日':>7} {'1月':>7} {'3月':>7} "
           f"{'模型':>12} {'板块':>10}")
    if "global_rank" in results[0] if results else False:
        hdr += " {'全排':>6} {'板排':>6}"
    print(hdr)
    print(f" {'-'*115}")

    for i, s in enumerate(top):
        models_str = "+".join(s["models_used"])
        sector = get_ticker_sector(s["ticker"])
        line = (f" {i+1:>3} {s['ticker']:>6} {s['score']:>7.3f} {s['confidence']:>5.2f} "
                f"{s['walk_forward_r2']:>6.3f} {s['pred_return']:>+6.2f}% "
                f"${s['price']:>6.2f} {s['chg_1d_pct']:>+5.1f}% {s['chg_5d_pct']:>+5.1f}% "
                f"{s['mom_1m']:>+5.1f}% {s['mom_3m']:>+5.1f}% {models_str:>12} {sector:>10}")
        if "global_rank" in s:
            line += f" {s['global_rank']:>6.2f} {s['sector_rank']:>6.2f}"
        print(line)

    print(f"\n  摘要:")
    print(f"    成功评分: {len(results)} / {len(US_WATCHLIST)} 只股票")
    avg_score = np.mean([s["score"] for s in top])
    avg_r2 = np.mean([s["walk_forward_r2"] for s in top if s["walk_forward_r2"] > -1])
    print(f"    平均 ML 评分: {avg_score:.3f}")
    print(f"    平均 Walk-Forward R²: {avg_r2:.3f}")

    strong = [s for s in top if s["score"] > 0.3]
    watch = [s for s in top if 0.15 < s["score"] <= 0.3]
    print(f"    强烈关注 (评分>0.30): {', '.join(s['ticker'] for s in strong[:8]) or '无'}")
    print(f"    值得跟踪 (0.15-0.30): {', '.join(s['ticker'] for s in watch[:8]) or '无'}")

    return top


def save_ml_results(top, filename="ml_picks"):
    """保存 ML 选股结果"""
    rows = []
    for s in top:
        row = {
            "ticker": s["ticker"], "price": s["price"],
            "ml_score": s["score"], "confidence": s["confidence"],
            "walk_forward_r2": s["walk_forward_r2"],
            "pred_return_5d": s["pred_return"],
            "chg_1d_pct": s["chg_1d_pct"], "chg_5d_pct": s["chg_5d_pct"],
            "mom_1m": s["mom_1m"], "mom_3m": s["mom_3m"],
            "models_used": "+".join(s["models_used"]),
        }
        if "global_rank" in s:
            row["global_rank"] = s["global_rank"]
            row["sector"] = s["sector"]
            row["sector_rank"] = s["sector_rank"]
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {csv_path}")
    return csv_path


# ═══════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]

    top_n = TOP_N
    if "--top" in args:
        idx = args.index("--top")
        if idx + 1 < len(args):
            top_n = int(args[idx + 1])

    models = ["xgb", "lgb", "rf"]
    if "--models" in args:
        idx = args.index("--models")
        if idx + 1 < len(args):
            models = args[idx + 1].split(",")

    quick = "--quick" in args
    use_adaptive = "--adaptive" in args or "--no-adaptive" not in args
    use_cross_section = "--cross-section" in args

    if quick:
        models = ["rf"]

    results, errors = run_ml_picking(
        models_to_use=models, top_n=top_n, quick=quick,
        use_adaptive=use_adaptive, use_cross_section=use_cross_section,
    )

    if results:
        top = print_ml_report(results, top_n=min(top_n, len(results)))
        save_ml_results(top)
        print(f"\n  ✅ ML 选股完成!")
    else:
        print(f"\n  ❌ 没有成功评分的股票")
        if errors:
            print(f"     失败: {', '.join(errors[:10])}")

    if errors:
        print(f"\n  ⚠️  数据不足/失败: {len(errors)} 只: {', '.join(errors[:10])}")
