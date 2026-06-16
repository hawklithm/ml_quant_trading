#!/usr/bin/env python3
"""
ML Stock Picker v4 — 优化版机器学习多因子选股系统

核心设计:
  1. 目标 → 未来 5 日收益（原始值，不做秩变换导致评估失真）
  2. 评估 → Spearman 排序相关性（选股排名一致性） + 方向准确率
  3. 模型 → CatBoost (默认) + XGBoost + RF Ensemble
  4. Walk-Forward 时序验证（无数据泄露）
  5. 输出: 每只股票的 ML 预测收益 + 可预测性评分 + 排序
  6. 对比: 原版固定权重 vs ML 排序

用法:
  python ml_stock_picker.py                         # 全量 Top 20
  python ml_stock_picker.py --top 10                # Top 10
  python ml_stock_picker.py --ticker AAPL           # 单只分析
  python ml_stock_picker.py --compare               # ML vs 传统因子对比
"""

import numpy as np
import pandas as pd
import warnings, sys
from datetime import datetime
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore")

# ════════════════════════════════════════
# 配置
# ════════════════════════════════════════
TOP_N = 20
MIN_TRADING_DAYS = 252
FORECAST = 5  # 预测未来 N 日收益

# Walk-Forward
WF_SPLITS = 4
WF_INITIAL = 0.55

MODELS = ["cb", "xgb", "rf"]

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


# ════════════════════════════════════════
# 特征工程 (~35个)
# ════════════════════════════════════════
def build_features(close, high, low, volume):
    n = len(close)
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1

    feats, names = [], []

    # 动量
    for p in [5,10,21,42,63,126]:
        if n > p:
            v = np.zeros(n)
            v[p:] = close[p:] / close[:-p] - 1
            feats.append(v); names.append(f"mom_{p}d")

    # 波动率
    for p in [5,21,63]:
        if n > p+1:
            v = np.zeros(n)
            for i in range(p, n):
                v[i] = np.std(ret[i-p+1:i+1])
            feats.append(v); names.append(f"vol_{p}d")

    # 均线偏离
    for p in [10,20,50,200]:
        if n > p:
            sma = np.full(n, np.nan)
            for i in range(p-1, n):
                sma[i] = np.mean(close[i-p+1:i+1])
            sma = np.nan_to_num(sma, 0)
            feats.append(close / np.maximum(sma, 1e-10) - 1)
            names.append(f"sma{p}_dev")

    # 均线斜率
    for p in [20,50]:
        if n > p+5:
            v = np.zeros(n)
            for i in range(p+4, n):
                s1, s2 = np.mean(close[i-p+1:i+1]), np.mean(close[i-p:i])
                v[i] = (s1 / s2 - 1) * 100 if s2 > 0 else 0
            feats.append(v); names.append(f"sma{p}_slope")

    # RSI
    rsi = np.full(n, 50.0)
    if n > 15:
        for i in range(14, n):
            g = np.maximum(ret[i-13:i+1], 0); l = -np.minimum(ret[i-13:i+1], 0)
            ag, al = np.mean(g), np.mean(l)
            rsi[i] = 100 - 100/(1+ag/al) if al > 0 else (100 if ag > 0 else 50)
    feats.append(rsi); names.append("rsi_14")

    # 价格位置
    pos = np.full(n, 0.5)
    if n > 60:
        for i in range(59, n):
            hh, ll = np.max(close[i-59:i+1]), np.min(close[i-59:i+1])
            pos[i] = (close[i]-ll)/(hh-ll) if hh > ll else 0.5
    feats.append(pos); names.append("price_position")

    # 高低比
    for p in [20,63]:
        if n > p:
            v = np.zeros(n)
            for i in range(p-1, n):
                hh, ll = np.max(high[i-p+1:i+1]), np.min(low[i-p+1:i+1])
                v[i] = hh/ll-1 if ll > 0 else 0
            feats.append(v); names.append(f"hl_ratio_{p}d")

    # 成交量比
    for p in [5,20]:
        if n > p:
            v = np.ones(n)
            for i in range(p-1, n):
                v[i] = volume[i] / max(np.mean(volume[i-p+1:i+1]), 1)
            feats.append(v); names.append(f"vol_ratio_{p}d")

    # 布林带 Z
    if n > 20:
        bb = np.zeros(n)
        for i in range(19, n):
            sma, std = np.mean(close[i-19:i+1]), np.std(close[i-19:i+1])
            bb[i] = (close[i]-sma)/std if std > 0 else 0
        feats.append(bb); names.append("bb_z_20")

    # 最大回撤
    if n > 63:
        mdd = np.zeros(n)
        for i in range(62, n):
            peak = np.maximum.accumulate(close[i-62:i+1])
            mdd[i] = np.min((close[i-62:i+1]-peak)/peak)
        feats.append(mdd); names.append("max_dd_63d")

    # 自相关
    if n > 22:
        ac = np.zeros(n)
        for i in range(21, n):
            seg = ret[i-20:i+1] - np.mean(ret[i-20:i+1])
            if np.var(ret[i-20:i+1]) > 1e-10:
                ac[i] = np.mean(seg[1:]*seg[:-1]) / (np.var(ret[i-20:i+1])+1e-10)
        feats.append(ac); names.append("autocorr_21d")

    # 日内波动
    feats.append(np.where(close > 0, (high-low)/close, 0))
    names.append("daily_range")

    # 对数价格
    feats.append(np.log(np.maximum(close, 0.01)))
    names.append("log_price")

    X = np.column_stack(feats)
    X = pd.DataFrame(X).ffill().bfill().values
    return X, names


# ═══════════════════════════════════════��
# Walk-Forward 分裂
# ════════════════════════════════════════
def wf_splits(n, n_splits=WF_SPLITS, init_frac=WF_INITIAL):
    sz = int(n * (1 - init_frac) / n_splits)
    end = int(n * init_frac)
    splits = []
    for _ in range(n_splits):
        s = end; e = min(s + sz, n)
        if e - s < 10: break
        splits.append((list(range(end)), list(range(s, e))))
        end = e
    return splits


# ════════════════════════════════════════
# 训练 + Walk-Forward 验证
# ════════════════════════════════════════
def train_stock(X, y, model_names=None):
    """训练多模型，返回 ensemble 预测 + 评估指标"""
    if model_names is None:
        model_names = MODELS
    n = len(X)
    splits = wf_splits(n)
    if len(splits) < 2:
        return {}, np.full(n, np.nan)

    preds_m = {m: np.full(n, np.nan) for m in model_names}
    fold_r2 = {m: [] for m in model_names}

    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb
    import catboost as cb

    for tr, te in splits:
        if len(te) < 10: continue
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # CatBoost
        cbm = cb.CatBoostRegressor(
            iterations=150, depth=5, learning_rate=0.08,
            l2_leaf_reg=5, verbose=0, random_seed=42
        )
        cbm.fit(X_tr, y_tr)
        preds_m["cb"][te] = cbm.predict(X_te)

        # XGB
        xm = xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42
        )
        xm.fit(X_tr, y_tr)
        preds_m["xgb"][te] = xm.predict(X_te)

        # RF
        rm = RandomForestRegressor(
            n_estimators=150, max_depth=5, min_samples_leaf=15,
            n_jobs=-1, random_state=42
        )
        rm.fit(X_tr, y_tr)
        preds_m["rf"][te] = rm.predict(X_te)

    # Ensemble
    stack = np.column_stack([preds_m[m] for m in model_names])
    ensemble = np.nanmean(stack, axis=1)

    # 尾部补全：最后一折训练集上再 predict 整个尾巴
    last_tr, last_te = splits[-1]
    if last_te[-1] < n - 1 or np.isnan(ensemble[-1]):
        tail_start = last_te[0]
        X_tr = X[last_tr]
        y_tr = y[last_tr]
        X_t = X[tail_start:]
        if len(X_t) > 0:
            cbm = cb.CatBoostRegressor(iterations=150, depth=5, learning_rate=0.08,
                                        l2_leaf_reg=5, verbose=0, random_seed=42)
            cbm.fit(X_tr, y_tr)
            et = cbm.predict(X_t)

            xm = xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08,
                                   subsample=0.8, colsample_bytree=0.8,
                                   verbosity=0, random_state=42)
            xm.fit(X_tr, y_tr)
            et += xm.predict(X_t)

            rm = RandomForestRegressor(n_estimators=150, max_depth=5, min_samples_leaf=15,
                                        n_jobs=-1, random_state=42)
            rm.fit(X_tr, y_tr)
            et += rm.predict(X_t)
            et /= 3.0

            unc = np.isnan(ensemble[tail_start:])
            if unc.any():
                ensemble[tail_start:][unc] = et[unc]

    # 评估：只用有预测的位置
    metrics = {}
    for m in model_names:
        ok = ~np.isnan(preds_m[m])
        if ok.sum() > 10:
            from sklearn.metrics import r2_score, mean_absolute_error
            mr2 = r2_score(y[ok], preds_m[m][ok])
            mae = mean_absolute_error(y[ok], preds_m[m][ok])
            # 方向准确率
            dir_ok = np.mean((preds_m[m][ok] > 0) == (y[ok] > 0))
            # Spearman 排序相关（选股排名一致性）
            sp, _ = spearmanr(preds_m[m][ok], y[ok])
            metrics[m] = {
                "r2": round(mr2, 4),
                "mae": round(mae, 6),
                "dir_acc": round(dir_ok, 4),
                "spearman": round(sp, 4),
                "fold_r2": [round(s, 4) for s in fold_r2[m]],
            }

    ok = ~np.isnan(ensemble)
    if ok.sum() > 10:
        from sklearn.metrics import r2_score, mean_absolute_error
        sp_e, _ = spearmanr(ensemble[ok], y[ok])
        metrics["ensemble"] = {
            "r2": round(r2_score(y[ok], ensemble[ok]), 4),
            "mae": round(mean_absolute_error(y[ok], ensemble[ok]), 6),
            "dir_acc": round(np.mean((ensemble[ok] > 0) == (y[ok] > 0)), 4),
            "spearman": round(sp_e, 4),
        }

    return metrics, ensemble


# ═══════════════���════════════════════════
# 个股 ML 评分
# ════════════════════════════════════════
def score_stock(hist_df, full_output=False):
    close = hist_df["close"].values.astype(float)
    high  = hist_df["high"].values.astype(float)
    low   = hist_df["low"].values.astype(float)
    vol   = hist_df["volume"].values.astype(float)
    n     = len(close)

    if n < MIN_TRADING_DAYS:
        return -999, {"error": f"数据不足 {n}<{MIN_TRADING_DAYS}"}

    # 特征
    X, feat_names = build_features(close, high, low, vol)

    # 目标：未来 5 日收益（原始值）
    target = np.full(n, np.nan)
    target[:-FORECAST] = close[FORECAST:] / close[:-FORECAST] - 1

    # 有效样本
    valid = ~(np.isnan(X).any(axis=1) | np.isnan(target))
    X_v, y_v = X[valid], target[valid]

    if len(X_v) < 150:
        return -999, {"error": f"有效样本不足 {len(X_v)}"}

    # 标准化
    from sklearn.preprocessing import StandardScaler
    X_s = StandardScaler().fit_transform(X_v)

    # 训练
    metrics, ensemble_pred = train_stock(X_s, y_v)
    if not metrics or np.all(np.isnan(ensemble_pred)):
        return -999, {"error": "模型训练失败"}

    # 最新预测
    latest = ensemble_pred[-1] if np.isfinite(ensemble_pred[-1]) else 0.0

    # ── 评分公式 ──
    ens = metrics.get("ensemble", {})
    ens_r2 = ens.get("r2", 0)
    ens_dir = ens.get("dir_acc", 0.5)
    ens_sp = ens.get("spearman", 0)

    # 信号强度（最新预测相对自身历史的 Z-score）
    all_v = ensemble_pred[~np.isnan(ensemble_pred)]
    z = (latest - np.mean(all_v)) / np.std(all_v) if np.std(all_v) > 0 else 0

    # 信号置信度
    # 方向准确率越高、Spearman 越高 → 置信度越高
    conf = np.clip(ens_dir * 2 - 1, 0, 1) * 0.4 + np.clip(ens_sp, 0, 1) * 0.4
    conf += np.clip(ens_r2 + 0.2, 0, 0.3) * 0.2  # R² > -0.2 才有贡献

    # 最终评分: 信号方向(+1/-1) × 信号强度(z) × 置信度 → 映射到 [0,1]
    raw = np.sign(latest) * min(abs(z), 3) / 3 * conf
    score = (raw + 1) / 2  # [-1,1] → [0,1]
    score = np.clip(score, 0, 1)

    details = {
        "ml_score": round(score, 4),
        "pred_return": round(latest, 6),
        "signal_z": round(z, 4),
        "conf": round(conf, 4),
        "dir_acc": round(ens_dir, 4),
        "spearman": round(ens_sp, 4),
        "ensemble_r2": round(ens_r2, 4),
        "model_metrics": {k: v for k, v in metrics.items() if k != "ensemble"},
        "samples": len(y_v),
        "features": X.shape[1],
    }

    return score, details


# ════════════════════════════════════════
# 主流程
# ════════════════════════════════════════
def run_picking(tickers=None, period="2y", compare=False):
    import yfinance as yf
    if tickers is None: tickers = US_WATCHLIST

    print(f"\n{'='*80}")
    print(f"  ML 选股系统 v4 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*80}")
    print(f"  标的: {len(tickers)}  周期: {period}  预测: {FORECAST}日")
    print(f"  模型: CatBoost + XGBoost + RF Ensemble")
    print(f"  评估: Spearman排序 + 方向准确率 + R²")
    if compare:
        print(f"  模式: 对比 ML vs 传统因子")
    print(f"{'='*80}")

    print(f"\n  下载数据中...")
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by='ticker')
    print(f"  完成")

    results = []
    for i, t in enumerate(tickers):
        if (i+1) % 10 == 0:
            print(f"  进度 [{i+1}/{len(tickers)}]")
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if t in data.columns.get_level_values(0):
                    df = data.xs(t, axis=1, level=0).dropna()
                elif t in data.columns.get_level_values(1):
                    df = data.xs(t, axis=1, level=1).dropna()
                else: continue
            else: continue

            if len(df) < MIN_TRADING_DAYS: continue

            hist = pd.DataFrame({
                "close":df["Close"].values.astype(float),
                "high":df["High"].values.astype(float),
                "low":df["Low"].values.astype(float),
                "volume":df["Volume"].values.astype(float),
            })
            price = float(df["Close"].iloc[-1])
            cls = df["Close"].values.astype(float)
            c1 = (cls[-1]/cls[-2]-1)*100 if len(cls)>=2 else 0
            c5 = (cls[-1]/cls[-min(6,len(cls))]-1)*100

            score, details = score_stock(hist)
            if score < -900: continue

            # 传统因子（用于对比）
            trad = None
            if compare:
                from stock_picker_v2 import calc_factors_from_hist, score_stock as trad_score
                facts = calc_factors_from_hist(hist, t)
                ts, _ = trad_score(facts)
                trad = ts

            results.append({
                "ticker":t, "price":price, "score":score,
                "chg_1d_pct":c1, "chg_5d_pct":c5,
                "details":details,
                "trad_score":trad,
            })
        except Exception as e:
            continue

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ════════════════════════════════════════
# 打印结果
# ════════════════════════════════════════
def print_results(results, top_n=20, compare=False):
    top = results[:top_n]

    print(f"\n{'='*120}")
    print(f"  Top {len(top)} ML 选股推荐")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*120}")

    hdr = (f"{'#':>4} {'代码':>6} {'ML评分':>8} {'价格':>10} {'1日':>6} {'5日':>6} "
           f"{'预测收益':>9} {'方向':>6} {'Spear':>7} {'EnsR²':>7}")
    if compare: hdr += f" {'传统分':>8}"
    print(hdr)
    print("-" * 120)

    for i, s in enumerate(top):
        d = s["details"]
        line = (f"{i+1:>4} {s['ticker']:>6} {s['score']:>8.4f} "
                f"${s['price']:>7.2f} "
                f"{s['chg_1d_pct']:>+5.1f}% {s['chg_5d_pct']:>+5.1f}% "
                f"{d.get('pred_return',0):>+8.4f} "
                f"{d.get('dir_acc',0):>5.0%} "
                f"{d.get('spearman',0):>+6.3f} "
                f"{d.get('ensemble_r2',0):>+6.3f}")
        if compare and s.get("trad_score") is not None:
            line += f" {s['trad_score']:>+7.3f}"
        print(line)

    print("-" * 120)

    scores = [s["score"] for s in top]
    dirs = [s["details"].get("dir_acc", 0.5) for s in top]
    sps = [s["details"].get("spearman", 0) for s in top]
    print(f"\n  📊 Top{len(top)} 统计:")
    print(f"     平均 ML 评分:      {np.mean(scores):.4f}")
    print(f"     平均方向准确率:     {np.mean(dirs):.1%}")
    print(f"     平均 Spearman ρ:   {np.mean(sps):+.4f}")
    print(f"     正向方向准确率占比: {sum(1 for d in dirs if d>0.5)/len(dirs):.0%}")
    print(f"     正 Spearman 占比:   {sum(1 for s in sps if s>0)/len(sps):.0%}")

    # 各模型对比
    print(f"\n  📊 各模型 Top{len(top)} 平均:")
    for m in MODELS:
        v_l = []
        for s in top:
            v = s["details"].get("model_metrics", {}).get(m, {})
            if v: v_l.append(v)
        if v_l:
            print(f"     {m.upper():>6}: 方向={np.mean([v['dir_acc'] for v in v_l]):.1%}"
                  f"  Spear={np.mean([v['spearman'] for v in v_l]):+.4f}"
                  f"  R²={np.mean([v['r2'] for v in v_l]):+.4f}")

    # Top 5 分析
    print(f"\n{'='*120}")
    print(f"  Top 5 详细分析")
    print(f"{'='*120}")
    for i, s in enumerate(top[:5]):
        d = s["details"]
        print(f"\n  #{i+1} {s['ticker']} (ML评分: {s['score']:.4f}, ${s['price']:.2f})")
        print(f"    预测收益: {d.get('pred_return',0):+.4f}  Z={d.get('signal_z',0):+.2f}")
        print(f"    方向准确率: {d.get('dir_acc',0):.1%}  Spearman ρ={d.get('spearman',0):+.4f}")
        print(f"    Ensemble R²={d.get('ensemble_r2',0):+.4f}  置信度={d.get('conf',0):.2f}")
        for m in MODELS:
            v = d.get("model_metrics", {}).get(m, {})
            if v:
                print(f"    {m.upper():>6}: R²={v.get('r2',0):+.4f} 方向={v.get('dir_acc',0):.1%} "
                      f"Spear={v.get('spearman',0):+.3f} MAE={v.get('mae',0):.4f}")

    # 对比
    if compare:
        print(f"\n{'='*120}")
        print(f"  ML vs 传统因子 — Top 20 对比")
        print(f"{'='*120}")
        for i, s in enumerate(top[:15]):
            trad = s.get("trad_score")
            if trad is not None:
                print(f"  {s['ticker']:>6}: ML={s['score']:.4f}  传统={trad:+.3f}")
        # 传统视角
        trad_sorted = sorted(results, key=lambda x: x.get("trad_score", -999), reverse=True)
        print(f"\n  传统因子视角 Top 10:")
        for i, s in enumerate(trad_sorted[:10]):
            print(f"  {i+1:>2}. {s['ticker']:>6}: 传统={s.get('trad_score',0):+.3f}  ML={s['score']:.4f}")

    # 建议
    strong = [s for s in top if s["score"] > 0.6]
    watch  = [s for s in top if 0.5 < s["score"] <= 0.6]
    print(f"\n{'='*120}")
    print(f"  策略建议")
    print(f"{'='*120}")
    print(f"\n  🟢 强��荐 (ML评分>0.6): {', '.join(s['ticker'] for s in strong[:5]) or '无'}")
    print(f"  🟡 值得关注 (0.5-0.6): {', '.join(s['ticker'] for s in watch[:8]) or '无'}")
    print(f"\n  💡 选择逻辑:")
    print(f"     - 预测收益: 模型预测的未来{FORECAST}日收益")
    print(f"     - 方向准确率: >55% = 模型方向判断有意义")
    print(f"     - Spearman ρ: >0 = 模型排序能力强（选股相关性）")
    print(f"     - 正 Spearman 和正方向准确率的标的更可靠")

    return top


def save(top, fn="ml_stock_picks_v4"):
    rows = []
    for s in top:
        d = s["details"]
        rows.append({
            "ticker":s["ticker"], "price":s["price"],
            "ml_score":round(s["score"],4),
            "pred_return":round(d.get("pred_return",0),6),
            "dir_acc":round(d.get("dir_acc",0),4),
            "spearman":round(d.get("spearman",0),4),
            "ensemble_r2":round(d.get("ensemble_r2",0),4),
        })
    df = pd.DataFrame(rows)
    p = f"{fn}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {p}")
    return p


# ════════════════════════════════════════
# CLI
# ════════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]
    top_n, tk, comp = TOP_N, None, False

    if "--top" in args:
        idx = args.index("--top")
        if idx+1 < len(args): top_n = int(args[idx+1])
    if "--ticker" in args:
        idx = args.index("--ticker")
        if idx+1 < len(args): tk = [args[idx+1].upper()]
    if "--compare" in args or "--comp" in args:
        comp = True

    rs = run_picking(tk, period="2y", compare=comp)
    if rs:
        top = print_results(rs, top_n=top_n, compare=comp)
        save(top)
        print("\n  ✅ 完成!")
    else:
        print("\n  ❌ 没有有效结果")
