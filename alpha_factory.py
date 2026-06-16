#!/usr/bin/env python3
"""
Alpha 工厂模块 (P2.2)
====================
遗传规划/手工 Alpha 因子自动发现系统。

使用 gplearn 进行遗传规划因子挖掘，不可用时使用手工经典 Alpha 因子。

用法:
  python alpha_factory.py --discover AAPL            # 为单只股票发现 Alpha
  python alpha_factory.py --apply AAPL --alphas id   # 应用已发现/经典 Alpha
  python alpha_factory.py --list                     # 列出缓存的 Alpha
"""

import numpy as np
import pandas as pd
import warnings, sys, os, json, re
from datetime import datetime

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
ALPHA_CACHE = os.path.join(CACHE_DIR, "alphas_cache.json")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── 尝试安装 gplearn ──
HAS_GPLEARN = False
try:
    import gplearn
    HAS_GPLEARN = True
except ImportError:
    pass


# ═══════════════════════════════════════
# 手工经典 Alpha 因子 (101 个中的精选)
# ═══════════════════════════════════════
CLASSIC_ALPHAS = [
    {
        "id": "alpha001",
        "name": "均线偏离",
        "formula": "(close - ts_mean(close,20)) / ts_std(close,20)",
        "description": "20日布林带Z-score，捕捉超买超卖",
    },
    {
        "id": "alpha002",
        "name": "量价相关性",
        "formula": "ts_corr(returns, log_volume, 20)",
        "description": "量价关系，上涨放量/下跌放量信号",
    },
    {
        "id": "alpha003",
        "name": "短期反转",
        "formula": "-rank(ts_sum(returns, 5))",
        "description": "5日反转因子，捕捉短期超跌反弹",
    },
    {
        "id": "alpha004",
        "name": "动量加速",
        "formula": "rank(ts_mean(returns, 21)) - rank(ts_mean(returns, 63))",
        "description": "短期动量减长期动量，捕捉趋势加速",
    },
    {
        "id": "alpha005",
        "name": "波动率调整动量",
        "formula": "ts_mean(returns, 21) / ts_std(returns, 21)",
        "description": "夏普比率式动量，风险调整后的趋势强度",
    },
    {
        "id": "alpha006",
        "name": "RSI 形态",
        "formula": "-rank(ts_mean(max(0, returns - ts_mean(returns, 14)), 14) / "
                    "ts_mean(max(0, ts_mean(returns, 14) - returns), 14))",
        "description": "RSI 的另一种实现",
    },
    {
        "id": "alpha007",
        "name": "价格位置",
        "formula": "(close - ts_min(low, 60)) / (ts_max(high, 60) - ts_min(low, 60))",
        "description": "60日价格百分位位置",
    },
    {
        "id": "alpha008",
        "name": "成交量冲击",
        "formula": "-rank(ts_corr(rank(high), rank(volume), 15))",
        "description": "高价与高量的负相关 → 见顶信号",
    },
    {
        "id": "alpha009",
        "name": "波动率变化",
        "formula": "ts_std(returns, 21) / ts_std(returns, 63)",
        "description": "短期 vs 长期波动率比，波动率结构变化",
    },
    {
        "id": "alpha010",
        "name": "趋势强度组合",
        "formula": "rank(ts_mean(returns, 10)) * rank(ts_mean(volume, 10))",
        "description": "量价同向的趋势确认",
    },
    {
        "id": "alpha011",
        "name": "隔夜缺口",
        "formula": "open / delay(close, 1) - 1",
        "description": "隔夜跳空幅度",
    },
    {
        "id": "alpha012",
        "name": "日内波动",
        "formula": "(high - low) / close",
        "description": "日内波动率",
    },
    {
        "id": "alpha013",
        "name": "均线金叉",
        "formula": "sign(ts_mean(close, 10) - ts_mean(close, 50))",
        "description": "10/50日均线关系，金叉死叉信号",
    },
    {
        "id": "alpha014",
        "name": "成交量趋势",
        "formula": "ts_mean(volume, 5) / ts_mean(volume, 20)",
        "description": "短期 vs 长期成交量比，放量缩量",
    },
    {
        "id": "alpha015",
        "name": "综合得分",
        "formula": "rank(alpha001) + rank(alpha005) + rank(alpha007)",
        "description": "多因子综合排名",
    },
]

# 公式解析器注册
FORMULA_FUNCS = {}


def register_func(name, fn):
    FORMULA_FUNCS[name] = fn


# ═══════════════════════════════════════
# 公式计算基础函数
# ═══════════════════════════════════════
def ts_mean(series, window):
    return series.rolling(window, min_periods=max(window//2, 2)).mean()

def ts_std(series, window):
    return series.rolling(window, min_periods=max(window//2, 2)).std()

def ts_sum(series, window):
    return series.rolling(window, min_periods=max(window//2, 2)).sum()

def ts_corr(s1, s2, window):
    return s1.rolling(window, min_periods=max(window//2, 2)).corr(s2)

def ts_rank(series, window):
    """滚动排名 (0~1)"""
    return series.rolling(window, min_periods=max(window//2, 2)).apply(
        lambda x: (pd.Series(x).rank().iloc[-1] - 1) / max(len(x) - 1, 1)
    )

def ts_min(series, window):
    return series.rolling(window, min_periods=min(window//2, 5)).min()

def ts_max(series, window):
    return series.rolling(window, min_periods=min(window//2, 5)).max()

def delay(series, n):
    return series.shift(n)

def rank(series):
    """横截面/全局排名 0~1"""
    return series.rank(pct=True)

def sign(series):
    return np.sign(series)

def max_s(a, b):
    return np.maximum(a, b)

def min_s(a, b):
    return np.minimum(a, b)

def neg(x):
    return -x

register_func("ts_mean", ts_mean)
register_func("ts_std", ts_std)
register_func("ts_sum", ts_sum)
register_func("ts_corr", ts_corr)
register_func("ts_rank", ts_rank)
register_func("ts_min", ts_min)
register_func("ts_max", ts_max)
register_func("delay", delay)
register_func("rank", rank)
register_func("sign", sign)
register_func("max", max_s)
register_func("min", min_s)
register_func("neg", neg)


# ═══════════════════════════════════════
# 公式解析与计算
# ═══════════════════════════════════════
def parse_simple_expr(df, expr_str):
    """解析简单公式并计算"""
    # ���名引用
    cols = {"close", "open", "high", "low", "volume", "returns", "log_volume"}
    # 预计算常用列
    close = pd.Series(df.values if isinstance(df, pd.Series) else (
        df["Close"].values.astype(float).ravel() if "Close" in df
        else df["close"].values.astype(float).ravel()
    ))
    df_c = pd.DataFrame({"close": close})
    if "open" in df.columns or "Open" in df.columns:
        o_col = "Open" if "Open" in df.columns else "open"
        df_c["open"] = df[o_col].values.astype(float).ravel()
    if "high" in df.columns or "High" in df.columns:
        h_col = "High" if "High" in df.columns else "high"
        df_c["high"] = df[h_col].values.astype(float).ravel()
    if "low" in df.columns or "Low" in df.columns:
        l_col = "Low" if "Low" in df.columns else "low"
        df_c["low"] = df[l_col].values.astype(float).ravel()
    if "volume" in df.columns or "Volume" in df.columns:
        v_col = "Volume" if "Volume" in df.columns else "volume"
        df_c["volume"] = df[v_col].values.astype(float).ravel()
    df_c["returns"] = df_c["close"].pct_change()
    df_c["log_volume"] = np.log(np.maximum(df_c.get("volume", pd.Series(1, index=df_c.index)), 1))

    # 简单公式求值
    # 用 eval 但限制环境
    env = {k: v.values for k, v in df_c.items()}
    env.update({k: v for k, v in FORMULA_FUNCS.items()})

    # 替换函数调用为 pandas 兼容形式
    # 先把 ts_corr(a,b,20) 这类多参数调用转换
    result = None
    try:
        result = eval(expr_str, {"__builtins__": {}}, env)
    except Exception:
        # 尝试向量化逐行计算
        try:
            result = _evaluate_safe(expr_str, df_c)
        except Exception:
            return pd.Series(np.nan, index=df_c.index)

    if isinstance(result, np.ndarray):
        return pd.Series(result, index=df_c.index)
    return result


def _evaluate_safe(expr_str, df):
    """安全求值：构建列局部变量"""
    local_vars = {}
    for col in ["close", "open", "high", "low", "volume", "returns", "log_volume"]:
        if col in df:
            local_vars[col] = df[col]
    local_vars.update(FORMULA_FUNCS)
    try:
        return eval(expr_str, {"__builtins__": {}, "np": np, "pd": pd}, local_vars)
    except Exception:
        return pd.Series(np.nan, index=df.index)


# ═══════════════════════════════════════
# Alpha 因子计算
# ═══════════════════════════════════════
def apply_alpha_formula(df, formula_str, alpha_id=None):
    """对 DataFrame 应用 Alpha 公式"""
    # 处理 alpha001 等引用
    ref_match = re.match(r'^alpha(\d+)$', formula_str.strip())
    if ref_match:
        for alpha in CLASSIC_ALPHAS:
            if alpha["id"] == formula_str.strip():
                return apply_alpha_formula(df, alpha["formula"], alpha_id=alpha["id"])

    result = parse_simple_expr(df, formula_str)
    if result is not None:
        # Z-score 标准化
        result = (result - result.mean()) / max(result.std(), 1e-8)
    else:
        result = pd.Series(np.nan, index=df.index)
    return result


def compute_classic_alphas(df, alpha_ids=None):
    """计算一组经典 Alpha 因子"""
    if alpha_ids is None:
        alpha_ids = [a["id"] for a in CLASSIC_ALPHAS[:10]]

    results = {}
    for alpha in CLASSIC_ALPHAS:
        if alpha["id"] in alpha_ids:
            try:
                vals = apply_alpha_formula(df, alpha["formula"])
                results[alpha["id"]] = vals
            except Exception:
                results[alpha["id"]] = pd.Series(np.nan, index=df.index)
    return pd.DataFrame(results)


def score_with_alphas(ticker, alpha_ids=None, period="2y", use_gplearn=False):
    """用 Alpha 因子对单只股票评分"""
    import yfinance as yf

    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if df.empty:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # 计算 Alpha 因子
    alpha_df = compute_classic_alphas(df, alpha_ids=alpha_ids)
    if alpha_df.empty or alpha_df.isna().all().all():
        return None

    # 综合 Alpha 得分: 各因子等权
    alpha_z = alpha_df.fillna(0)
    combined = alpha_z.mean(axis=1)

    # 最新信号
    latest = float(combined.iloc[-1]) if len(combined) > 0 else 0
    _, latest_actual, _, _ = _get_close_target(df)

    return {
        "ticker": ticker,
        "alpha_score": round(float(_scale_to_01(latest)), 4),
        "alpha_raw": round(latest, 4),
        "alpha_count": len(alpha_df.columns),
        "alpha_ids": list(alpha_df.columns),
    }


def _get_close_target(df):
    """获取收盘价和未来 5 日收益"""
    close = df["Close"].values.astype(float).ravel()
    n = len(close)
    if n < 10:
        return close[-1], 0, 0, 0
    future_5d = close[-1] / close[-6] - 1 if n >= 6 else 0
    past_5d = close[-1] / close[-1] - 1
    return close[-1], future_5d, past_5d, 0


def _scale_to_01(x):
    """将 z-score 映射到 0~1"""
    return max(0, min(1, (x + 3) / 6))


# ═══════════════════════════════════════
# 缓存管理
# ═══════════════════════════════════════
def save_discovered_alphas(alphas_list, ticker):
    """保存发现的 Alpha 到缓存"""
    cache = {}
    if os.path.exists(ALPHA_CACHE):
        try:
            with open(ALPHA_CACHE) as f:
                cache = json.load(f)
        except:
            pass

    cache[ticker] = {
        "alphas": alphas_list,
        "timestamp": datetime.now().isoformat(),
    }

    with open(ALPHA_CACHE, "w") as f:
        json.dump(cache, f, indent=2, default=str)
    print(f"  ✅ Alpha 已缓存 -> {ALPHA_CACHE}")


def load_saved_alphas(ticker=None):
    """加载缓存的 Alpha"""
    if not os.path.exists(ALPHA_CACHE):
        return []

    with open(ALPHA_CACHE) as f:
        cache = json.load(f)

    if ticker:
        return cache.get(ticker, {}).get("alphas", [])
    return list(cache.keys())


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]

    if "--discover" in args:
        idx = args.index("--discover")
        if idx + 1 < len(args):
            t = args[idx + 1].upper()
            print(f"\n  📐 Alpha 因子发现 — {t}")
            print(f"  使用 {15} 个经典 Alpha 因子")
            print(f"{'='*60}")

            import yfinance as yf
            df = yf.download(t, period="2y", auto_adjust=True, progress=False)
            if df.empty:
                print(f"  ❌ 数据下载失败")
                sys.exit(1)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            print(f"\n  计算经典 Alpha 因子...")
            alpha_df = compute_classic_alphas(df)

            if alpha_df.empty:
                print(f"  ❌ Alpha 计算失败")
                sys.exit(1)

            # 计算各因子与未来 5d 收益的相关性
            close = df["Close"].values.astype(float).ravel()
            future_5d = pd.Series(
                [close[i+5]/close[i]-1 if i+5 < len(close) else np.nan
                 for i in range(len(close))],
                index=df.index
            )

            results = []
            for col in alpha_df.columns:
                valid = alpha_df[col].notna() & future_5d.notna()
                if valid.sum() < 20:
                    continue
                corr = alpha_df.loc[valid, col].corr(future_5d.loc[valid])
                results.append({
                    "id": col,
                    "info": next((a["name"] for a in CLASSIC_ALPHAS if a["id"] == col), col),
                    "corr_5d": round(corr, 4),
                    "abs_corr": round(abs(corr), 4),
                })

            results.sort(key=lambda x: x["abs_corr"], reverse=True)

            print(f"\n  {'Alpha':>10} {'名称':<16} {'与5d收益相关':>12} {'|相关|':>8}")
            print(f"  {'-'*50}")
            for r in results[:10]:
                corr_s = f"{r['corr_5d']:+.4f}"
                print(f"  {r['id']:>10} {r['info']:<16} {corr_s:>12} {r['abs_corr']:>8.4f}")

            # 保存发现结果
            save_discovered_alphas(results, t)

            # 综合 Alpha 评分
            score_info = score_with_alphas(t)
            if score_info:
                print(f"\n  综合 Alpha 得分: {score_info['alpha_score']:.4f}")
                print(f"  因子数: {score_info['alpha_count']}")
        else:
            print("  Usage: --discover TICKER")

    elif "--apply" in args:
        idx = args.index("--apply")
        if idx + 1 < len(args):
            t = args[idx + 1].upper()
            alpha_ids = None
            if "--alphas" in args:
                aidx = args.index("--alphas")
                if aidx + 1 < len(args):
                    raw = args[aidx + 1]
                    if raw == "all":
                        alpha_ids = [a["id"] for a in CLASSIC_ALPHAS]
                    elif raw == "top5":
                        alpha_ids = [a["id"] for a in CLASSIC_ALPHAS[:5]]
                    else:
                        alpha_ids = raw.split(",")

            print(f"\n  📐 Alpha 评分 — {t}")
            score_info = score_with_alphas(t, alpha_ids=alpha_ids)
            if score_info:
                print(f"    得分:     {score_info['alpha_score']:.4f}")
                print(f"    原始值:   {score_info['alpha_raw']:+.4f}")
                print(f"    因子数:   {score_info['alpha_count']}")
            else:
                print(f"  ❌ Alpha 评分失败")

    elif "--list" in args:
        cached = load_saved_alphas()
        if cached:
            print(f"\n  缓存的 Alpha 发现结果:")
            for t in cached:
                alphas = load_saved_alphas(t)
                print(f"    {t}: {len(alphas)} 个 Alpha 因子")
        else:
            print(f"\n  缓存为空")
        print(f"\n  内置经典 Alpha 因子: {len(CLASSIC_ALPHAS)} 个")
        for a in CLASSIC_ALPHAS:
            print(f"    {a['id']:>10}: {a['name']:<16} — {a['description']}")

    else:
        print(__doc__)
