#!/usr/bin/env python3
"""
A股/美股多因子选股系统 v2
基于 AkShare(国内) / yfinance(美股) 数据源
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta
import sys

# ============ 配置 ============
TOP_N = 20
MIN_TRADING_DAYS = 60

# 因子权重 (总计 1.0)
FACTOR_WEIGHTS = {
    # 动量因子 (35%)
    "momentum_1m":  0.08,   # 1月动量
    "momentum_3m":  0.12,   # 3月动量
    "momentum_6m":  0.10,   # 6月动量
    "momentum_12m": 0.05,   # 12月动量 (动量因子全部正向)

    # 质量因子 (25%)
    "sma20_dev":    0.05,   # 价格/20日均线偏离 (正值=强度)
    "volatility":  -0.06,   # 波动率 (负权重, 低波动好)
    "max_dd":      -0.05,   # 最大回撤 (负权重)
    "volume_trend": 0.04,   # 成交量趋势
    "rsi_14":       0.05,   # RSI (适中加分, 极端扣分)

    # 资金/情绪 (20%)
    "price_level": -0.05,   # 价格位置 (低位加分)
    "volume_ratio": 0.04,   # 量比
    "turnover":     0.04,   # 换手率 (适度活跃)
    "high_low_3m":  0.07,   # 3月高低比 (突破形态)

    # 美股专用 (20%)
    "sma50_dev":    0.05,   # 50日均线偏离
    "sma200_dev":   0.05,   # 200日均线偏离
    "bb_width":    -0.03,   # 布林带宽度 (压缩后可能突破)
    "sma20_slope":  0.07,   # 20日均线斜率 (趋势强度)
}


def calc_factors_from_hist(hist_df, ticker):
    """从历史数据计算因子"""
    close = hist_df["close"].values.astype(float)
    high = hist_df["high"].values.astype(float)
    low = hist_df["low"].values.astype(float)
    volume = hist_df["volume"].values.astype(float)

    factors = {}

    # ---- 动量因子 ----
    factors["momentum_1m"]  = (close[-1] / close[-21] - 1) if len(close) >= 21 else 0
    factors["momentum_3m"]  = (close[-1] / close[-63] - 1) if len(close) >= 63 else 0
    factors["momentum_6m"]  = (close[-1] / close[-126] - 1) if len(close) >= 126 else 0
    factors["momentum_12m"] = (close[-1] / close[-252] - 1) if len(close) >= 252 else 0

    # ---- 均线偏离 ----
    sma20 = pd.Series(close).rolling(20).mean().values[-1]
    sma50 = pd.Series(close).rolling(50).mean().values[-1]
    sma200 = pd.Series(close).rolling(200).mean().values[-1]

    factors["sma20_dev"]  = (close[-1] / sma20 - 1) if sma20 > 0 else 0
    factors["sma50_dev"]  = (close[-1] / sma50 - 1) if sma50 > 0 else 0
    factors["sma200_dev"] = (close[-1] / sma200 - 1) if sma200 > 0 else 0

    # ---- 20日均线斜率 ----
    sma20_series = pd.Series(close).rolling(20).mean().dropna().values
    if len(sma20_series) >= 5:
        slope = (sma20_series[-1] - sma20_series[-5]) / sma20_series[-5] * 100
        factors["sma20_slope"] = min(max(slope, -10), 10)
    else:
        factors["sma20_slope"] = 0

    # ---- 波动率 ----
    returns = pd.Series(close).pct_change().dropna().values
    factors["volatility"] = np.std(returns[-63:]) * np.sqrt(252) if len(returns) >= 63 else 0

    # ---- 最大回撤 (3个月) ----
    if len(close) >= 63:
        peak = pd.Series(close[-63:]).cummax().values
        dd = (close[-63:] - peak) / peak
        factors["max_dd"] = np.min(dd)
    else:
        factors["max_dd"] = 0

    # ---- RSI ----
    if len(returns) >= 14:
        gains = np.maximum(returns[-14:], 0)
        losses = -np.minimum(returns[-14:], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            factors["rsi_14"] = 100 - (100 / (1 + rs))
        else:
            factors["rsi_14"] = 100 if avg_gain > 0 else 50
    else:
        factors["rsi_14"] = 50

    # ---- 价格位置 (60日高低) ----
    if len(close) >= 60:
        h60, l60 = np.max(close[-60:]), np.min(close[-60:])
        factors["price_level"] = (close[-1] - l60) / (h60 - l60) if (h60 - l60) > 0 else 0.5
    else:
        factors["price_level"] = 0.5

    # ---- 3月高低比 ----
    if len(high) >= 63 and len(low) >= 63:
        h3m = np.max(high[-63:])
        l3m = np.min(low[-63:])
        factors["high_low_3m"] = (h3m / l3m - 1) if l3m > 0 else 0
    else:
        factors["high_low_3m"] = 0

    # ---- 布林带宽度 ----
    if len(close) >= 20:
        sma20_v = pd.Series(close).rolling(20).mean().values[-1]
        std20 = pd.Series(close).rolling(20).std().values[-1]
        if sma20_v > 0:
            factors["bb_width"] = (2 * std20) / sma20_v  # 布林带宽度/价格
        else:
            factors["bb_width"] = 0
    else:
        factors["bb_width"] = 0

    # ---- 成交量趋势 ----
    vol_sma20 = pd.Series(volume).rolling(20).mean().values[-1] if len(volume) >= 20 else np.mean(volume)
    factors["volume_trend"] = (volume[-1] / vol_sma20 - 1) if vol_sma20 > 0 else 0

    # ---- 量比 (当日成交量/5日均量) ----
    vol_sma5 = pd.Series(volume).rolling(5).mean().values[-1] if len(volume) >= 5 else np.mean(volume)
    factors["volume_ratio"] = (volume[-1] / vol_sma5) if vol_sma5 > 0 else 1

    # ---- 换手率 (成交量/流通股, 用相对值) ----
    factors["turnover"] = volume[-1] / np.mean(volume[-20:]) if len(volume) >= 20 else 1

    return factors


def score_stock(factors):
    """因子加权评分"""
    if factors is None:
        return -999, {}

    score = 0
    details = {}
    weights = FACTOR_WEIGHTS

    for fn, w in weights.items():
        if fn not in factors:
            continue
        raw = factors[fn]
        # 标准化
        if fn.startswith("momentum"):
            clipped = np.clip(raw, -0.30, 0.30)
            norm = clipped / 0.30
        elif fn == "sma20_dev":
            norm = np.clip(raw / 0.10, -1, 1)
        elif fn == "sma50_dev":
            norm = np.clip(raw / 0.15, -1, 1)
        elif fn == "sma200_dev":
            norm = np.clip(raw / 0.20, -1, 1)
        elif fn == "sma20_slope":
            norm = np.clip(raw / 3.0, -1, 1)
        elif fn == "volatility":
            norm = -np.clip(raw / 0.40, -1, 1)
        elif fn == "max_dd":
            norm = -np.clip(np.abs(raw) / 0.25, 0, 1)
        elif fn == "rsi_14":
            norm = (50 - abs(raw - 50)) / 50
        elif fn == "price_level":
            norm = -np.clip((raw - 0.3) / 0.4, -1, 1)
        elif fn == "high_low_3m":
            norm = np.clip(raw / 0.50, -1, 1)
        elif fn == "bb_width":
            norm = -np.clip(raw / 0.15, -1, 1)
        elif fn == "volume_trend":
            norm = np.clip(raw / 2.0, -1, 1)
        elif fn == "volume_ratio":
            norm = np.clip(raw / 3.0, -1, 1)
        elif fn == "turnover":
            norm = np.clip(raw / 3.0, -1, 1)
        else:
            norm = np.clip(raw, -1, 1)

        contrib = norm * w
        score += contrib
        details[fn] = {"raw": raw, "norm": round(norm, 3), "weight": w, "contrib": round(contrib, 3)}

    return score, details


def run_us_stock_picking(tickers, period="1y"):
    """美股多因子选股"""
    import yfinance as yf

    print(f"\n{'='*70}")
    print(f"  多因子选股分析 - {datetime.now().strftime('%Y-%m-%d')}")
    print(f"{'='*70}")
    print(f"  数据源: yfinance (美股)")
    print(f"  分析标的: {len(tickers)} 只")
    print(f"  数据周期: {period}")
    print(f"{'='*70}")

    # 批量下载
    print(f"\n  下载数据中...")
    data = yf.download(tickers, period=period, auto_adjust=True, progress=False, group_by='ticker')
    print(f"  数据下载完成")

    results = []
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[1]:
                # MultiIndex 情况 (多只股票)
                df = data.xs(t, axis=1, level=1).dropna()
            elif hasattr(data, 'columns') and t in data.columns:
                df = data[t].dropna()
            else:
                continue

            if len(df) < MIN_TRADING_DAYS:
                continue

            # 转成统一格式
            hist = pd.DataFrame({
                "close": df["Close"].values.astype(float),
                "high": df["High"].values.astype(float),
                "low": df["Low"].values.astype(float),
                "volume": df["Volume"].values.astype(float),
            })

            factors = calc_factors_from_hist(hist, t)
            score, details = score_stock(factors)

            # 当前价格和涨跌幅
            cur_price = float(df["Close"].iloc[-1])
            
            # 计算近1日和5日涨跌幅
            closes = df["Close"].values.astype(float)
            chg_1d = (closes[-1] / closes[-2] - 1) * 100 if len(closes) >= 2 else 0
            chg_5d = (closes[-1] / closes[-min(6, len(closes))] - 1) * 100

            results.append({
                "ticker": t,
                "price": cur_price,
                "score": score,
                "chg_1d_pct": chg_1d,
                "chg_5d_pct": chg_5d,
                "factors": factors,
                "details": details,
            })
        except Exception as e:
            continue

    # 按评分排序
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


def print_us_results(results, top_n=20, detail=False):
    """打印选股结果"""
    top = results[:top_n]

    print(f"\n{'='*90}")
    print(f"  多因子选股结果 Top {len(top)}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*90}")

    # 表头
    header = f"{'#':>3} {'代码':>6} {'评分':>6} {'价格':>8} {'1日涨跌':>8} {'5日涨跌':>8} {'1月动量':>8} {'3月动量':>8} {'RSI':>5} {'波动':>6} "
    print(header)
    print("-" * 90)

    for i, s in enumerate(top):
        f = s["factors"]
        chg_1d = f"{s['chg_1d_pct']:+.1f}%"
        chg_5d = f"{s['chg_5d_pct']:+.1f}%"
        mom1 = f"{f.get('momentum_1m',0)*100:+.1f}%"
        mom3 = f"{f.get('momentum_3m',0)*100:+.1f}%"
        
        print(f"{i+1:>3} {s['ticker']:>6} {s['score']:>6.2f} ${s['price']:>6.2f} {chg_1d:>8} {chg_5d:>8} {mom1:>8} {mom3:>8} {f.get('rsi_14',50):>5.0f} {f.get('volatility',0)*100:>5.1f}%")

    print("-" * 90)

    # 统计
    avg_score = np.mean([s["score"] for s in top])
    avg_mom3 = np.mean([s["factors"].get("momentum_3m", 0) for s in top])
    avg_mom1 = np.mean([s["factors"].get("momentum_1m", 0) for s in top])
    avg_rsi = np.mean([s["factors"].get("rsi_14", 50) for s in top])
    avg_vol = np.mean([s["factors"].get("volatility", 0) for s in top])
    
    print(f"\n  Top{len(top)} 统计:")
    print(f"     平均评分: {avg_score:.3f}")
    print(f"     平均1月动量: {avg_mom1*100:+.1f}%")
    print(f"     平均3月动量: {avg_mom3*100:+.1f}%")
    print(f"     平均RSI: {avg_rsi:.0f}")
    print(f"     平均波动率(年化): {avg_vol*100:.1f}%")

    # 因子贡献
    if detail:
        print(f"\n{'='*90}")
        print(f"  Top 10 详细因子分析")
        print(f"{'='*90}")
        for i, s in enumerate(top[:10]):
            print(f"\n  #{i+1} {s['ticker']}  (评分: {s['score']:.3f}, 价格: ${s['price']:.2f})")
            print(f"  {'因子名称':<16} {'原始值':<12} {'标准化':<10} {'权重':<8} {'贡献':<10}")
            print(f"  {'-'*56}")
            sorted_details = sorted(s["details"].items(), key=lambda x: abs(x[1]["contrib"]), reverse=True)
            for fn, fd in sorted_details[:10]:
                arrow = "+" if fd["contrib"] > 0 else ""
                print(f"  {fn:<16} {fd['raw']:<12.4f} {fd['norm']:<10.3f} {fd['weight']:<+8.2f} {arrow}{fd['contrib']:.3f}")

    # 推荐逻辑
    print(f"\n{'='*90}")
    print(f"  策略解读")
    print(f"{'='*90}")

    strong_buy = [s for s in top if s["score"] > 0.3]
    watch = [s for s in top if 0.15 < s["score"] <= 0.3]
    
    print(f"\n  \U0001f7e2 强烈关注 (评分>0.3): {', '.join(s['ticker'] for s in strong_buy[:5]) or '无'}")
    print(f"  \U0001f7e1 值得跟踪 (评分0.15-0.3): {', '.join(s['ticker'] for s in watch[:5]) or '无'}")
    
    print(f"\n  \U0001f4a1 因子权重分配:")
    print(f"     动量(35%) + 质量(25%) + 资金情绪(20%) + 趋势(20%)")
    print(f"     说明: 评分越高说明股票在动量、质量和趋势上综合表现越好")
    print(f"     建议: 优先选择评分>0.15的股票, 结合个人风险偏好再筛选")

    return top


def save_us_results(top, filename="stock_picks_us"):
    """保存结果"""
    rows = []
    for s in top:
        row = {"ticker": s["ticker"], "price": s["price"], "score": round(s["score"], 3),
               "chg_1d_pct": round(s["chg_1d_pct"], 2), "chg_5d_pct": round(s["chg_5d_pct"], 2)}
        row.update({k: round(v, 4) for k, v in s["factors"].items()})
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = f"{filename}_{datetime.now().strftime('%Y%m%d')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {csv_path}")
    return csv_path


# 美股关注列表 (大市值 + 热门板块)
US_WATCHLIST = [
    # 科技巨头
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    # 半导体
    "AMD", "INTC", "QCOM", "TXN", "MU", "ASML", "ARM",
    # 金融
    "JPM", "BAC", "GS", "MS", "V", "MA", "BLK",
    # 消费
    "WMT", "COST", "HD", "LOW", "PG", "KO", "PEP", "MCD", "SBUX", "NKE",
    # 医疗
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO",
    # 能源/工业
    "XOM", "CVX", "CAT", "GE", "BA", "HON",
    # ETF 核心
    "SPY", "QQQ", "IWM", "DIA",
    # 板块 ETF
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLU", "XLRE", "XLC",
    # 热门
    "PLTR", "SOFI", "RDDT", "HOOD", "MSTR", "COIN", "TSM",
]

if __name__ == "__main__":
    results = run_us_stock_picking(US_WATCHLIST, period="1y")
    
    if results:
        top = print_us_results(results, top_n=20, detail="--detail" in sys.argv)
        save_us_results(top)
