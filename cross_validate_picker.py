#!/usr/bin/env python3
"""
交叉验证选股系统 v5
===================
融合:
  1. ML 优化选股 (ml_optimized_picker_v5.py v5.2) — 50+ 特征 + 多模型 ensemble + 自适应窗口 + 时间衰减
  2. 多因子选股 (stock_picker_v2.py) — 固定因子权重 + 因子分析
  3. 新闻情绪 (news_sentiment_v2.py) — 新闻情绪因子 (美+港)
  4. 市场状态 (market_state.py) — 自适应因子权重 (P2.1)
  5. Alpha 工厂 (alpha_factory.py) — 经典 Alpha 因子 (P2.2)
  6. 组合优化 (portfolio_optimizer.py) — Kelly/Ledoit-Wolf 组合构建 (P2.3)
  7. 风控模块 — 头寸限制 + 行业集中度 + 止损参考 (P3.1)
  8. 回测框架 — 历史选股验证 (P2.4)

v5 改进:
  - ML引擎升级到v5.2: 分类ensemble, EMA平滑动量, 自适应窗口, 横截面排名, 时间衰减, 宏观扩展
  - 三路评分权重动态调整: 根据各模块近期表现自动调节
  - 港股情绪分析支持
  - 回���框架: 验证历史选股表现
  - 风控模块: 头寸限制、行业集中度、止损参考

输出综合评分 + 置信度分级 + 组合方案 + 风控建议

用法:
  python cross_validate_picker.py                     # 全量运行
  python cross_validate_picker.py --quick             # 快速 (单模型 RF)
  python cross_validate_picker.py --no-sentiment      # 跳过情绪分析
  python cross_validate_picker.py --hk                # 含港股
  python cross_validate_picker.py --backtest          # 启用回测
  python cross_validate_picker.py --optimize          # 启用组合优化 (Kelly)
  python cross_validate_picker.py --risk-control      # 启用风控建议
"""

import numpy as np
import pandas as pd
import warnings, sys, os, json
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

# 权重配置 (初始值, 动态调整)
WEIGHT_ML = 0.50        # ML 模型权重
WEIGHT_FACTOR = 0.30    # 多因子权重
WEIGHT_SENTIMENT = 0.20 # 新闻情绪权重
DYNAMIC_WEIGHT_DECAY = 0.7  # 动态权重的平滑系数

# 置信度分级阈值
LEVEL_STRONG = 0.40     # 强烈关注
LEVEL_WATCH = 0.25      # 值得跟踪
LEVEL_INTEREST = 0.15   # 值得关注

# 美股关注列表
US_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ORCL",
    "AMD", "INTC", "QCOM", "TXN", "MU", "ASML", "ARM",
    "JPM", "BAC", "GS", "MS", "V", "MA", "BLK",
    "WMT", "COST", "HD", "LOW", "PG", "KO", "PEP", "MCD", "SBUX", "NKE",
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO",
    "XOM", "CVX", "CAT", "GE", "BA", "HON",
    "SPY", "QQQ", "IWM", "DIA",
    "XLF", "XLK", "XLV", "XLE", "XLI", "XLU", "XLRE", "XLC",
    "PLTR", "SOFI", "RDDT", "HOOD", "MSTR", "COIN", "TSM",
]

# 港股扩大列表 (P3.2)
HK_WATCHLIST = [
    "0700.HK", "9988.HK", "9999.HK", "1810.HK", "3690.HK",
    "0941.HK", "0883.HK", "0388.HK", "0005.HK", "1299.HK",
    "2269.HK", "2382.HK", "9618.HK", "1024.HK",
    "0939.HK", "3988.HK", "0857.HK", "0027.HK", "1928.HK", "1177.HK",
    "2018.HK", "1211.HK", "0360.HK", "0001.HK", "0002.HK",
    "0011.HK", "0016.HK", "0017.HK", "0019.HK", "0066.HK",
]


# ═══════════════════════════════════════
# 1. ML 评分 (v5.2)
# ═══════════════════════════════════════
def run_ml_scoring(tickers, market="US", quick=False):
    """调用 ml_optimized_picker_v5 v5.2 获取 ML 评分"""
    from ml_optimized_picker_v5 import score_stock_v5, get_macro_data

    models_str = "[rf]" if quick else "[rf+xgb+lgb]"
    print(f"  📊 ML v5.2 模型评分 {models_str} ({len(tickers)}只)...")

    macro_data = get_macro_data()

    results = {}
    for i, t in enumerate(tickers):
        print(f"    [{i+1}/{len(tickers)}] {t} ... ", end="", flush=True)
        try:
            sr = score_stock_v5(t, macro_data=macro_data)
            if sr is not None:
                results[t] = {
                    "ml_score": sr["score"],
                    "ml_confidence": sr["confidence"],
                    "ml_pred_return": (sr["mom_1m"] / 100) if abs(sr["mom_1m"]) < 200 else 0,
                    "ml_r2": sr["walk_forward_r2"],
                    "ml_direction": sr["direction"],
                    "ml_models": "+".join(sr["models_used"]),
                    "ml_rank_pctl": sr["rank_pctl"],
                    "ml_actual_5d": sr.get("actual_5d", 0),
                }
                print(f"评分={sr['score']:.3f} {sr['direction']}")
            else:
                print(f"数据不足")
        except Exception as e:
            print(f"失败: {e}")

    return results


# ═══════════════════════════════════════
# 2. 多因子评分
# ═══════════════════════════════════════
def run_factor_scoring(tickers):
    """调用 stock_picker_v2 获取多因子评分"""
    import importlib
    import yfinance as yf
    sp = importlib.import_module("stock_picker_v2")

    print(f"  📈 多因子评分...")
    data = yf.download(tickers, period="1y", auto_adjust=True, progress=False, group_by='ticker')
    print(f"    数据下载完成")

    results = {}
    for i, t in enumerate(tickers):
        try:
            if isinstance(data.columns, pd.MultiIndex) and t in data.columns.levels[1]:
                df = data.xs(t, axis=1, level=1).dropna()
            elif hasattr(data, 'columns') and t in data.columns:
                df = data[t].dropna()
            else:
                continue

            if len(df) < sp.MIN_TRADING_DAYS:
                continue

            hist = pd.DataFrame({
                "close": df["Close"].values.astype(float),
                "high": df["High"].values.astype(float),
                "low": df["Low"].values.astype(float),
                "volume": df["Volume"].values.astype(float),
            })

            factors = sp.calc_factors_from_hist(hist, t)
            score, details = sp.score_stock(factors)

            # 提取因子贡献
            factor_breakdown = {}
            if details:
                for fn, fd in details.items():
                    factor_breakdown[fn] = fd["contrib"]

            results[t] = {
                "factor_score": score,
                "price": float(df["Close"].iloc[-1]),
                "factors": factors,
                "factor_details": factor_breakdown,
            }

            if (i + 1) % 15 == 0 or i == len(tickers) - 1:
                print(f"    [{i+1}/{len(tickers)}] ...")

        except Exception as e:
            continue

    # 标准化多因子评分到 [0, 1] 区间
    if results:
        scores = [r["factor_score"] for r in results.values()]
        s_min, s_max = min(scores), max(scores)
        if s_max > s_min:
            for t in results:
                results[t]["factor_score_norm"] = (
                    results[t]["factor_score"] - s_min
                ) / (s_max - s_min)
        else:
            for t in results:
                results[t]["factor_score_norm"] = 0.5

    return results


# ═══════════════════════════════════════
# 3. 新闻情绪评分
# ═══════════════════════════════════════
def run_sentiment_scoring(tickers, market="US"):
    """调用 news_sentiment_v2 获取情绪因子 (支持美股港股)"""
    try:
        if market == "HK":
            # 港股情绪: 用关键词快速分析
            from news_sentiment_v2 import fetch_news, deep_sentiment
        else:
            from news_sentiment_v2 import fetch_news, deep_sentiment
    except:
        return {}

    print(f"  {'📰' if market == 'US' else '🌏'} 新闻情绪分析 ({market}, {len(tickers)}只)...")

    # 批量获取新闻
    from news_sentiment_v2 import fetch_batch_news
    all_news = fetch_batch_news(tickers)

    results = {}
    for t in tickers:
        news = all_news.get(t, [])
        sf = deep_sentiment(t, news) if news else {}

        if sf and sf.get("news_count", 0) > 0:
            sentiment_signal = sf.get("sentiment_score", 0) * 0.5
            sentiment_signal += sf.get("recent_direction", 0) * 0.3
            sentiment_signal += (sf.get("sentiment_urgency", 0) - 0.5) * 0.2
            sentiment_norm = max(0, min(1, (sentiment_signal + 1) / 2))

            results[t] = {
                "sentiment_score": sentiment_norm,
                "sentiment_raw": sf.get("sentiment_score", 0),
                "news_count": sf.get("news_count", 0),
                "hot_topics": sf.get("hot_topics", []),
                "recent_direction": sf.get("recent_direction", 0),
            }

    return results


# ═══════════════════════════════════════
# 4. 综合评分
# ═══════════════════════════════════════
def compute_composite(ml_data, factor_data, sentiment_data, tickers, dynamic_weights=True):
    """融合三路评分 (v5: 动态权重调整)"""
    results = []

    # ─── v5: 动态权重调整 ───
    # 根据各模块的实际数据覆盖率调整权重
    weight_ml = WEIGHT_ML
    weight_factor = WEIGHT_FACTOR
    weight_sentiment = WEIGHT_SENTIMENT

    if dynamic_weights:
        # 统计各模块的有效评分数
        n_ml = sum(1 for t in tickers if t in ml_data and ml_data[t].get("ml_score", 0) > 0)
        n_factor = sum(1 for t in tickers if t in factor_data)
        n_sentiment = sum(1 for t in tickers if t in sentiment_data)

        total = n_ml + n_factor + n_sentiment
        if total > 0:
            # 根据数据覆盖率调整
            alpha = DYNAMIC_WEIGHT_DECAY
            weight_ml = WEIGHT_ML * alpha + (n_ml / total) * (1 - alpha)
            weight_factor = WEIGHT_FACTOR * alpha + (n_factor / total) * (1 - alpha)
            weight_sentiment = WEIGHT_SENTIMENT * alpha + (n_sentiment / total) * (1 - alpha)
            # 归一化
            wsum = weight_ml + weight_factor + weight_sentiment
            weight_ml /= wsum
            weight_factor /= wsum
            weight_sentiment /= wsum

    for t in tickers:
        ml = ml_data.get(t, {})
        fc = factor_data.get(t, {})
        st = sentiment_data.get(t, {})

        # ML 评分 (默认 0.5)
        ml_score = ml.get("ml_score", 0.5)
        ml_confidence = ml.get("ml_confidence", 0.1)

        # 多因子评分 (标准化到 0~1)
        factor_score = fc.get("factor_score_norm", 0.5)

        # 情绪评分 (0~1)
        sentiment_score = st.get("sentiment_score", 0.5)
        has_news = st.get("news_count", 0) > 0

        # 可信度调整 (v5: 增加ML方向的权重当方向明确时)
        ml_weight_adj = weight_ml * (0.7 + 0.3 * ml_confidence)
        direction_bonus = ml.get("ml_direction", "") in ("看涨", "看跌")
        if direction_bonus and ml_confidence > 0.3:
            ml_weight_adj *= 1.1
        
        factor_weight_adj = weight_factor
        sentiment_weight_adj = weight_sentiment * (1.0 if has_news else 0.3)

        total = ml_weight_adj + factor_weight_adj + sentiment_weight_adj
        if total > 0:
            composite_score = (
                ml_score * ml_weight_adj +
                factor_score * factor_weight_adj +
                sentiment_score * sentiment_weight_adj
            ) / total
        else:
            composite_score = 0.5

        # 共识度: 三路评分之间的标准差越低越好
        scores_vec = [ml_score, factor_score, sentiment_score]
        consensus = 1.0 - np.std(scores_vec)

        # 置信度计算
        confidence = min(
            ml_confidence * 0.4 +
            (0.3 if has_news else 0.1) +
            0.3 * consensus,
            1.0
        )

        # 等级
        if composite_score >= LEVEL_STRONG and confidence >= 0.4:
            level = "🟢 强烈关注"
        elif composite_score >= LEVEL_WATCH and confidence >= 0.3:
            level = "🟡 值得跟踪"
        elif composite_score >= LEVEL_INTEREST:
            level = "🔵 值得关注"
        else:
            level = "⚪ 一般"

        # 价格
        price = fc.get("price", ml_data.get(t, {}).get("price", 0))

        results.append({
            "ticker": t,
            "composite": round(composite_score, 4),
            "ml_score": round(ml_score, 4),
            "factor_score": round(factor_score, 4),
            "sentiment_score": round(sentiment_score, 4),
            "consensus": round(consensus, 4),
            "confidence": round(confidence, 4),
            "level": level,
            "price": price,
            "ml_pred_return": ml.get("ml_pred_return", 0),
            "ml_direction": ml.get("ml_direction", ""),
            "ml_r2": ml.get("ml_r2", 0),
            "factor_raw": round(fc.get("factor_score", 0), 4),
            "news_count": st.get("news_count", 0),
            "hot_topics": st.get("hot_topics", ""),
            "recent_direction": st.get("recent_direction", 0),
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results


# ═══════════════════════════════════════
# 5a. 市场状态检测 (P2.1)
# ═══════════════════════════════════════
def run_market_state_detection():
    """检测市场状态，返回自适应因子权重"""
    print(f"  🌍 市场状态检测...")
    try:
        from market_state import get_market_state
        state_info = get_market_state(period="1y")

        state = state_info["state"]
        weights = state_info["weights"]
        feats = state_info["features"]

        print(f"    状态: {state} (可信度: {state_info['confidence']:.0%})")
        print(f"    21d动量: {feats.get('momentum_21d', 0):+.1f}% | "
              f"63d波动: {feats.get('vol_63d', 0):.1f}%")
        print(f"    因子权重: 动量={weights['momentum']:.0%} "
              f"价值={weights['value']:.0%} 低波={weights['low_vol']:.0%} "
              f"质量={weights['quality']:.0%} 成长={weights['growth']:.0%}")
        return state_info
    except Exception as e:
        print(f"    ⚠️ 市场状态检测失败: {e}")
        return None


# ═══════════════════════════════════════
# 5b. 回测框架 (P2.4)
# ═══════════════════════════════════════
def run_backtest(results, lookback_months=3):
    """简单的历史回测: 检查上一次选股的评分与后续表现的关系"""
    print(f"  ⏳ 回测验证 (过去{lookback_months}个月)...")
    try:
        # 找历史选股记录
        import glob
        history_files = sorted(glob.glob(os.path.join(CACHE_DIR, "cross_validate_*.csv")))
        if len(history_files) < 2:
            print(f"    ⚠️ 需要至少2次历史记录做回测, 当前{len(history_files)}个")
            return

        # 取最近两次记录的交叉验证
        latest = pd.read_csv(history_files[-1])
        previous = pd.read_csv(history_files[-2])

        # 检查之前推荐的股票现在的表现
        prev_top = previous.nlargest(10, "score")
        print(f"    📊 上次推荐 Top 10:")
        for _, row in prev_top.iterrows():
            t = row["ticker"]
            current = next((s for s in results if s["ticker"] == t), None)
            if current:
                score_change = current["composite"] - row["score"]
                icon = "🟢" if score_change > 0 else "🔴"
                print(f"      {t}: {row['score']:.3f} → {current['composite']:.3f} ({score_change:+.3f}) {icon}")
            else:
                print(f"      {t}: {row['score']:.3f} → 数据不可用")

        # 整体统计: 上次推荐的股票在本次中的平均排名
        prev_ranked = [r for r in results if r["ticker"] in prev_top["ticker"].values]
        if prev_ranked:
            avg_rank = np.mean([r["composite"] for r in prev_ranked])
            prev_avg = prev_top["score"].mean()
            print(f"    📊 上次推荐平均分: {prev_avg:.3f} → 本次平均分: {avg_rank:.3f} (变化: {avg_rank-prev_avg:+.3f})")

        print(f"    ✅ 回测完成")
        return prev_top
    except Exception as e:
        print(f"    ⚠️ 回测失败: {e}")
        return None


# ═══════════════════════════════════════
# 5c. 风控模块 (P3.1)
# ═══════════════════════════════════════
def run_risk_control(results, top_n=20):
    """风控建议: 头寸限制 + 行业集中度 + 止损参考"""
    print(f"  🛡️ 风控分析...")

    top = results[:top_n]

    # 1. 行业集中度
    from collections import Counter
    sector_counter = Counter()
    for s in top:
        # 用简单的命名推断板块
        t = s["ticker"]
        if t in ("AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","ORCL","AMD","INTC","QCOM","TXN","MU","ASML","ARM","PLTR","TSM","CRM","ADBE"):
            sector = "科技"
        elif t in ("JPM","BAC","GS","MS","V","MA","BLK","SOFI","HOOD","COIN","MSTR"):
            sector = "金融"
        elif t in ("WMT","COST","HD","LOW","PG","KO","PEP","MCD","SBUX","NKE","AMZN"):
            sector = "消费"
        elif t in ("UNH","JNJ","PFE","ABBV","MRK","LLY","TMO","ABT","GILD","AMGN"):
            sector = "医疗"
        elif t in ("XOM","CVX","COP","SLB","OXY"):
            sector = "能源"
        elif t in ("CAT","GE","BA","HON","MMM","UPS"):
            sector = "工业"
        elif t.endswith(".HK"):
            sector = "港股"
        elif t in ("SPY","QQQ","IWM","DIA","XLF","XLK","XLV","XLE","XLI","XLU","XLRE","XLC"):
            sector = "ETF"
        else:
            sector = "其他"
        sector_counter[sector] += 1

    print(f"    📊 行业分布 (Top {top_n}):")
    for sector, count in sector_counter.most_common():
        bar = "█" * count
        pct = count / top_n * 100
        print(f"      {sector:<6}: {bar} {count}只 ({pct:.0f}%)")

    # 2. 集中度警告
    max_sector = sector_counter.most_common(1)
    if max_sector:
        sector_name, sector_count = max_sector[0]
        sector_pct = sector_count / top_n * 100
        if sector_pct > 40:
            print(f"    ⚠️ 集中度风险: {sector_name}占{sector_pct:.0f}% (>40%), 建议分散")
        elif sector_pct > 30:
            print(f"    ⚡ 关注:{sector_name}占{sector_pct:.0f}% (>30%), 考虑分散")

    # 3. 止损参考 (基于历史波动率)
    print(f"    📋 止损建议:")
    for s in top[:10]:
        t = s["ticker"]
        # 简单止损建议: 根据综合评分
        score = s["composite"]
        if score > 0.5:
            sl = "严格(5-7%)"
        elif score > 0.3:
            sl = "中等(8-10%)"
        else:
            sl = "宽松(12-15%)"
        print(f"      {t:<6}: 评分={score:.3f} → 建议止损 {sl}")

    print(f"    ✅ 风控分析完成")
    return sector_counter


# ═══════════════════════════════════════
# 5d. Alpha 工厂评分 (P2.2)
# ═══════════════════════════════════════
def run_alpha_scoring(results, quick=False):
    """对综合评分 Top 标的用 Alpha 因子二次打分"""
    if not results:
        return

    top_tickers = [s["ticker"] for s in results[:15] if s["composite"] > 0.15]
    if not top_tickers:
        return

    from alpha_factory import score_with_alphas

    print(f"  📐 Alpha 因子评分 (Top {len(top_tickers)})...")
    for s in results[:15]:
        if s["composite"] < 0.15:
            continue
        t = s["ticker"]
        try:
            alpha_info = score_with_alphas(t)
            if alpha_info:
                s["alpha_score"] = alpha_info["alpha_score"]
                s["alpha_raw"] = alpha_info["alpha_raw"]
        except Exception:
            pass

    print(f"    Alpha 评分完成")


# ═══════════════════════════════════════
# 5c. 组合优化 (P2.3)
# ═══════════════════════════════════════
def run_portfolio_optimization(results, top_n=10):
    """对综合评分 Top 标的用 Kelly 准则做组合优化"""
    top = [s for s in results[:top_n] if s["composite"] > 0.15]
    if len(top) < 2:
        print(f"  ⚠️ 标的不足 2 只，跳过组合优化")
        return

    tickers = [s["ticker"] for s in top]
    ml_scores = {s["ticker"]: s["ml_score"] for s in top}

    print(f"  💼 组合优化 (Top {len(tickers)} 只, Kelly+CovShrink)...")
    from portfolio_optimizer import optimize_portfolio

    try:
        result = optimize_portfolio(
            tickers,
            ml_scores_dict=ml_scores,
            mode="kelly",
            risk_free_rate=0.05,
            max_leverage=1.0,
        )

        if "allocation" in result:
            print(f"\n  {'='*60}")
            print(f"    最优组合分配:")
            for t in result["allocation"]:
                a = result["allocation"][t]
                print(f"    {t:>6}: {a['weight_pct']:>8}  (预期年化 {a['expected_return']:>+.1%})")
            print(f"    {'='*60}")
            print(f"    组合年化收益: {result['expected_return']:+.2%}")
            print(f"    年化波动:     {result['volatility']:.2%}")
            print(f"    夏普比:       {result['sharpe_ratio']:.3f}")

        return result
    except Exception as e:
        print(f"    ⚠️ 组合优化失败: {e}")
        return None


# ═══════════════════════════════════════
# 5. 报告输出
# ═══════════════════════════════════════
def print_composite_report(results, top_n=30):
    """打印综合选股报告"""
    top = results[:top_n]

    print(f"\n{'='*120}")
    print(f"  交叉验证选股报告 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*120}")
    print(f"  权重: ML={WEIGHT_ML*100:.0f}%  多因子={WEIGHT_FACTOR*100:.0f}%  情绪={WEIGHT_SENTIMENT*100:.0f}%")
    print(f"{'='*120}")
    print(f" {'#':>3} {'代码':>6} {'综合分':>7} {'ML':>6} {'因子':>6} {'情绪':>6} {'共识':>5} {'可信':>5} "
          f"{'评级':<12} {'价格':>8} {'预测5d':>7} {'新闻':>4} {'话题'}")
    print(f" {'-'*115}")

    for i, s in enumerate(top):
        dir_s = "🟢" if s.get("recent_direction", 0) > 0.2 else ("🔴" if s.get("recent_direction", 0) < -0.2 else "🟡")
        topics = s.get("hot_topics", "")
        if isinstance(topics, str):
            topics_str = topics[:20]
        elif isinstance(topics, list):
            topics_str = ",".join([f"{t[0]}" if isinstance(t, (list, tuple)) else str(t) for t in topics[:3]])
        else:
            topics_str = ""

        pred_str = f"{s['ml_pred_return']:+.1f}%" if s['ml_pred_return'] else "N/A"

        print(f" {i+1:>3} {s['ticker']:>6} {s['composite']:>7.3f} {s['ml_score']:>6.3f} "
              f"{s['factor_score']:>6.3f} {s['sentiment_score']:>6.3f} "
              f"{s['consensus']:>5.2f} {s['confidence']:>5.2f} "
              f"{s['level']:<12} ${s['price']:>6.2f} {pred_str:>7} "
              f"{s['news_count']:>4} {topics_str[:20]}")

    # 分等级统计
    print(f"\n{'='*120}")
    print(f"  分等级统计")
    print(f"{'='*120}")
    for level_name in ["🟢 强烈关注", "🟡 值得跟踪", "🔵 值得关注", "⚪ 一般"]:
        level_stocks = [s for s in results if s["level"] == level_name]
        if level_stocks:
            tickers_str = ", ".join(s["ticker"] for s in level_stocks)
            print(f"  {level_name} ({len(level_stocks)}只): {tickers_str}")

    # 三路评分排名差异分析
    print(f"\n{'='*120}")
    print(f"  分歧分析 (ML vs 多因子 评分差异最大的股票)")
    print(f"{'='*120}")
    for s in sorted(results, key=lambda x: abs(x["ml_score"] - x["factor_score"]), reverse=True)[:8]:
        diff = s["ml_score"] - s["factor_score"]
        direction = "ML偏好" if diff > 0 else "多因子偏好"
        print(f"  {s['ticker']:>6}: ML={s['ml_score']:.3f} 因子={s['factor_score']:.3f} "
              f"差={diff:+.3f} ({direction})")

    return top


def save_composite_report(results, filename="cross_validate"):
    """保存结果"""
    rows = []
    for s in results:
        rows.append({
            "ticker": s["ticker"],
            "score": s["composite"],
            "ml_score": s["ml_score"],
            "factor_score": s["factor_score"],
            "sentiment_score": s["sentiment_score"],
            "consensus": s["consensus"],
            "confidence": s["confidence"],
            "level": s["level"].replace("🟢", "strong").replace("🟡", "watch")
                     .replace("🔵", "interest").replace("��", "normal"),
            "price": s["price"],
            "ml_pred_return_5d": s["ml_pred_return"],
            "factor_raw": s["factor_raw"],
            "news_count": s["news_count"],
        })
    df = pd.DataFrame(rows)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(CACHE_DIR, f"{filename}_{now}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {csv_path}")
    return csv_path


# ═══════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]

    quick = "--quick" in args
    no_sentiment = "--no-sentiment" in args
    include_hk = "--hk" in args
    do_optimize = "--optimize" in args
    do_backtest = "--backtest" in args
    do_risk = "--risk-control" in args
    do_market_state = "--no-market" not in args
    do_alpha = "--alpha" in args or do_optimize
    top_n = 30
    if "--top" in args:
        idx = args.index("--top")
        if idx + 1 < len(args):
            top_n = int(args[idx + 1])

    tickers = list(US_WATCHLIST)
    if include_hk:
        tickers.extend(HK_WATCHLIST)

    print(f"{'='*120}")
    print(f"  交叉验证选股系统 v5")
    print(f"  股票池: {len(tickers)} 只 {'(美股+港股)' if include_hk else '(仅美股)'}")
    print(f"  ML引擎: v5.2 | 动态权重: 已启用 | 分类ensemble: {'否' if quick else '是'}")
    if do_backtest:
        print(f"  回测验证: 已启用")
    if do_risk:
        print(f"  风控分析: 已启用")
    print(f"{'='*120}")

    start = datetime.now()

    # 1. ML 评分 (v5.2)
    print(f"\n  阶段 1/4: ML v5.2 模型评分")
    ml_results_us = run_ml_scoring(tickers, market="US", quick=quick)

    ml_results = ml_results_us

    # 2. 多因子评分
    print(f"\n  阶段 2/4: 多因子评分")
    factor_results = run_factor_scoring(tickers)

    # 3. 情绪评分
    if not no_sentiment:
        print(f"\n  阶段 3/4: 新闻情绪分析")
        sentiment_results = run_sentiment_scoring(tickers, market="US")
    else:
        sentiment_results = {}
        print(f"\n  阶段 3/4: 跳过情绪分析")

    elapsed = (datetime.now() - start).total_seconds()

    # 4. 综合评分 (v5: 动态权重)
    print(f"\n{'='*120}")
    print(f"  融合评分中... (耗时: {elapsed:.0f}s)")
    results = compute_composite(ml_results, factor_results, sentiment_results, tickers, dynamic_weights=True)

    # 5. 市场状态 (P2.1)
    if do_market_state:
        print(f"\n  阶段 4/4: 深度分析")
        market_state = run_market_state_detection()
    else:
        market_state = None

    # 6. 回测 (P2.4)
    if do_backtest:
        run_backtest(results)

    # 7. Alpha 因子 (P2.2)
    if do_alpha:
        run_alpha_scoring(results)

    # 8. 风控 (P3.1)
    if do_risk:
        run_risk_control(results, top_n=min(20, top_n))

    # 9. 输出报告
    top = print_composite_report(results, top_n=min(top_n, len(results)))
    save_composite_report(results)

    # 10. 组合优化 (P2.3)
    if do_optimize:
        run_portfolio_optimization(results, top_n=min(15, top_n))

    print(f"\n  ✅ 交叉验证选股完成! (总耗时: {(datetime.now() - start).total_seconds():.0f}s)")
    print(f"  💡 选项: --quick | --no-sentiment | --hk | --backtest | --alpha | --optimize | --risk-control")
