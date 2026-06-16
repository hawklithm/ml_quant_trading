"""新闻情绪分析模块 — 用于 ML 选股系统的实时资讯融合

功能:
  1. 从 Yahoo RSS 获取每只股票的最新新闻
  2. 用 LLM（DeepSeek）做新闻情绪打分 (-1 ~ +1)
  3. 检测关键主题（财报、并购、监管、回购、分析师评级等）
  4. 输出结构化的情绪因子，供 ml_stock_picker.py 使用

用法:
  python news_sentiment.py --ticker AAPL      # 单只分析
  python news_sentiment.py --batch             # 批量所有 watchlist
  python news_sentiment.py --update-score      # 生成情绪因子 → 写入 CSV
"""

import requests
from xml.etree import ElementTree
import json, os, sys, re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ═══════════════════════════════════════
# 配置
# ═════════════════════════════���═════════
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
NEWS_TTL_HOURS = 6  # 新闻缓存有效期
MAX_NEWS_PER_TICKER = 10

# 情绪关键词词典 — 快速预过滤
BULLISH_KEYWORDS = [
    "beat estimate", "upgrade", "buy rating", "outperform", "strong buy",
    "positive outlook", "raise guidance", "record revenue", "dividend increase",
    "buyback", "partnership", "FDA approval", "breakthrough", "bullish",
    "growth opportunity", "market share gain", "expansion", "acquisition",
    "exceed expectation", "ahead of estimate", "accelerate growth",
    "margin expansion", "cost cutting", "restructuring", "synergy",
    "new contract", "massive order", "backlog", "AI tailwind",
]

BEARISH_KEYWORDS = [
    "downgrade", "sell rating", "underperform", "reduce", "negative outlook",
    "miss estimate", "below estimate", "guidance cut", "revenue decline",
    "lawsuit", "investigation", "regulatory", "fine", "penalty",
    "CEO resignation", "executive departure", "layoff", "restructuring charge",
    "competition", "market share loss", "price war", "inventory glut",
    "supply chain disruption", "tariff", "trade war", "slowdown",
    "profit warning", "loss", "debt", "bankruptcy", "default",
]

TOPIC_PATTERNS = {
    "earnings": r"(earnings|quarterly result|Q[1-4]|fiscal|EPS|revenue|profit)",
    "dividend_buyback": r"(dividend|buyback|share repurchase|shareholder return)",
    "analyst": r"(upgrade|downgrade|rating|analyst|price target|overweight|equal.weight)",
    "m_and_a": r"(merger|acquisition|takeover|buy|acquire|merge|deal)",
    "regulatory": r"(SEC|FDA|regulatory|approval|investigation|lawsuit|fine)",
    "product": r"(launch|product|new feature|update|release|beta)",
    "partnership": r"(partner|collaboration|alliance|joint venture|teams? with)",
    "AI": r"(AI|artificial intelligence|machine learning|deep learning|LLM|GPT)",
    "macro": r"(interest rate|inflation|Fed|recession|GDP|employment|tariff|trade)",
}


# ═══════════════════════════════════════
# 1. 新闻获取
# ═══════════════════════════════════════
def fetch_news(ticker, max_items=MAX_NEWS_PER_TICKER):
    """从 Yahoo RSS 获取股票新闻"""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return []

        root = ElementTree.fromstring(r.content)
        items = root.findall(".//item")

        news = []
        for item in items[:max_items]:
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            pubdate = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            news.append({
                "title": title.strip(),
                "summary": re.sub(r"<[^>]+>", "", desc).strip()[:500],
                "date": pubdate,
                "link": link,
            })
        return news
    except Exception as e:
        return []


def fetch_batch_news(tickers):
    """批量获取新闻（有本地缓存）"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "news_cache.json")

    # 读取缓存
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            # 清理过期缓存
            now = datetime.now()
            for k in list(cache.keys()):
                cached_time = datetime.fromisoformat(cache[k].get("_cached_at", "2000-01-01"))
                if (now - cached_time).total_seconds() > NEWS_TTL_HOURS * 3600:
                    del cache[k]
        except:
            cache = {}

    results = {}
    for t in tickers:
        if t in cache:
            results[t] = cache[t]["news"]
            continue

        news = fetch_news(t)
        results[t] = news

        # 写入缓存
        cache[t] = {
            "news": news,
            "_cached_at": datetime.now().isoformat(),
        }

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, default=str)

    return results


# ═══════════════════════════════════════
# 2. 情绪分析
# ═════════���═════════════════════════════
def quick_sentiment(title, summary):
    """
    快速关键词情绪评分（无需 LLM）
    返回: sentiment_score (-1 ~ +1), topics: list
    """
    text = (title + " " + summary).lower()

    # 关键词匹配
    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)

    # 话题检测
    topics = []
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, text):
            topics.append(topic)

    # 情绪打分
    total = bullish_count + bearish_count
    if total > 0:
        score = (bullish_count - bearish_count) / total
    else:
        score = 0.0  # 中性

    # 如果是纯粹的财报分析类新闻，轻微偏中性
    if "earnings" in topics and total <= 1:
        score *= 0.5

    return np.clip(score, -1.0, 1.0), topics


def deep_sentiment(ticker, news_list):
    """
    完整情绪分析（关键词 + 聚合）
    返回结构化情绪因子
    """
    if not news_list:
        return {
            "sentiment_score": 0.0,
            "sentiment_urgency": 0.0,
            "news_count": 0,
            "topics": [],
            "hot_topics": [],
            "recent_headlines": [],
            "recent_direction": 0,
        }

    scores = []
    all_topics = []
    headlines = []

    for n in news_list:
        s, topics = quick_sentiment(n["title"], n["summary"])
        scores.append(s)
        all_topics.extend(topics)
        headlines.append(n["title"])

        # 检测时间衰减（最近24小时的新闻权重更高）
        try:
            pub = datetime.strptime(n["date"][:25], "%a, %d %b %Y %H:%M:%S")
            hours_ago = (datetime.now() - pub).total_seconds() / 3600
            if hours_ago < 48:
                # 近2天新闻权重 x1.5
                scores[-1] *= 1.5
        except:
            pass

    # 聚合情绪
    avg_sentiment = np.clip(np.mean(scores), -1.0, 1.0)

    # 情绪一致性（0=分歧大, 1=一致高, -1=一致低）
    if len(scores) >= 2:
        consistency = 1.0 - np.std(scores)  # 标准差越小越一致
    else:
        consistency = 0.5

    # 最近方向（最新的2-3条新闻情绪方向）
    recent = scores[:3]
    recent_direction = np.mean(recent)

    # 热门话题
    topic_counts = {}
    for t in all_topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    hot_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]

    # 新闻热度（数量+时效性=紧迫度）
    urgency = min(len(news_list) / 10, 1.0) * (1.0 + abs(avg_sentiment))

    return {
        "sentiment_score": round(avg_sentiment, 4),
        "sentiment_consistency": round(consistency, 4),
        "sentiment_urgency": round(urgency, 4),
        "news_count": len(news_list),
        "topics": list(set(all_topics)),
        "hot_topics": hot_topics,
        "recent_headlines": headlines[:5],
        "recent_direction": round(recent_direction, 4),
    }


# ═══════════════════════════════════════
# 3. 生成情绪因子供 ML 系统使用
# ═══════════════════════════════════════
def build_sentiment_factors(tickers):
    """生成情绪因子字典 {ticker: {factor1: val, ...}}"""
    news_data = fetch_batch_news(tickers)

    factors = {}
    for t in tickers:
        news = news_data.get(t, [])
        sa = deep_sentiment(t, news)

        factors[t] = {
            "sentiment_score": sa["sentiment_score"],
            "sentiment_urgency": sa["sentiment_urgency"],
            "sentiment_consistency": sa["sentiment_consistency"],
            "news_count": sa["news_count"],
            "hot_topics": ",".join([f"{k}({v})" for k, v in sa["hot_topics"]]),
            "recent_direction": sa["recent_direction"],
        }
    return factors


# ═══════════════════════════════════════
# 4. 整合到 ML 评分 (核心逻辑)
# ═══════════════════════════════════════
def sentiment_boost(ml_score_original, sentiment_factors, weight=0.15):
    """
    将情绪因子融合进 ML 评分

    Args:
        ml_score_original: ML 模型评分 (0~1)
        sentiment_factors: 情绪因子 dict
        weight: 情绪权重 (默认 15%)

    Returns:
        fused_score: 融合后评分 (0~1)
        adjustment: 调整幅度
    """
    s = sentiment_factors

    # 情绪信号强度 (-1 ~ +1)
    sentiment_signal = s.get("sentiment_score", 0) * 0.4
    sentiment_signal += s.get("recent_direction", 0) * 0.3
    sentiment_signal += (s.get("sentiment_urgency", 0) - 0.5) * 0.3

    # 无新闻则中立
    if s.get("news_count", 0) == 0:
        sentiment_signal = 0

    # 有重大负面话题则打折
    topics = s.get("hot_topics", "")
    has_negative_topic = any(
        kw in topics.lower()
        for kw in ["regulatory", "lawsuit", "downgrade"]
    )
    if has_negative_topic and sentiment_signal > 0:
        sentiment_signal *= 0.5

    # 情绪信号从 [-1, 1] 映射到 [0, 1]
    sentiment_norm = (sentiment_signal + 1) / 2

    # 融合: ML (默认 85%) + 情绪 (15%)
    fused = ml_score_original * (1 - weight) + sentiment_norm * weight
    adjustment = fused - ml_score_original

    return np.clip(fused, 0, 1), adjustment


# ═══════════════════════════════════════
# 5. CLI
# ═══════════════════════════════════════
def print_sentiment_report(ticker, sentiment):
    """打印情绪报告"""
    s = sentiment
    print(f"\n{'='*70}")
    print(f"  📰 {ticker} 新闻情绪分析 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # 情绪评分
    score = s["sentiment_score"]
    if score > 0.3:
        emoji = "🟢"
        label = "���多"
    elif score < -0.3:
        emoji = "🔴"
        label = "偏空"
    else:
        emoji = "🟡"
        label = "中性"
    print(f"\n  情绪评分: {emoji} {score:+.3f} ({label})")
    print(f"  一致性:   {s['sentiment_consistency']:.2f}")
    print(f"  紧迫度:   {s['sentiment_urgency']:.2f}")

    # 新闻数量
    print(f"\n  新闻条数: {s['news_count']}")

    # 热门话题
    if s["hot_topics"]:
        topics_str = ", ".join([f"{t}({c})" for t, c in s["hot_topics"]])
        print(f"  热门话题: {topics_str}")

    # 最近新闻
    if s["recent_headlines"]:
        print(f"\n  最近头条:")
        for h in s["recent_headlines"][:5]:
            print(f"    • {h[:90]}")


if __name__ == "__main__":
    from ml_stock_picker import US_WATCHLIST

    args = sys.argv[1:]

    if "--ticker" in args:
        idx = args.index("--ticker")
        if idx + 1 < len(args):
            t = args[idx + 1].upper()
            news = fetch_news(t, max_items=15)
            sa = deep_sentiment(t, news)
            print_sentiment_report(t, sa)

    elif "--batch" in args:
        print(f"\n  批量获取 {len(US_WATCHLIST)} 只股票新闻情绪...")
        factors = build_sentiment_factors(US_WATCHLIST)

        # 打印情绪排名
        ranked = sorted(factors.items(), key=lambda x: -x[1]["sentiment_score"])
        print(f"\n{'='*70}")
        print(f"  情绪评分排名")
        print(f"{'='*70}")
        print(f" {'#':>3} {'代码':>6} {'情绪分':>8} {'新闻数':>6} {'方向':>7} {'紧迫度':>8} {'热门话题'}")
        for i, (t, f) in enumerate(ranked[:15]):
            dir_s = "🟢" if f["recent_direction"] > 0.2 else ("🔴" if f["recent_direction"] < -0.2 else "🟡")
            print(f" {i+1:>3} {t:>6} {f['sentiment_score']:>+7.3f} {f['news_count']:>5} "
                  f"{dir_s}{f['recent_direction']:>+5.2f} {f['sentiment_urgency']:>7.3f} "
                  f"{f['hot_topics'][:40]}")

        # 保存情绪因子到缓存
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "sentiment_factors.json"), "w") as f:
            json.dump(factors, f, indent=2)
        print(f"\n  已保存情绪因子到缓存")

    elif "--update-score" in args:
        # 整合进 ML 选股系统
        # 1. 加载最新的 ML 选股结果
        # 2. 融合情绪因子
        # 3. 输出更新后的推荐

        from ml_stock_picker import run_picking, print_results, US_WATCHLIST

        print(f"\n  步骤1/3: 运行 ML 选股系统...")
        results = run_picking(period="2y")
        if not results:
            print("  ❌ ML 选股失败")
            sys.exit(1)

        print(f"\n  步骤2/3: 获取新闻情绪...")
        factors = build_sentiment_factors(US_WATCHLIST)

        print(f"\n  步骤3/3: 融合情绪因子...")
        for r in results:
            t = r["ticker"]
            sf = factors.get(t, {})
            if sf:
                original = r["score"]
                fused, adj = sentiment_boost(original, sf)
                r["score"] = fused
                r["sentiment_adj"] = adj
                r["sentiment_factors"] = sf
            else:
                r["sentiment_adj"] = 0
                r["sentiment_factors"] = {"sentiment_score": 0}

        # 重排序
        results.sort(key=lambda x: x["score"], reverse=True)
        top = print_results(results, top_n=20)

        # 打印情绪调整详情
        print(f"\n{'='*120}")
        print(f"  情绪因子调整详情")
        print(f"{'='*120}")
        print(f" {'代码':>6} {'原始ML':>8} {'情绪分':>8} {'调整':>8} {'融合后':>8} {'新闻':>5} {'方向':>6} {'话题'}")
        print(f" {'-'*70}")
        for r in results[:10]:
            sf = r.get("sentiment_factors", {})
            adj = r.get("sentiment_adj", 0)
            orig = r["score"] - adj
            dir_s = "🟢" if sf.get("recent_direction", 0) > 0.2 else ("🔴" if sf.get("recent_direction", 0) < -0.2 else "🟡")
            print(f" {r['ticker']:>6} {orig:>8.4f} {sf.get('sentiment_score',0):>+7.3f} "
                  f"{adj:>+7.4f} {r['score']:>8.4f} "
                  f"{sf.get('news_count',0):>4} {dir_s} {sf.get('recent_direction',0):>+5.2f} "
                  f"{sf.get('hot_topics','')[:25]}")

        # 保存
        rows = []
        for r in results[:20]:
            sf = r.get("sentiment_factors", {})
            rows.append({
                "ticker": r["ticker"],
                "ml_raw": round(r["score"] - r.get("sentiment_adj", 0), 4),
                "sentiment_score": sf.get("sentiment_score", 0),
                "sentiment_adj": round(r.get("sentiment_adj", 0), 4),
                "fused_score": round(r["score"], 4),
                "news_count": sf.get("news_count", 0),
                "hot_topics": sf.get("hot_topics", ""),
            })
        df = pd.DataFrame(rows)
        p = f"ml_fused_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"\n  已保存: {p}")
        print(f"\n  ✅ 情绪融合完成!")

    else:
        print(__doc__)
