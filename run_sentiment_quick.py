#!/usr/bin/env python3
"""Fast sentiment + anomaly detection using keyword fallback only (no LLM)."""
import sys, os, json, re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Quick keyword-only approach — no LLM calls
from finbert_sentiment import (
    fetch_batch_news, detect_events, keyword_sentiment,
    detect_topics, compute_event_discount, sentiment_boost,
    EVENT_LABELS, CACHE_DIR
)

state_path = os.path.expanduser("~/.cache/hermes-quant/market_jobs/us_state.json")
with open(state_path) as f:
    state = json.load(f)

predictions = state.get("last_predictions", [])
if not predictions:
    print("No predictions found")
    sys.exit(1)

all_tickers = [p["ticker"] for p in predictions]
print(f"正在获取 {len(all_tickers)} 只股票的最新新闻...")

news_data = fetch_batch_news(all_tickers)

factors = {}
for t in all_tickers:
    news = news_data.get(t, [])
    if not news:
        factors[t] = {
            "sentiment_score": 0.0, "sentiment_urgency": 0.0,
            "sentiment_consistency": 0.5, "news_count": 0,
            "hot_topics": "", "recent_direction": 0.0, "method": "keyword",
            "events": [], "event_discount": 1.0, "event_labels": [],
        }
        continue

    scores = []
    all_topics = []
    all_events = []

    for n in news:
        result = keyword_sentiment(n["title"], n["summary"])
        s = result["score"]
        events = detect_events(n["title"], n["summary"])
        topics = detect_topics(n["title"], n["summary"])

        if events:
            all_events.extend(events)
        scores.append(s)
        all_topics.extend(topics)

    avg_sentiment = np.clip(np.mean(scores), -1.0, 1.0)
    consistency = 1.0 - np.std(scores) if len(scores) >= 2 else 0.5
    recent_dir = np.mean(scores[:3]) if scores else 0.0
    urgency = min(len(news) / 10, 1.0) * (1.0 + abs(avg_sentiment))

    topic_counts = {}
    for tpc in all_topics:
        topic_counts[tpc] = topic_counts.get(tpc, 0) + 1
    hot_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]
    hot_str = ",".join(f"{k}({v})" for k, v in hot_topics)

    event_counts = {}
    for evt in all_events:
        et = evt["event_type"]
        if et not in event_counts:
            event_counts[et] = {"count": 0, "min_severity": 1.0}
        event_counts[et]["count"] += 1
        event_counts[et]["min_severity"] = min(event_counts[et]["min_severity"], evt["severity"])

    unique_events = [{"event_type": et, "count": info["count"], "severity": info["min_severity"]}
                     for et, info in event_counts.items()]
    event_discount = compute_event_discount(unique_events)

    factors[t] = {
        "sentiment_score": round(avg_sentiment, 4),
        "sentiment_urgency": round(urgency, 4),
        "sentiment_consistency": round(consistency, 4),
        "news_count": len(news),
        "hot_topics": hot_str,
        "recent_direction": round(recent_dir, 4),
        "method": "keyword",
        "events": unique_events,
        "event_discount": round(event_discount, 4),
        "event_labels": [EVENT_LABELS.get(e["event_type"], e["event_type"]) for e in unique_events],
    }

# Apply sentiment + event adjustments
event_stocks = []
for p in predictions:
    t = p["ticker"]
    sf = factors.get(t, {})
    if sf and sf.get("news_count", 0) > 0:
        original = p["score"]
        fused, adj, evt_adj = sentiment_boost(original, sf)
        p["score"] = fused
        p["sentiment_adj"] = adj
        p["event_adj"] = evt_adj
        if sf.get("events"):
            event_stocks.append((t, sf["event_labels"], sf["event_discount"]))

# Report
print(f"\n{'='*60}")
print(f"  📰 情绪融合 + 异常事件检测结果 — 2026-06-22")
print(f"{'='*60}")

if event_stocks:
    print(f"\n  ⚠️  异常事件预警 ({len(event_stocks)} 只):")
    for t, labels, discount in event_stocks:
        print(f"    🔴 {t}: {' + '.join(labels)} (折扣 {discount:.3f})")
else:
    print("\n  ✅ 未检测到异常事件")

sentiment_count = sum(1 for p in predictions if "sentiment_adj" in p)
print(f"  情绪融合完成: {sentiment_count}/{len(all_tickers)} 只股票有新闻数据")

# Sort and print Top 10
predictions.sort(key=lambda x: x["score"], reverse=True)
print(f"\n  {'#':>3} {'代码':>8} {'评分':>7} {'方向':>6} {'价格':>8} {'情绪调':>8}")
print(f"  {'-'*55}")
for i, p in enumerate(predictions[:10]):
    adj_str = f"{p.get('sentiment_adj',0):+.4f}" if "sentiment_adj" in p else "   N/A  "
    evt_str = f"|事件{p.get('event_adj',0):+.4f}" if p.get("event_adj",0) != 0 else ""
    print(f"  {i+1:>3} {p['ticker']:>8} {p['score']:>7.4f} {p['direction']:>6} {p['price']:>8.2f} {adj_str}{evt_str}")

# Direction distribution
directions = {}
for p in predictions:
    directions.setdefault(p["direction"], 0)
    directions[p["direction"]] += 1
dir_str = " | ".join(f"{k}: {v}只" for k, v in sorted(directions.items()))
print(f"\n  方向分布: {dir_str}")

# Show some headlines for top event stocks
if event_stocks:
    print(f"\n  📋 相关新闻头条:")
    for t, labels, _ in event_stocks:
        news = news_data.get(t, [])
        if news:
            print(f"\n  {t} ({' + '.join(labels)}):")
            for n in news[:3]:
                print(f"    · {n['title'][:100]}")

# Save updated state
state["last_predictions"] = predictions
with open(state_path, "w") as f:
    json.dump(state, f, indent=2, default=str)
print(f"\n  💾 状态已保存")
