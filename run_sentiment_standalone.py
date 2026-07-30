#!/usr/bin/env python3
"""Run sentiment + anomaly detection using saved predictions from state file."""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finbert_sentiment import build_sentiment_factors, sentiment_boost

state_path = os.path.expanduser("~/.cache/hermes-quant/market_jobs/us_state.json")
with open(state_path) as f:
    state = json.load(f)

if not state.get("last_predictions"):
    print("No predictions found")
    sys.exit(1)

predictions = state["last_predictions"]
all_tickers = [p["ticker"] for p in predictions]
print(f"正在获取 {len(all_tickers)} 只股票的最新新闻 + 情绪分析...")

sentiment_factors = build_sentiment_factors(all_tickers)

event_stocks = []
for p in predictions:
    t = p["ticker"]
    sf = sentiment_factors.get(t, {})
    if sf and sf.get("news_count", 0) > 0:
        original = p["score"]
        fused, adj, evt_adj = sentiment_boost(original, sf)
        p["score"] = fused
        p["sentiment_adj"] = adj
        p["event_adj"] = evt_adj
        if sf.get("events"):
            event_stocks.append((t, sf.get("event_labels", []), sf.get("event_discount", 1.0)))

if event_stocks:
    print(f'\n  \u26a0\ufe0f  异常事件预警 ({len(event_stocks)} 只):')
    for t, labels, discount in event_stocks:
        print(f'    \U0001f534 {t}: {" + ".join(labels)} (折扣 {discount:.3f})')
else:
    print("\n  \u2705 未检测到异常事件")

sentiment_count = sum(1 for p in predictions if "sentiment_adj" in p)
print(f"  情绪融合完成: {sentiment_count}/{len(all_tickers)} 只股票有新闻数据")

# Save updated state
state["last_predictions"] = predictions
with open(state_path, "w") as f:
    json.dump(state, f, indent=2, default=str)
print("  \U0001f4be 更新后的评分已保存到状态文件")

# Print sorted results
predictions.sort(key=lambda x: x["score"], reverse=True)
print(f"\n  {'='*60}")
print(f"  \U0001f4ca 情绪融合后 Top 10")
print(f"  {'#':>3} {'代码':>8} {'评分':>7} {'方向':>6} {'价格':>8} {'情绪调':>6}")
print(f"  {'-'*50}")
for i, p in enumerate(predictions[:10]):
    adj_str = f"{p.get('sentiment_adj',0):+.3f}" if "sentiment_adj" in p else "  N/A"
    print(f"  {i+1:>3} {p['ticker']:>8} {p['score']:>7.3f} {p['direction']:>6} {p['price']:>8.2f} {adj_str:>6}")

# Direction distribution
directions = {}
for p in predictions:
    directions.setdefault(p["direction"], 0)
    directions[p["direction"]] += 1
dir_str = " | ".join(f"{k}: {v}只" for k, v in sorted(directions.items()))
print(f"\n  方向分布: {dir_str}")
