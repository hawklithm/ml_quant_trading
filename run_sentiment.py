#!/usr/bin/env python3
"""单独跑情绪融合，基于已有us_state.json的预测数据"""
import sys, os, json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 读状态
state_path = os.path.expanduser("~/.cache/hermes-quant/market_jobs/us_state.json")
with open(state_path) as f:
    state = json.load(f)

predictions = state["last_predictions"]
all_tickers = [p["ticker"] for p in predictions]
print(f"加载 {len(all_tickers)} 只股票的新闻情绪...")

from finbert_sentiment import build_sentiment_factors, sentiment_boost

sentiment_factors = build_sentiment_factors(all_tickers)

event_stocks = []
llm_count = 0
keyword_count = 0

for p in predictions:
    t = p["ticker"]
    sf = sentiment_factors.get(t, {})
    if sf and sf.get("news_count", 0) > 0:
        method = sf.get("method", "keyword")
        if method == "llm":
            llm_count += 1
        else:
            keyword_count += 1
        original = p["score"]
        fused, adj, evt_adj = sentiment_boost(original, sf)
        p["score"] = fused
        p["sentiment_adj"] = adj
        p["event_adj"] = evt_adj
        if sf.get("events"):
            event_stocks.append((t, sf["event_labels"], sf.get("event_discount", 1.0)))
    else:
        keyword_count += 1

predictions.sort(key=lambda x: x["score"], reverse=True)

print(f"\nDeepSeek-V4-Flash: {llm_count} 只 | 关键词/无新闻: {keyword_count} 只")
if event_stocks:
    print(f"\n⚠️  异常事件预警 ({len(event_stocks)} 只):")
    for t, labels, discount in event_stocks:
        print(f"    🔴 {t}: {' + '.join(labels)} (折扣 {discount:.3f})")
else:
    print("\n✅ 未检测到异常事件")

print(f"\n{'='*60}")
print(f"  Top 10 (情绪融合后)")
print(f"{'='*60}")
print(f"  {'#':>3} {'代码':>6} {'评分':>7} {'方向':>6} {'情绪调整':>9} {'事件折扣':>9}")
for i, p in enumerate(predictions[:10]):
    sa = p.get("sentiment_adj", 0)
    ea = p.get("event_adj", 1)
    print(f"  {i+1:>3} {p['ticker']:>6} {p['score']:.3f} {p['direction']:>6} {sa:+.4f} {ea:.4f}")

# 保存
state["last_predictions"] = predictions
with open(state_path, "w") as f:
    json.dump(state, f, indent=2, default=str)
print(f"\n✅ 情绪融合结果已保存")

# 输出简洁版给终端
print("\n---CRON_RESULT---")
result = {
    "market": "US",
    "date": state.get("last_pre_date", ""),
    "time": state.get("last_pre_time", ""),
    "total": len(predictions),
    "llm_count": llm_count,
    "events": len(event_stocks),
}
print(json.dumps(result, ensure_ascii=False))
