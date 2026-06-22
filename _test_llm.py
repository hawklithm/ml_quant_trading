#!/usr/bin/env python3
"""Test llm_sentiment_by_ticker with actual code"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finbert_sentiment import _get_llm_client, llm_sentiment_by_ticker, fetch_news

client = _get_llm_client()
print(f"Client: {'OK' if client else 'FAIL'}")

news = fetch_news("AAPL", max_items=3)
print(f"News: {len(news)} items")

result = llm_sentiment_by_ticker("AAPL", news)
print(f"Result: {result}")

if result:
    print(f"\nscore={result['sentiment_score']} events={result['events']} reason={result.get('sentiment_reason','')}")
    print("ALL PASS")
else:
    print("FAIL")
    sys.exit(1)
