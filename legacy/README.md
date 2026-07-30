# Legacy entrypoints

The former `ml_optimized_picker.py`, `ml_optimized_picker_v4.py`,
`news_sentiment.py`, `news_sentiment_v2.py`, and their `ml_deep_scan*`
consumers were removed from the runtime tree after migration to:

- `ml_optimized_picker_v5.py` for model scoring;
- `finbert_sentiment.py` for news factors;
- `quant/backtest` for point-in-time replay;
- `cross_sectional_picker.py` and `cron_market_job.py` for maintained flows.

The deleted implementations remain available in Git history before the
refactor commit. New code must not import them. Use `quant/ENTRYPOINTS.md`
for maintained commands.
