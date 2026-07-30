# Runtime entrypoints

Use these maintained entrypoints for new work:

- `python ml_optimized_picker_v5.py`: model scoring and selection.
- `python cron_market_job.py --dry-run`: scheduled selection and configuration proposals without writes.
- `python cron_market_job.py --apply-config`: explicitly apply a reviewed configuration proposal.
- `python scripts/run_historical_report.py --prices prices.csv --start 2024-01-01 --end 2025-01-01`: point-in-time replay report.
- `quant.backtest.engine`: reusable backtest API for tests and notebooks.

The root-level `ml_optimized_picker.py`, `ml_optimized_picker_v4.py`,
`news_sentiment.py`, and `news_sentiment_v2.py` are legacy compatibility
copies. Do not add new features to them. They are byte-identical in this
repository and should be removed or retained under a versioned Git tag only
after checking downstream cron, notebook, and deployment references.

Temporary/debug scripts (`_debug_*`, `_tmp.py`, exploratory demos) are not
production interfaces. New experiments belong under `experiments/` or in a
test, with a short README documenting data assumptions and reproducibility.
