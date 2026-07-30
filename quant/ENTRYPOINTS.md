# Runtime entrypoints

Use these maintained entrypoints for new work:

- `python ml_optimized_picker_v5.py`: model scoring and selection.
- `python cron_market_job.py --dry-run`: scheduled selection and configuration proposals without writes.
- `python cron_market_job.py --apply-config`: explicitly apply a reviewed configuration proposal.
- `python scripts/run_historical_report.py --prices prices.csv --start 2024-01-01 --end 2025-01-01`: point-in-time replay report.
- `quant.backtest.engine`: reusable backtest API for tests and notebooks.

The former root-level duplicate picker and news modules were removed after
checking repository references. Their implementations remain recoverable
from Git history; do not reintroduce them as runtime entrypoints.

Temporary/debug scripts (`_debug_*`, `_tmp.py`, exploratory demos) are not
production interfaces. New experiments belong under `experiments/` or in a
test, with a short README documenting data assumptions and reproducibility.
