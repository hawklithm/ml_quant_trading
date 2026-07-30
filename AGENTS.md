# Repository Guidelines

## Project Structure & Module Organization

This repository is a flat Python quant-trading research project; there is no `src/` package or dedicated assets directory. Core entry points and reusable modules are at the repository root:

- `ml_optimized_picker_v5.py`, `cross_sectional_picker.py`, and `cron_market_job.py` contain the current ML selection and scheduled market workflows.
- `news_sentiment*.py`, `finbert_sentiment.py`, `market_state.py`, and `tencent_data.py` provide sentiment, regime, and market-data integrations.
- `live_pipeline.py`, `data_archiver.py`, `strategies.py`, and `bt_*.py` support simulation, backtesting, and runtime data handling.
- `v5_config.json` holds model and workflow configuration. Generated databases, charts, CSVs, logs, caches, and secrets are intentionally ignored by Git.

## Build, Test, and Development Commands

Use a Python 3.12 virtual environment and install the dependencies described in `README.md` (including NumPy, pandas, scikit-learn, yfinance, AkShare, Backtrader, and TA-Lib).

```bash
python hello_quant.py AAPL
python ml_optimized_picker_v5.py --us-only --top 10
python cron_market_job.py --market US --mode post --sentiment
python live_pipeline.py --backfill AAPL SPY
python live_pipeline.py --report
```

These commands provide a data smoke test, ML selection run, scheduled pre/post workflow, historical backfill, and report generation. Run from the repository root so relative configuration and cache paths resolve correctly.

## Coding Style & Naming Conventions

Follow standard Python style with four-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and uppercase constants. Keep command-line behavior behind `if __name__ == "__main__":`; use `argparse` for new flags. Match the existing straightforward, script-oriented style and add concise comments where data-source fallbacks or model assumptions are non-obvious. No formatter or linter is currently configured.

## Testing Guidelines

There is no formal test framework or coverage gate. Use `test_yf.py` and the relevant executable scripts as smoke tests, and validate both US and HK paths when changing market-data logic. For model changes, compare backtest results and ensure missing, throttled, or empty data is handled without crashing.

## Commit & Pull Request Guidelines

Recent commits use short, imperative-style prefixes such as `feat:`, `fix:`, `refactor:`, and `init:`, with concise Chinese or English descriptions. Follow that pattern and keep each commit focused. Pull requests should explain the affected workflow, configuration changes, data-source/API implications, and validation commands/results; include sample output or charts when behavior or model performance changes.

## Security & Configuration Tips

Never commit API keys, `.env` files, credentials, cached databases, or generated outputs. Prefer environment variables for secrets, review `v5_config.json` changes carefully, and remember that live-pipeline execution is simulated and must not be treated as production trading without independent validation.
