from dataclasses import asdict

import pandas as pd

from .costs import TransactionCostModel
from .metrics import performance_metrics
from quant.data.validation import validate_price_frame, validate_signal_frame


def run_long_only_backtest(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    initial_cash: float = 100_000.0,
    top_n: int = 10,
    hold_days: int = 5,
    costs: TransactionCostModel | None = None,
) -> dict:
    """Run a deterministic next-session, equal-weight, long-only backtest.

    signals must contain signal_date, ticker, score. prices must contain
    date, ticker, close and optionally open. Signals are sorted and evaluated
    only on their signal_date; execution uses the next available price row.
    """
    if initial_cash <= 0 or top_n <= 0 or hold_days <= 0:
        raise ValueError("initial_cash, top_n and hold_days must be positive")
    costs = costs or TransactionCostModel()
    costs.validate()
    suspicious = [column for column in signals.columns if str(column).lower().startswith("future_")]
    if suspicious:
        raise ValueError(f"signals contain forbidden future-looking columns: {suspicious}")
    sig = validate_signal_frame(signals)
    px = validate_price_frame(prices)
    dates = sorted(px["date"].unique())
    cash = float(initial_cash)
    positions = {}
    trades = []
    equity_rows = []
    last_rebalance_index = None

    def price_for(date, ticker, field="close"):
        rows = px[(px["date"] == date) & (px["ticker"] == ticker)]
        if rows.empty:
            return None
        value = rows.iloc[0].get(field, rows.iloc[0]["close"])
        return float(value) if pd.notna(value) and float(value) > 0 else None

    for date_index, date in enumerate(dates):
        due = sig[sig["signal_date"] == date]
        can_rebalance = (
            last_rebalance_index is None
            or date_index - last_rebalance_index >= hold_days
        )
        if not due.empty and can_rebalance:
            future_dates = [candidate for candidate in dates if candidate > date]
            if future_dates:
                execution_date = future_dates[0]
                selected = due.head(top_n)["ticker"].tolist()
                target = {ticker: 1.0 / len(selected) for ticker in selected}
                last_rebalance_index = date_index
                current_value = cash + sum(
                    quantity * (price_for(execution_date, ticker) or average_cost)
                    for ticker, (quantity, average_cost) in positions.items()
                )
                all_tickers = set(positions) | set(target)
                for ticker in sorted(all_tickers):
                    current_price = price_for(execution_date, ticker, "open") or price_for(execution_date, ticker)
                    if current_price is None:
                        continue
                    current_quantity = positions.get(ticker, (0.0, current_price))[0]
                    buy_cost_rate = costs.rate("BUY")
                    target_quantity = current_value * target.get(ticker, 0.0) / (current_price * (1.0 + buy_cost_rate))
                    delta = target_quantity - current_quantity
                    if abs(delta) < 1e-10:
                        continue
                    side = "BUY" if delta > 0 else "SELL"
                    notional = abs(delta) * current_price
                    fee = costs.amount(notional, side)
                    if side == "BUY":
                        cash -= notional + fee
                    else:
                        cash += notional - fee
                    if abs(target_quantity) < 1e-10:
                        positions.pop(ticker, None)
                    else:
                        positions[ticker] = (target_quantity, current_price)
                    trades.append({
                        "signal_date": date,
                        "execution_date": execution_date,
                        "ticker": ticker,
                        "side": side,
                        "quantity": float(delta),
                        "price": current_price,
                        "notional": notional,
                        "cost": fee,
                    })
        market_value = cash + sum(
            quantity * (price_for(date, ticker) or average_cost)
            for ticker, (quantity, average_cost) in positions.items()
        )
        equity_rows.append({"date": date, "equity": market_value, "cash": cash})

    equity = pd.DataFrame(equity_rows).set_index("date")
    trade_frame = pd.DataFrame(trades)
    return {
        "equity": equity,
        "trades": trade_frame,
        "metrics": performance_metrics(equity["equity"]),
        "cost_model": asdict(costs),
        "initial_cash": initial_cash,
    }


def run_point_in_time_backtest(
    prices: pd.DataFrame,
    signal_generator,
    universe_root=None,
    market: str = "US",
    signal_dates=None,
    initial_cash: float = 100_000.0,
    top_n: int = 10,
    hold_days: int = 5,
    costs: TransactionCostModel | None = None,
) -> dict:
    """Generate signals using only information available on each signal date.

    signal_generator is called as generator(as_of, tickers, history) and must
    return columns signal_date, ticker and score. The history passed to it is
    truncated to date <= as_of; this is the primary anti-lookahead boundary.
    """
    px = validate_price_frame(prices)
    available_dates = sorted(px["date"].unique())
    requested = available_dates if signal_dates is None else [
        pd.Timestamp(value).normalize() for value in signal_dates
    ]
    from quant.data.universe import load_universe_as_of

    generated = []
    for as_of in requested:
        history = px[px["date"] <= as_of].copy()
        if history.empty:
            continue
        if universe_root is None:
            tickers = sorted(history["ticker"].dropna().unique().tolist())
        else:
            tickers = load_universe_as_of(universe_root, as_of, market)
        if not tickers:
            continue
        result = signal_generator(as_of, tickers, history)
        if result is None:
            continue
        result = validate_signal_frame(pd.DataFrame(result))
        if (result["signal_date"] > as_of).any():
            raise ValueError("signal_generator returned a signal dated after as_of")
        generated.append(result)
    if not generated:
        raise ValueError("signal_generator produced no point-in-time signals")
    signals = pd.concat(generated, ignore_index=True)
    result = run_long_only_backtest(
        signals,
        px,
        initial_cash=initial_cash,
        top_n=top_n,
        hold_days=hold_days,
        costs=costs,
    )
    result["signals"] = signals
    return result
