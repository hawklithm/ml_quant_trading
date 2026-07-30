import numpy as np
import pandas as pd


def performance_metrics(equity: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """Calculate reproducible portfolio metrics from daily equity values."""
    values = pd.Series(equity, dtype=float).dropna().sort_index()
    if len(values) < 2 or (values <= 0).any():
        raise ValueError("equity must contain at least two positive observations")
    daily = values.pct_change().dropna()
    years = max((values.index[-1] - values.index[0]).days / 365.25, 1 / 365.25)
    total_return = values.iloc[-1] / values.iloc[0] - 1.0
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0
    volatility = daily.std(ddof=1) * np.sqrt(252) if len(daily) > 1 else 0.0
    sharpe = (daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if daily.std(ddof=1) > 0 else 0.0
    downside = daily[daily < 0].std(ddof=1)
    sortino = (daily.mean() / downside * np.sqrt(252)) if pd.notna(downside) and downside > 0 else 0.0
    drawdown = values / values.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    result = {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "annualized_volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(cagr / abs(max_drawdown)) if max_drawdown < 0 else 0.0,
        "observations": int(len(values)),
    }
    if benchmark is not None:
        aligned = pd.concat([values.rename("strategy"), pd.Series(benchmark, dtype=float).rename("benchmark")], axis=1).dropna()
        if len(aligned) >= 2:
            result["benchmark_return"] = float(aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0] - 1.0)
            result["excess_return"] = float(aligned["strategy"].iloc[-1] / aligned["strategy"].iloc[0] - aligned["benchmark"].iloc[-1] / aligned["benchmark"].iloc[0])
    return result


def rank_ic(scores: pd.Series, realized_returns: pd.Series) -> float:
    """Spearman rank correlation for cross-sectional signal validation."""
    frame = pd.concat([pd.Series(scores, name="score"), pd.Series(realized_returns, name="return")], axis=1).dropna()
    if len(frame) < 3 or frame["score"].nunique() < 2 or frame["return"].nunique() < 2:
        return 0.0
    return float(frame["score"].rank().corr(frame["return"].rank()))


def quantile_spread(scores: pd.Series, realized_returns: pd.Series, quantiles: int = 5) -> dict:
    """Return top/bottom quantile mean returns and their spread."""
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    frame = pd.concat([pd.Series(scores, name="score"), pd.Series(realized_returns, name="return")], axis=1).dropna()
    if len(frame) < quantiles:
        return {"top_mean": 0.0, "bottom_mean": 0.0, "spread": 0.0, "observations": len(frame)}
    frame["bucket"] = pd.qcut(frame["score"].rank(method="first"), quantiles, labels=False)
    top = frame.loc[frame["bucket"] == quantiles - 1, "return"]
    bottom = frame.loc[frame["bucket"] == 0, "return"]
    top_mean = float(top.mean()) if not top.empty else 0.0
    bottom_mean = float(bottom.mean()) if not bottom.empty else 0.0
    return {
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "spread": top_mean - bottom_mean,
        "observations": int(len(frame)),
    }


def turnover(previous_weights: pd.Series, current_weights: pd.Series) -> float:
    """One-way turnover as half the absolute weight change."""
    frame = pd.concat(
        [pd.Series(previous_weights, dtype=float), pd.Series(current_weights, dtype=float)],
        axis=1,
    ).fillna(0.0)
    return float(frame.iloc[:, 0].sub(frame.iloc[:, 1]).abs().sum() / 2.0)
