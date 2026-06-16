#!/usr/bin/env python3
"""
组合构建优化模块 (P2.3)
====================
Kelly 准则 + Ledoit-Wolf 收缩协方差矩阵的组合优化。

用法:
  python portfolio_optimizer.py --tickers AAPL,MSFT,GOOGL,NVDA,META      # 优化组合
  python portfolio_optimizer.py --tickers ... --ml-scores 0.6,0.5,0.4,0.3,0.2  # 带 ML 评分先验
  python portfolio_optimizer.py --tickers ... --risk-parity              # 风险平价模式
  python portfolio_optimizer.py --tickers ... --equal-weight             # 等权模式对比
"""

import numpy as np
import pandas as pd
import warnings, sys, os
from datetime import datetime
from scipy.optimize import minimize, Bounds, LinearConstraint

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

# ═════════════════════════════════════���═
# 1. Ledoit-Wolf 收缩协方差
# ═══════════════════════════════════════
def ledoit_wolf_cov(returns_df):
    """
    Ledoit-Wolf 收缩协方差矩阵估计。

    Args:
        returns_df: DataFrame, 每列为一只股票的日收益率序列

    Returns:
        np.array: 收缩后的协方差矩阵
    """
    from sklearn.covariance import LedoitWolf

    # 清理
    clean = returns_df.dropna(how="all")
    if clean.empty:
        raise ValueError("没有有效数据")

    # 填充少量 NaN 为 0 (影响很小)
    clean = clean.fillna(clean.mean())

    lw = LedoitWolf().fit(clean.values)
    return lw.covariance_


# ═══════════════════════════════════════
# 2. Kelly 最优权重
# ═══════════════════════════════════════
def kelly_optimal_weights(expected_returns, cov_matrix, risk_free_rate=0.05,
                          max_leverage=1.0, no_short=False):
    """
    Kelly 准则最优权重。

    纯 Kelly: w = Σ^(-1) * μ
    带约束版本: 用 scipy.minimize ���解 max E[log(1 + w'R)]

    Args:
        expected_returns: np.array, 预期年化收益率
        cov_matrix: np.array, 年化协方差矩阵
        risk_free_rate: 无风险利率
        max_leverage: 最大杠杆 (1.0 = 满仓)
        no_short: 是否禁止做空

    Returns:
        np.array: 最优权重
    """
    n = len(expected_returns)

    # 纯 Kelly (无约束)
    try:
        raw_weights = np.linalg.solve(cov_matrix, expected_returns - risk_free_rate)
        # 归一化到 max_leverage
        gross = np.sum(np.abs(raw_weights))
        if gross > max_leverage:
            raw_weights = raw_weights / gross * max_leverage
    except np.linalg.LinAlgError:
        raw_weights = np.ones(n) / n

    # 如果不需要约束，直接返回
    if not no_short and max_leverage >= np.sum(np.abs(raw_weights)):
        return raw_weights

    # 带约束 (max_leverage + no_short)
    def _neg_growth(w):
        port_ret = np.dot(w, expected_returns - risk_free_rate)
        port_var = np.dot(w.T, np.dot(cov_matrix, w))
        # 近似 Kelly: 最大化 log(1 + r) ≈ r - 0.5 * var(r)
        return -(port_ret - 0.5 * port_var)

    # 约束条件
    constraints = []
    if max_leverage < 10:
        # 总杠杆约束
        constraints.append({
            "type": "eq",
            "fun": lambda x: np.sum(x) - 1.0  # 满仓
        })

    bounds = Bounds(-1 if not no_short else 0, max_leverage)

    # 初值
    x0 = np.ones(n) / n

    result = minimize(
        _neg_growth, x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12}
    )

    if result.success:
        return result.x
    else:
        # 兜底: 等权
        return np.ones(n) / n


# ═══════════════════════════════════════
# 3. 风险平价
# ═══════════════════════════════════════
def risk_parity_portfolio(cov_matrix):
    """
    风险平价组合 (Risk Parity)。

    每只资产的风险贡献相等。

    Returns:
        np.array: 权重
    """
    n = cov_matrix.shape[0]

    def _risk_contribution(w, cov):
        """每只资产的风险贡献"""
        port_var = np.dot(w.T, np.dot(cov, w))
        return np.dot(cov, w) * w / np.sqrt(port_var)

    def _risk_parity_obj(w, cov):
        rc = _risk_contribution(w, cov)
        target = np.mean(rc)
        return np.sum((rc - target) ** 2)

    bounds = Bounds(0, 1)
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1.0}]
    x0 = np.ones(n) / n

    result = minimize(
        _risk_parity_obj, x0,
        args=(cov_matrix,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000}
    )

    if result.success:
        return result.x
    return x0


# ═══════════════════════════════════════
# 4. 等权组合
# ═══════════════════════════════════════
def equal_weight_portfolio(tickers):
    """等权组合"""
    n = len(tickers)
    return {
        "method": "equal_weight",
        "tickers": tickers,
        "weights": {t: 1.0/n for t in tickers},
        "weights_array": np.ones(n) / n,
        "note": "无优化，纯等权基准",
    }


# ═══════════════════════════════════════
# 5. 主优化函数
# ═══════════════════════════════════════
def optimize_portfolio(selected_stocks, ml_scores_dict=None, historical_returns_df=None,
                       period="1y", mode="kelly", risk_free_rate=0.05, max_leverage=1.0):
    """
    完整组合优化流程。

    Args:
        selected_stocks: list of ticker strings
        ml_scores_dict: {ticker: ml_score}, 用于预期收益先验 (可选)
        historical_returns_df: 预计算的历史收益率 DataFrame (可选)
        period: 数据周期 (当 historical_returns_df=None 时)
        mode: "kelly" | "risk_parity" | "equal_weight"
        risk_free_rate: 无风险利率
        max_leverage: 最大杠杆

    Returns:
        dict: 优化结果
    """
    if len(selected_stocks) < 2:
        return {
            "method": mode,
            "tickers": selected_stocks,
            "weights": {selected_stocks[0]: 1.0} if selected_stocks else {},
            "note": "标的数<2，单标满仓",
        }

    # ── 获取收益率数据 ──
    if historical_returns_df is not None:
        returns_df = historical_returns_df[selected_stocks].copy()
    else:
        import yfinance as yf
        print(f"    下载 {len(selected_stocks)} 只股票数据...")
        data = yf.download(selected_stocks, period=period, auto_adjust=True, progress=False)

        returns_list = []
        for t in selected_stocks:
            if isinstance(data.columns, pd.MultiIndex):
                if t in data.columns.levels[1]:
                    df = data.xs(t, axis=1, level=1)
                else:
                    continue
            else:
                df = data
            if "Close" in df.columns:
                close = df["Close"].values.astype(float).ravel()
            else:
                continue
            ret = pd.Series(np.diff(np.log(close)), index=df.index[1:], name=t)
            returns_list.append(ret)

        if not returns_list:
            raise ValueError("下载数据为空")

        returns_df = pd.concat(returns_list, axis=1).dropna(how="all")
        if returns_df.shape[1] < 2:
            return {"method": mode, "tickers": selected_stocks,
                    "weights": {t: 1.0 for t in selected_stocks}, "note": "仅一只股票有数据"}

    # 用有效列
    valid_tickers = returns_df.columns.tolist()
    n = len(valid_tickers)

    if n < 2:
        return {"method": mode, "tickers": valid_tickers,
                "weights": {t: 1.0 for t in valid_tickers}, "note": "仅一只股票有效"}

    # ── 预期收益 ──
    if ml_scores_dict:
        # 用 ML 评分做先验: 高评分 = 高预期收益
        scores = np.array([ml_scores_dict.get(t, 0.5) for t in valid_tickers])
        # 将 0~1 评分映射到 -5%~+20% 年化
        expected_returns = scores * 0.25 - 0.05
    else:
        # 历史均值 (年化)
        expected_returns = returns_df.mean().values * 252

    # ── 协方差矩阵 ──
    cov_matrix = ledoit_wolf_cov(returns_df) * 252

    # ── 优化 ──
    if mode == "kelly":
        weights = kelly_optimal_weights(expected_returns, cov_matrix,
                                         risk_free_rate, max_leverage)
    elif mode == "risk_parity":
        weights = risk_parity_portfolio(cov_matrix)
    else:  # equal_weight
        weights = np.ones(n) / n

    # ── 结果汇总 ──
    port_ret = np.dot(weights, expected_returns)
    port_var = np.dot(weights.T, np.dot(cov_matrix, weights))
    port_vol = np.sqrt(port_var)
    sharpe = (port_ret - risk_free_rate) / max(port_vol, 1e-8)

    # 边际风险贡献
    port_risk_contrib = np.dot(cov_matrix, weights) * weights / max(port_vol, 1e-8)

    allocation = {}
    for i, t in enumerate(valid_tickers):
        allocation[t] = {
            "weight": round(float(weights[i]), 4),
            "weight_pct": f"{weights[i]*100:.1f}%",
            "expected_return": round(float(expected_returns[i]), 4),
            "risk_contribution": round(float(port_risk_contrib[i]), 4),
        }

    return {
        "method": mode,
        "tickers": valid_tickers,
        "weights_array": weights,
        "allocation": allocation,
        "expected_return": round(float(port_ret), 4),
        "volatility": round(float(port_vol), 4),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_leverage": max_leverage,
        "n_assets": n,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
if __name__ == "__main__":
    args = sys.argv[1:]

    tickers = None
    if "--tickers" in args:
        idx = args.index("--tickers")
        if idx + 1 < len(args):
            tickers = [t.upper() for t in args[idx + 1].split(",")]

    if not tickers:
        print(__doc__)
        sys.exit(1)

    ml_scores = None
    if "--ml-scores" in args:
        idx = args.index("--ml-scores")
        if idx + 1 < len(args):
            scores_list = [float(s) for s in args[idx + 1].split(",")]
            ml_scores = {t: s for t, s in zip(tickers, scores_list)}

    use_kelly = "--risk-parity" not in args and "--equal-weight" not in args
    use_rp = "--risk-parity" in args
    use_ew = "--equal-weight" in args

    mode = "kelly"
    if use_rp:
        mode = "risk_parity"
    elif use_ew:
        mode = "equal_weight"

    print(f"\n{'='*70}")
    print(f"  组合优化 — {mode.upper()} 模式")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    print(f"  标的: {', '.join(tickers)}")
    if ml_scores:
        print(f"  ML 先验: {ml_scores}")
    print()

    try:
        result = optimize_portfolio(tickers, ml_scores_dict=ml_scores, mode=mode)

        if "note" in result:
            print(f"  ⚠️  {result['note']}\n")

        if "allocation" in result:
            print(f"  {'代码':>6} {'权重':>10} {'预期年化':>10} {'风险贡献':>10}")
            print(f"  {'-'*40}")
            for t in result["allocation"]:
                a = result["allocation"][t]
                print(f"  {t:>6} {a['weight_pct']:>10} {a['expected_return']:>+9.1%} "
                      f"{a['risk_contribution']:>10.4f}")

            print(f"\n  组合指标:")
            print(f"    预期年化收益: {result['expected_return']:+.2%}")
            print(f"    年化波动率:   {result['volatility']:.2%}")
            print(f"    夏普比率:     {result['sharpe_ratio']:.3f}")
            if result.get("max_leverage"):
                print(f"    最大杠杆:     {result['max_leverage']:.0%}")

        print(f"\n  ✅ 组合优化完成!")

    except Exception as e:
        print(f"\n  ❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
