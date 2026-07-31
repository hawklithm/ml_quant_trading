#!/usr/bin/env python3
"""
valuation_screener.py — 智能估值选股器

基于基本面分析，估算个股公允价值，找出当前价接近或低于估值价的标的。

估值方法：
  1. Graham Number  —  √(22.5 × EPS × BVPS)，巴菲特的老师格厄姆经典估值
  2. Comparable PE  —  行业内市盈率中位数 × 个股 EPS，相对同行溢价/折价
  3. FCF Yield      —  自由现金流收益率，隐含的合理市值
  4. PEG Ratio      —  PE / 增长率，低 PEG 暗示低估

用法:
  python valuation_screener.py --market US
  python valuation_screener.py --market HK --force-refresh
  python valuation_screener.py --market US --top 10 --detail

依赖:
  yfinance, numpy, pandas, scipy
  ml_optimized_picker_v5.py (获取 watchlist + sector 信息)
"""

import sys, os, json, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import zscore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_optimized_picker_v5 import US_WATCHLIST, HK_WATCHLIST, NAMES_HK

# ─── 配置 ───
RISK_FREE_RATE = 0.045        # 无风险利率 ~4.5%
EQUITY_RISK_PREMIUM = 0.055   # 股权风险溢价 ~5.5%
REQUIRED_RETURN = RISK_FREE_RATE + EQUITY_RISK_PREMIUM  # ~10%
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

# 估值方法权重 (等权)
METHOD_WEIGHTS = {
    "graham": 0.30,
    "comparable_pe": 0.30,
    "fcf_yield": 0.25,
    "peg_signal": 0.15,
}


def _sector_median(df_tickers, field):
    """计算行业内中位数，至少需要 3 个样本"""
    vals = [t.get(field) for t in df_tickers if t.get(field) is not None]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    if len(vals) < 3:
        return None
    return float(np.median(vals))


def fetch_fundamentals(tickers, force_refresh=False):
    """批量获取每只股票的基本面数据

    返回: [{ticker, sector, price, eps, bvps, fcf_ps, pe, forward_pe, pb, ...}]
    """
    results = []
    total = len(tickers)

    for i, t in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{total}] {t:<12} ... ")
        sys.stdout.flush()
        try:
            import yfinance as yf
            info = yf.Ticker(t).info
            if not info or info.get("regularMarketPrice") is None:
                print("无价格数据")
                continue

            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if price is None or price <= 0:
                print("价格无效")
                continue

            eps = info.get("trailingEps") or info.get("forwardEps")
            bvps = info.get("bookValue")
            fcf = info.get("freeCashflow")
            shares = info.get("sharesOutstanding")
            sector = info.get("sector") or "Other"
            industry = info.get("industry") or ""
            market_cap = info.get("marketCap")
            trailing_pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            pb = info.get("priceToBook")
            ps = info.get("priceToSalesTrailing12Months")
            ev_ebitda = info.get("enterpriseToEbitda")
            dividend_yield = info.get("dividendYield")
            earnings_growth = info.get("earningsGrowth")
            revenue_growth = info.get("revenueGrowth")
            roe = info.get("returnOnEquity")
            profit_margin = info.get("profitMargins")
            debt_equity = info.get("debtToEquity")
            target_mean = info.get("targetMeanPrice")

            fcf_ps = (fcf / shares) if (fcf and shares and shares > 0) else None

            record = {
                "ticker": t,
                "name": NAMES_HK.get(t, t),
                "sector": sector,
                "price": price,
                "eps": eps,
                "bvps": bvps,
                "fcf_ps": fcf_ps,
                "trailing_pe": trailing_pe,
                "forward_pe": forward_pe,
                "pb": pb,
                "ps": ps,
                "ev_ebitda": ev_ebitda,
                "dividend_yield": dividend_yield,
                "earnings_growth": earnings_growth,
                "revenue_growth": revenue_growth,
                "roe": roe,
                "profit_margin": profit_margin,
                "debt_equity": debt_equity,
                "market_cap": market_cap,
                "target_mean": target_mean,
                "fcf": fcf,
                "shares": shares,
            }

            # ─── 估值计算 ───

            # 1. Graham Number
            graham_fv = None
            if eps and bvps and eps > 0 and bvps > 0:
                graham_fv = np.sqrt(22.5 * eps * bvps)
            record["graham_fv"] = round(graham_fv, 2) if graham_fv else None
            record["graham_discount"] = (
                round((graham_fv - price) / graham_fv * 100, 1)
                if graham_fv and graham_fv > 0 else None
            )

            # 2. Comparable PE（同行比较）
            # 同行中位数在 aggregate 阶段计算
            record["comparable_pe_fv"] = None  # 稍后填充
            record["comparable_pe_discount"] = None

            # 3. FCF Yield 估值
            fcf_fv = None
            if fcf_ps and fcf_ps > 0:
                fcf_fv = fcf_ps / REQUIRED_RETURN
            record["fcf_fv"] = round(fcf_fv, 2) if fcf_fv else None
            record["fcf_discount"] = (
                round((fcf_fv - price) / fcf_fv * 100, 1)
                if fcf_fv and fcf_fv > 0 else None
            )

            # 4. PEG 信号 (越小越低估)
            peg = None
            usable_pe = forward_pe or trailing_pe
            usable_growth = earnings_growth or revenue_growth
            if usable_pe and usable_growth and usable_pe > 0 and usable_growth > 0:
                peg = usable_pe / (usable_growth * 100)
            record["peg"] = round(peg, 2) if peg else None

            results.append(record)
            print(f"✓ ${price:.1f} EPS={eps or 'N/A'}")

        except Exception as e:
            print(f"✗ {e}")

        # 防限流
        if i < total - 1:
            time.sleep(0.3)

    return results


def compute_valuation_scores(records):
    """对估值结果进行聚合评分

    1. 按行业计算 comparable PE 估值
    2. 每个方法计算 z-score
    3. 加权合成综合低估分数 (0-100)
    """
    if not records:
        return records

    df = pd.DataFrame(records)

    # ─── Comparable PE: 按行业中位数 ───
    sector_medians = {}
    for sector, group in df.groupby("sector"):
        pes = group["trailing_pe"].dropna()
        pes = pes[(pes > 0) & (pes < 100)]  # 过滤异常值
        if len(pes) >= 3:
            sector_medians[sector] = pes.median()

    for i, r in enumerate(records):
        s = r.get("sector", "Other")
        med_pe = sector_medians.get(s)
        eps = r.get("eps")
        if med_pe and eps and eps > 0:
            fv = med_pe * eps
            r["comparable_pe_fv"] = round(fv, 2)
            r["comparable_pe_discount"] = round((fv - r["price"]) / fv * 100, 1) if fv > 0 else None

    # ─── 构造评分矩阵 ───
    score_cols = []
    score_labels = []

    # Graham discount (正数 = 折价/低估)
    graham_discounts = [r.get("graham_discount") for r in records]
    if any(d is not None for d in graham_discounts):
        vals = np.array([d if d is not None else np.nan for d in graham_discounts], dtype=float)
        z = zscore(vals, nan_policy="omit")
        for i, r in enumerate(records):
            r["_z_graham"] = z[i] if not np.isnan(z[i]) else 0
        score_cols.append("_z_graham")
        score_labels.append("graham")

    # Comparable PE discount
    pe_discounts = [r.get("comparable_pe_discount") for r in records]
    if any(d is not None for d in pe_discounts):
        vals = np.array([d if d is not None else np.nan for d in pe_discounts], dtype=float)
        z = zscore(vals, nan_policy="omit")
        for i, r in enumerate(records):
            r["_z_comparable_pe"] = z[i] if not np.isnan(z[i]) else 0
        score_cols.append("_z_comparable_pe")
        score_labels.append("comparable_pe")

    # FCF discount
    fcf_discounts = [r.get("fcf_discount") for r in records]
    if any(d is not None for d in fcf_discounts):
        vals = np.array([d if d is not None else np.nan for d in fcf_discounts], dtype=float)
        z = zscore(vals, nan_policy="omit")
        for i, r in enumerate(records):
            r["_z_fcf"] = z[i] if not np.isnan(z[i]) else 0
        score_cols.append("_z_fcf")
        score_labels.append("fcf_yield")

    # PEG (越小越好 → 取负号后z-score)
    pegs = [r.get("peg") for r in records]
    if any(p is not None for p in pegs):
        vals = np.array([p if p is not None else np.nan for p in pegs], dtype=float)
        # PEG < 1 为低估，取 -PEG 让低PEG得高分
        z = zscore(-vals, nan_policy="omit")
        for i, r in enumerate(records):
            r["_z_peg"] = z[i] if not np.isnan(z[i]) else 0
        score_cols.append("_z_peg")
        score_labels.append("peg_signal")

    # ─── 合成综合分数 ───
    for r in records:
        total_weight = 0
        weighted_sum = 0
        for col, label in zip(score_cols, score_labels):
            w = METHOD_WEIGHTS.get(label, 0.2)
            v = r.get(col, 0)
            if v != 0 or col in r:  # 至少参与过计算
                weighted_sum += v * w
                total_weight += w
        composite = weighted_sum / total_weight if total_weight > 0 else 0
        # 映射到 0-100 分: z-score 在 [-3, 3] 范围内有意义
        r["value_score"] = round(min(100, max(0, (composite / 3 + 1) * 50)), 1)

    return records


def print_valuation_report(records, market, top_n=20):
    """输出估值报告"""
    if not records:
        print("\n  没有有效的估值数据")
        return

    records.sort(key=lambda r: r.get("value_score", 0), reverse=True)

    label = "港股" if market == "HK" else "美股"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*95}")
    print(f"  📊 {label}估值选股报告")
    print(f"  {now}")
    print(f"{'='*95}")

    # 统计概要
    n_valued = sum(1 for r in records if r.get("value_score", 0) > 0)
    n_graham = sum(1 for r in records if r.get("graham_fv"))
    n_fcf = sum(1 for r in records if r.get("fcf_fv"))
    n_pe_comp = sum(1 for r in records if r.get("comparable_pe_fv"))
    n_peg = sum(1 for r in records if r.get("peg"))
    print(f"  评估: {len(records)} 只 | 估值有效: {n_valued} 只")
    print(f"  方法覆盖率: Graham={n_graham} ComparablePE={n_pe_comp} FCF={n_fcf} PEG={n_peg}")
    print(f"  估值方法权重: {', '.join(f'{k}={v:.0%}' for k,v in METHOD_WEIGHTS.items())}")

    # Top N 排名
    print(f"  {'#':>3} {'代码':>10} {'名称':>6} {'现价':>8} {'估值分':>6} {'Graham':>8} {'PE估值':>8} {'FCF估值':>8} {'PEG' :>5} {'折/溢%':>6}")
    print(f"  {'-'*80}")
    for i, r in enumerate(records[:top_n]):
        score = r.get("value_score", 0)
        price = r.get("price", 0)

        # 计算平均折价（多方法平均）
        discounts = []
        for fld in ["graham_discount", "comparable_pe_discount", "fcf_discount"]:
            v = r.get(fld)
            if v is not None:
                discounts.append(v)
        avg_discount = round(np.mean(discounts), 1) if discounts else None
        if avg_discount is not None:
            if avg_discount > 0:
                discount_str = f"折{avg_discount:+.1f}"
            else:
                discount_str = f"溢{avg_discount:+.1f}"
        else:
            discount_str = "N/A"

        graham_str = f"${r['graham_fv']:.0f}" if r.get("graham_fv") else "N/A"
        pe_str = f"${r['comparable_pe_fv']:.0f}" if r.get("comparable_pe_fv") else "N/A"
        fcf_str = f"${r['fcf_fv']:.0f}" if r.get("fcf_fv") else "N/A"
        peg_str = f"{r['peg']:.1f}" if r.get("peg") else "N/A"

        name = r.get("name", r["ticker"])
        score_str = f"{score:.0f}" if score > 0 else "-"

        print(f"  {i+1:>3} {r['ticker']:>10} {name:>6} ${price:>6.1f} {score_str:>6} {graham_str:>8} {pe_str:>8} {fcf_str:>8} {peg_str:>5} {discount_str:>6}")

    # ─── 深度个股分析 (顶部 3 只) ───
    print(f"\n{'='*95}")
    print(f"  详细分析 Top 3")
    print(f"{'='*95}")
    for r in records[:3]:
        t = r["ticker"]
        name = r.get("name", t)
        sector = r.get("sector", "N/A")
        price = r.get("price", 0)

        print(f"\n  ── {t} ({name}) | {sector} | 现价 ${price:.1f} ──")
        print(f"     估值综合分: {r.get('value_score', 0):.0f}/100")

        if r.get("graham_fv"):
            d = r.get("graham_discount", 0)
            sign = "折价" if d > 0 else "溢价"
            print(f"     Graham数: ${r['graham_fv']:.1f} ({sign} {abs(d):.1f}%)")

        if r.get("comparable_pe_fv"):
            d = r.get("comparable_pe_discount", 0)
            med_pe = None
            for rr in records:
                if rr.get("sector") == sector and rr.get("trailing_pe") and rr["trailing_pe"] > 0 and rr["trailing_pe"] < 100:
                    # find the sector median
                    pass
            sign = "折价" if d > 0 else "溢价"
            print(f"     同行PE估值: ${r['comparable_pe_fv']:.1f} ({sign} {abs(d):.1f}%)")

        if r.get("fcf_fv"):
            d = r.get("fcf_discount", 0)
            sign = "折价" if d > 0 else "溢价"
            print(f"     FCF估值: ${r['fcf_fv']:.1f} ({sign} {abs(d):.1f}%, 要求回报率{REQUIRED_RETURN:.0%})")

        if r.get("peg"):
            peg_label = "低估" if r["peg"] < 1 else "合理" if r["peg"] < 2 else "高估"
            print(f"     PEG: {r['peg']:.2f} ({peg_label})")

        # 基本面快照
        fundamentals = []
        if r.get("trailing_pe"):
            fundamentals.append(f"PE={r['trailing_pe']:.1f}")
        if r.get("forward_pe"):
            fundamentals.append(f"FwdPE={r['forward_pe']:.1f}")
        if r.get("pb"):
            fundamentals.append(f"PB={r['pb']:.1f}")
        if r.get("ev_ebitda"):
            fundamentals.append(f"EV/EBITDA={r['ev_ebitda']:.1f}")
        dy = r.get("dividend_yield")
        if dy is not None:
            # yfinance dividend_yield 有时是小数(0.02=2%)，有时已是百分数(2.0)
            dy_pct = dy * 100 if dy < 1 else dy
            if dy_pct < 50:  # 过滤异常值
                fundamentals.append(f"股息率={dy_pct:.2f}%")
        if r.get("roe"):
            fundamentals.append(f"ROE={r['roe']*100:.1f}%")
        if r.get("profit_margin"):
            fundamentals.append(f"利润率={r['profit_margin']*100:.1f}%")
        if r.get("market_cap"):
            fundamentals.append(f"市值={r['market_cap']/1e9:.1f}B")

        if fundamentals:
            print(f"     基本面: {' | '.join(fundamentals)}")

        if r.get("target_mean"):
            diff = (r["target_mean"] - price) / price * 100
            print(f"     分析师目标价: ${r['target_mean']:.1f} (潜在{'上涨' if diff > 0 else '下跌'}{abs(diff):.1f}%)")


def save_results(records, market):
    """保存估值结果到 JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe = [r for r in records if r.get("value_score", 0) > 0]
    path = os.path.join(CACHE_DIR, f"valuation_{market}_{timestamp}.json")
    with open(path, "w") as f:
        json.dump({
            "market": market,
            "generated_at": datetime.now().isoformat(),
            "total": len(records),
            "valued": len(safe),
            "results": safe,
        }, f, indent=2, default=str)
    print(f"\n  💾 已保存: {path}")
    return path


# ═══════════════════════════════════════
# CLI
# ═══════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="智能估值选股器")
    parser.add_argument("--market", required=True, choices=["US", "HK"])
    parser.add_argument("--top", type=int, default=20, help="Top N 只展示")
    parser.add_argument("--force-refresh", action="store_true", help="强制刷新 yfinance 数据")
    parser.add_argument("--save", action="store_true", help="保存结果到 JSON")
    args = parser.parse_args()

    tickers = US_WATCHLIST if args.market == "US" else HK_WATCHLIST
    label = "港股" if args.market == "HK" else "美股"

    print(f"📊 {label}估值分析 — {len(tickers)} 只标的")
    print(f"  请求回报率: {REQUIRED_RETURN:.0%} = 无风险{RISK_FREE_RATE:.1%} + ERP{EQUITY_RISK_PREMIUM:.1%}")

    records = fetch_fundamentals(tickers, force_refresh=args.force_refresh)
    if not records:
        print("❌ 没有获取到基本面数据")
        sys.exit(1)

    records = compute_valuation_scores(records)
    print_valuation_report(records, args.market, top_n=args.top)

    if args.save:
        save_results(records, args.market)
