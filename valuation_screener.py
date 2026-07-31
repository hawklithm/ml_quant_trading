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
  python valuation_screener.py --market US --top 10 --save

依赖:
  requests, numpy, pandas, scipy
  ml_optimized_picker_v5.py (获取 watchlist + sector 信息)
"""

import sys, os, json, time, hashlib, tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import zscore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ml_optimized_picker_v5 import US_WATCHLIST, HK_WATCHLIST, NAMES_HK


def _configure_console():
    """Keep CLI/report output usable on Windows consoles with non-UTF8 code pages."""
    stream = getattr(sys, "stdout", None)
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError, ValueError):
            pass

# ─── 配置：按市场显式区分估值假设 ───
MARKET_ASSUMPTIONS = {
    "US": {"risk_free_rate": 0.045, "equity_risk_premium": 0.055, "cost_of_debt": 0.045, "tax_rate": 0.21, "terminal_growth": 0.020},
    "HK": {"risk_free_rate": 0.035, "equity_risk_premium": 0.060, "cost_of_debt": 0.045, "tax_rate": 0.16, "terminal_growth": 0.020},
}
RISK_FREE_RATE = MARKET_ASSUMPTIONS["US"]["risk_free_rate"]
EQUITY_RISK_PREMIUM = MARKET_ASSUMPTIONS["US"]["equity_risk_premium"]
REQUIRED_RETURN = RISK_FREE_RATE + EQUITY_RISK_PREMIUM
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL_SECONDS = 24 * 60 * 60
CACHE_SCHEMA_VERSION = 4
MODEL_VERSION = "valuation_free_v1"

# 估值方法权重 (等权)
METHOD_WEIGHTS = {
    "graham": 0.20,
    "comparable_pe": 0.25,
    "dcf": 0.25,
    "peg_signal": 0.10,
    "pb_signal": 0.20,
}

VALUATION_POLICIES = {
    "financial": {"methods": {"comparable_pe", "pb_signal"}, "keywords": ("bank", "insurance", "financial", "金融", "银行", "保险")},
    "reit": {"methods": {"dcf"}, "keywords": ("reit", "real estate", "房地产", "地产")},
    "cyclical": {"methods": {"comparable_pe", "dcf"}, "keywords": ("energy", "materials", "mining", "oil", "gas", "能源", "材料")},
    "default": {"methods": set(METHOD_WEIGHTS), "keywords": ()},
}


def _market_assumptions(market):
    return MARKET_ASSUMPTIONS.get(str(market).upper(), MARKET_ASSUMPTIONS["US"])


def _valuation_policy(industry, sector):
    text = f"{industry or ''} {sector or ''}".lower()
    for name, policy in VALUATION_POLICIES.items():
        if name != "default" and any(keyword.lower() in text for keyword in policy["keywords"]):
            return name, sorted(policy["methods"])
    return "default", sorted(VALUATION_POLICIES["default"]["methods"])


def _wacc(market_cap, total_debt, assumptions, cost_of_equity):
    """Calculate market-value WACC; fall back to cost of equity if capital data is absent."""
    equity = float(market_cap or 0)
    debt = float(total_debt or 0)
    if equity <= 0 or debt < 0:
        return cost_of_equity
    total = equity + debt
    return (equity / total) * cost_of_equity + (debt / total) * assumptions["cost_of_debt"] * (1 - assumptions["tax_rate"])


def _fundamentals_cache_path(market):
    return os.path.join(CACHE_DIR, f"valuation_fundamentals_{str(market).upper()}.json")


def _read_fundamentals_cache(market, tickers):
    path = _fundamentals_cache_path(market)
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > CACHE_TTL_SECONDS:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION or payload.get("model_version") != MODEL_VERSION:
            return None
        records = payload.get("records", [])
        wanted = set(tickers)
        if wanted and not wanted.issubset({record.get("ticker") for record in records}):
            return None
        fetch_fundamentals.last_errors = payload.get("errors", [])
        return records
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write_fundamentals_cache(market, records, errors=None):
    path = _fundamentals_cache_path(market)
    directory = os.path.dirname(path)
    fd, temporary = tempfile.mkstemp(prefix="valuation_", suffix=".tmp", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": CACHE_SCHEMA_VERSION, "model_version": MODEL_VERSION, "market": market, "fetched_at": datetime.now().isoformat(), "records": records, "errors": errors or []}, handle, ensure_ascii=False, indent=2, default=str)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _dcf_per_share(fcf_ps, growth, discount_rate, terminal_growth, net_debt_ps=0.0, dilution_rate=0.0, cash_flow_type="fcff"):
    """Five-year DCF per share; FCFF subtracts net debt, FCFE does not."""
    if fcf_ps is None or not np.isfinite(fcf_ps) or fcf_ps <= 0:
        return None
    growth = float(np.clip(growth if growth is not None and np.isfinite(growth) else 0.0, -0.05, 0.15))
    if discount_rate <= terminal_growth:
        return None
    value = 0.0
    for year in range(1, 6):
        value += fcf_ps * ((1 + growth) ** year) / ((1 + discount_rate) ** year)
    terminal = fcf_ps * ((1 + growth) ** 5) * (1 + terminal_growth) / (discount_rate - terminal_growth)
    enterprise_value = value + terminal / ((1 + discount_rate) ** 5)
    equity_value = enterprise_value - float(net_debt_ps or 0.0) if cash_flow_type.lower() == "fcff" else enterprise_value
    return equity_value / ((1 + max(float(dilution_rate), 0.0)) ** 5) if equity_value > 0 else None


def _normalized_fcf_per_share(cashflow, fallback_fcf, shares):
    """Use the median of available annual FCF observations, with a safe latest-value fallback."""
    if shares is None or not np.isfinite(shares) or shares <= 0:
        return None, "unavailable"
    values = []
    if isinstance(cashflow, pd.DataFrame) and not cashflow.empty:
        for label in ("Free Cash Flow", "FreeCashFlow"):
            if label in cashflow.index:
                values = pd.to_numeric(cashflow.loc[label], errors="coerce").dropna().tolist()
                break
        if not values and "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
            operating = pd.to_numeric(cashflow.loc["Operating Cash Flow"], errors="coerce")
            capex = pd.to_numeric(cashflow.loc["Capital Expenditure"], errors="coerce")
            values = (operating + capex).dropna().tolist()
    values = [float(value) for value in values if np.isfinite(value)]
    if len(values) >= 2:
        normalized = float(np.median(values)) / shares
        return (normalized, "annual_median") if normalized > 0 else (None, "annual_median_nonpositive")
    if fallback_fcf is not None and np.isfinite(fallback_fcf) and fallback_fcf > 0:
        return float(fallback_fcf) / shares, "latest"
    return None, "unavailable"


def _growth_percent(value):
    """Normalize provider decimal/percentage growth to percentage points."""
    if value is None or not np.isfinite(value) or value <= 0:
        return None
    return float(_growth_decimal(value) * 100) if _growth_decimal(value) is not None else None


def _growth_decimal(value):
    """Normalize decimal/percentage growth to a decimal fraction."""
    if value is None or not np.isfinite(value) or value <= -1 or value > 100:
        return None
    return float(value if abs(value) <= 1 else value / 100)


def _days_since(value):
    """Convert provider timestamps/ISO strings to age in days."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float, np.integer, np.floating)):
            timestamp = pd.to_datetime(value, unit="s", errors="coerce")
        else:
            timestamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(timestamp):
            return None
        if getattr(timestamp, "tzinfo", None) is not None:
            timestamp = timestamp.tz_localize(None)
        return max(0, (datetime.now() - timestamp.to_pydatetime()).days)
    except (TypeError, ValueError, OverflowError):
        return None


def _ensure_valuation_fields(records, market):
    """Build intrinsic-value fields for records from any normalized provider."""
    assumptions = _market_assumptions(market)
    required_return = assumptions["risk_free_rate"] + assumptions["equity_risk_premium"]
    for record in records:
        record.setdefault("required_return", required_return)
        record.setdefault("terminal_growth", assumptions["terminal_growth"])
        price = record.get("price")
        eps = record.get("trailing_eps", record.get("eps"))
        bvps = record.get("bvps")
        if record.get("graham_fv") is None and eps and bvps and eps > 0 and bvps > 0:
            record["graham_fv"] = round(float(np.sqrt(22.5 * eps * bvps)), 2)
        if record.get("graham_discount") is None and record.get("graham_fv") and price:
            record["graham_discount"] = round((record["graham_fv"] - price) / record["graham_fv"] * 100, 1)
        if record.get("fcf_fv") is None and record.get("fcf_ps") and record["fcf_ps"] > 0 and price:
            growth = _growth_decimal(record.get("earnings_growth") or record.get("revenue_growth"))
            fcf_fv = _dcf_per_share(
                record["fcf_ps"], growth, required_return, assumptions["terminal_growth"],
                cash_flow_type=record.get("cash_flow_type", "fcfe"),
            )
            record["fcf_fv"] = round(fcf_fv, 2) if fcf_fv else None
            record["dcf_fv"] = record["fcf_fv"]
        if record.get("fcf_discount") is None and record.get("fcf_fv") and price:
            record["fcf_discount"] = round((record["fcf_fv"] - price) / record["fcf_fv"] * 100, 1)
        if record.get("peg") is None:
            pe = record.get("forward_pe") or record.get("trailing_pe")
            growth = _growth_percent(record.get("earnings_growth") or record.get("revenue_growth"))
            if pe and growth and pe > 0 and growth > 0:
                record["peg"] = round(pe / growth, 2)
    return records
def _sector_median(df_tickers, field):
    """计算行业内中位数，至少需要 3 个样本"""
    vals = [t.get(field) for t in df_tickers if t.get(field) is not None]
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    if len(vals) < 3:
        return None
    return float(np.median(vals))


def fetch_fundamentals(tickers, force_refresh=False, market="US"):
    """批量获取每只股票的基本面数据

    返回: [{ticker, sector, price, eps, bvps, fcf_ps, pe, forward_pe, pb, ...}]
    """
    market = str(market).upper()
    if not force_refresh:
        cached = _read_fundamentals_cache(market, tickers)
        if cached is not None:
            print(f"  使用估值基本面缓存: {len(cached)} 只")
            return cached

    # The valuation path intentionally uses the free provider stack by default.
    # Set QUANT_DATA_PROVIDER=yahoo only for legacy compatibility during migration.
    if os.getenv("QUANT_DATA_PROVIDER", "free").lower() != "yahoo":
        from quant.data.free_provider import fetch_free_fundamentals
        results, errors = fetch_free_fundamentals(tickers, market)
        results = _ensure_valuation_fields(results, market)
        _write_fundamentals_cache(market, results, errors)
        fetch_fundamentals.last_errors = errors
        return results

    results = []
    errors = []
    total = len(tickers)
    assumptions = _market_assumptions(market)
    required_return = assumptions["risk_free_rate"] + assumptions["equity_risk_premium"]

    for i, t in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{total}] {t:<12} ... ")
        sys.stdout.flush()

        # 指数退避重试
        max_retries = 3
        info = None
        for attempt in range(max_retries):
            try:
                import yfinance as yf
                if attempt > 0:
                    backoff = (2 ** attempt) * 2.0
                    sys.stdout.write(f"  等待{backoff:.0f}s重试... ")
                    sys.stdout.flush()
                    time.sleep(backoff)
                info = yf.Ticker(t).info
                if info and info.get("regularMarketPrice") is not None:
                    break
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    continue
                errors.append({"ticker": t, "status": "error", "reason": last_error, "data_asof": datetime.now().isoformat(), "market": market})
                print(f"✗ {last_error}")
                info = None
                break

        try:
            if info is None or info.get("regularMarketPrice") is None:
                print("无价格数据")
                continue

            price = info.get("regularMarketPrice") or info.get("currentPrice") or info.get("previousClose")
            if price is None or price <= 0:
                errors.append({"ticker": t, "status": "missing_price", "reason": "no valid market price", "data_asof": datetime.now().isoformat(), "market": market})
                print("价格无效")
                continue

            trailing_eps = info.get("trailingEps")
            forward_eps = info.get("forwardEps")
            eps = trailing_eps if trailing_eps is not None else forward_eps
            bvps = info.get("bookValue")
            fcf = info.get("freeCashflow")
            shares = info.get("sharesOutstanding")
            sector = info.get("sector") or "Other"
            industry = info.get("industry") or ""
            valuation_group = industry or sector
            valuation_policy, allowed_methods = _valuation_policy(industry, sector)
            market_cap = info.get("marketCap")
            total_debt = info.get("totalDebt")
            cash_total = info.get("totalCash")
            trailing_pe = info.get("trailingPE")
            forward_pe = info.get("forwardPE")
            comparable_pe = trailing_pe if trailing_pe is not None and trailing_eps is not None else forward_pe
            comparable_eps = trailing_eps if trailing_pe is not None and trailing_eps is not None else forward_eps
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
            financial_period_end = info.get("mostRecentQuarter") or info.get("lastFiscalYearEnd")

            try:
                cashflow = ticker.cashflow
            except Exception:
                cashflow = None
            fcf_ps, fcf_source = _normalized_fcf_per_share(cashflow, fcf, shares)
            cash_flow_type = "fcfe_proxy"

            record = {
                "ticker": t,
                "name": NAMES_HK.get(t, t),
                "sector": sector,
                "industry": industry,
                "valuation_group": valuation_group,
                "price": price,
                "eps": eps,
                "trailing_eps": trailing_eps,
                "forward_eps": forward_eps,
                "bvps": bvps,
                "fcf_ps": fcf_ps,
                "fcf_source": fcf_source,
                "cash_flow_type": cash_flow_type,
                "trailing_pe": trailing_pe,
                "forward_pe": forward_pe,
                "comparable_pe": comparable_pe,
                "comparable_eps": comparable_eps,
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
                "total_debt": total_debt,
                "cash_total": cash_total,
                "target_mean": target_mean,
                "fcf": fcf,
                "shares": shares,
                "data_asof": datetime.now().isoformat(),
                "price_asof": datetime.now().isoformat(),
                "financial_period_end": financial_period_end,
                "point_in_time_ready": False,
                "point_in_time_status": "snapshot_only",
                "market": market,
                "required_return": required_return,
                "terminal_growth": assumptions["terminal_growth"],
                "valuation_policy": valuation_policy,
                "allowed_valuation_methods": allowed_methods,
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
            growth_for_dcf = _growth_decimal(earnings_growth if earnings_growth is not None else revenue_growth)
            # Yahoo freeCashflow is not guaranteed to be FCFF. Treat it as an
            # FCFE proxy and do not subtract net debt a second time.
            net_debt_ps = 0.0
            dcf_rate = required_return
            if fcf_ps and fcf_ps > 0:
                fcf_fv = _dcf_per_share(
                    fcf_ps, growth_for_dcf, dcf_rate, assumptions["terminal_growth"],
                    net_debt_ps=net_debt_ps,
                    cash_flow_type="fcfe",
                )
            record["fcf_fv"] = round(fcf_fv, 2) if fcf_fv else None
            record["dcf_fv"] = record["fcf_fv"]
            record["fcf_discount"] = (
                round((fcf_fv - price) / fcf_fv * 100, 1)
                if fcf_fv and fcf_fv > 0 else None
            )

            # 4. PEG 信号 (越小越低估)
            peg = None
            usable_pe = forward_pe if forward_pe is not None and np.isfinite(forward_pe) and forward_pe > 0 else trailing_pe
            usable_growth = earnings_growth if earnings_growth is not None and np.isfinite(earnings_growth) and earnings_growth > 0 else revenue_growth
            if usable_pe and usable_growth and usable_pe > 0 and usable_growth > 0:
                growth_percent = _growth_percent(usable_growth)
                peg = usable_pe / growth_percent if growth_percent and growth_percent > 0 else None
            record["peg"] = round(peg, 2) if peg else None

            results.append(record)
            print(f"✓ ${price:.1f} EPS={eps or 'N/A'}")

        except Exception as e:
            errors.append({"ticker": t, "status": "error", "reason": str(e), "data_asof": datetime.now().isoformat(), "market": market})
            print(f"✗ {e}")

        # 防限流：港股限流更严，加大间隔
        if i < total - 1:
            time.sleep(1.0 if market == "HK" else 0.5)

    _write_fundamentals_cache(market, results, errors)
    fetch_fundamentals.last_errors = errors
    return results


def compute_valuation_scores(records, market="US"):
    """对估值结果进行聚合评分

    1. 按行业计算 comparable PE 估值
    2. 每个方法计算 z-score
    3. 加权合成综合低估分数 (0-100)
    """
    if not records:
        return records

    _ensure_valuation_fields(records, market)

    df = pd.DataFrame(records)
    for record in records:
        if "allowed_valuation_methods" not in record:
            policy, methods = _valuation_policy(record.get("industry", ""), record.get("sector", "Other"))
            record["valuation_policy"] = policy
            record["allowed_valuation_methods"] = methods

    # ─── Comparable PE: 按行业中位数 ───
    sector_medians = {}
    group_column = "valuation_group" if "valuation_group" in df.columns else "sector"
    pe_column = "comparable_pe" if "comparable_pe" in df.columns else "trailing_pe"
    eps_column = "comparable_eps" if "comparable_eps" in df.columns else "eps"
    for group_name, group in df.groupby(group_column):
        pes = pd.to_numeric(group[pe_column], errors="coerce").dropna()
        pes = pes[(pes > 0) & (pes < 100)]  # 过滤异常值
        if len(pes) >= 3:
            sector_medians[group_name] = pes.median()

    for i, r in enumerate(records):
        s = r.get(group_column, r.get("sector", "Other"))
        med_pe = sector_medians.get(s)
        eps = r.get("eps")
        own_pe = r.get("trailing_pe")
        price = r.get("price")
        if price is None or not np.isfinite(price) or price <= 0:
            continue
        peer_values = df.loc[(df[group_column] == s) & (df.index != i), pe_column].dropna()
        peer_values = peer_values[(peer_values > 0) & (peer_values < 100)]
        peer_median = peer_values.median() if len(peer_values) >= 2 else med_pe
        if peer_median and eps and eps > 0:
            fv = peer_median * eps
            r["comparable_pe_fv"] = round(fv, 2)
            r["comparable_pe_discount"] = round((fv - r["price"]) / fv * 100, 1) if fv > 0 else None

    # Financial companies are better compared by P/B than by Graham or DCF.
    pb_medians = {}
    if "pb" in df.columns and "bvps" in df.columns:
        for group_name, group in df.groupby(group_column):
            pbs = pd.to_numeric(group["pb"], errors="coerce").dropna()
            pbs = pbs[(pbs > 0) & (pbs < 20)]
            if len(pbs) >= 3:
                pb_medians[group_name] = float(pbs.median())
    for r in records:
        if r.get("valuation_policy") == "financial":
            median_pb = pb_medians.get(r.get(group_column, r.get("sector", "Other")))
            if median_pb and r.get("bvps") and r.get("bvps") > 0:
                r["pb_fv"] = round(median_pb * r["bvps"], 2)
                r["pb_discount"] = round((r["pb_fv"] - r["price"]) / r["pb_fv"] * 100, 1)

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
        score_labels.append("dcf")

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

    pb_discounts = [r.get("pb_discount") for r in records]
    if any(value is not None for value in pb_discounts):
        vals = np.array([value if value is not None else np.nan for value in pb_discounts], dtype=float)
        z = zscore(vals, nan_policy="omit")
        for i, r in enumerate(records):
            r["_z_pb"] = z[i] if not np.isnan(z[i]) else 0
        score_cols.append("_z_pb")
        score_labels.append("pb_signal")

    # ─── 合成综合分数 ───
    for r in records:
        total_weight = 0
        weighted_sum = 0
        for col, label in zip(score_cols, score_labels):
            allowed_methods = set(r.get("allowed_valuation_methods", METHOD_WEIGHTS))
            if label not in allowed_methods:
                continue
            w = METHOD_WEIGHTS.get(label, 0.2)
            method_fields = {"_z_graham": "graham_discount", "_z_comparable_pe": "comparable_pe_discount", "_z_fcf": "fcf_discount", "_z_peg": "peg", "_z_pb": "pb_discount"}
            v = r.get(col, 0)
            if r.get(method_fields.get(col)) is not None:
                weighted_sum += v * w
                total_weight += w
        composite = weighted_sum / total_weight if total_weight > 0 else 0
        # 映射到 0-100 分: z-score 在 [-3, 3] 范围内有意义
        r["value_score"] = round(min(100, max(0, (composite / 3 + 1) * 50)), 1)

        discount_methods = {
            "graham_discount": "graham",
            "comparable_pe_discount": "comparable_pe",
            "fcf_discount": "dcf",
            "pb_discount": "pb_signal",
        }
        allowed_methods = set(r.get("allowed_valuation_methods", METHOD_WEIGHTS))
        discounts = [
            r.get(field) for field, method in discount_methods.items()
            if method in allowed_methods
        ]
        discounts = [float(value) for value in discounts if value is not None and np.isfinite(value)]
        r["valuation_coverage"] = len(discounts)
        r["average_discount"] = round(float(np.mean(discounts)), 1) if discounts else None
        roe = r.get("roe")
        margin = r.get("profit_margin")
        debt = r.get("debt_equity")
        policy = r.get("valuation_policy", "default")
        quality_values = [roe, debt] if policy == "reit" else [roe, margin, debt]
        quality_complete = all(value is not None and np.isfinite(value) for value in quality_values)
        quality_checks = (
            [roe > 0, margin > 0, debt < 2000] if policy == "financial" else
            [roe > -0.10, debt < 800] if policy == "reit" else
            [roe > 0, margin > 0, debt < 300]
        ) if quality_complete else []
        r["quality_status"] = "complete" if quality_complete else "incomplete"
        r["quality_pass"] = bool(quality_complete and all(quality_checks))
        pe = r.get("forward_pe") if r.get("forward_pe") is not None else r.get("trailing_pe")
        r["hard_valuation_pass"] = bool(
            (pe is None or not np.isfinite(pe) or pe <= 80)
            and (r.get("peg") is None or r.get("peg") <= 3)
        )
        r["eligible"] = bool(
            r["valuation_coverage"] >= (1 if policy == "reit" else 2)
            and r["average_discount"] is not None
            and 10 <= r["average_discount"] <= 50
            and r["quality_pass"]
            and r["hard_valuation_pass"]
        )
        allowed = set(r.get("allowed_valuation_methods", METHOD_WEIGHTS))
        r["data_completeness"] = round(min(1.0, r["valuation_coverage"] / max(len(allowed), 1)), 2)
        r["method_dispersion"] = round(float(np.std(discounts)), 2) if len(discounts) >= 2 else None
        r["method_agreement"] = round(1.0 / (1.0 + r["method_dispersion"] / 25.0), 2) if r["method_dispersion"] is not None else 0.5
        r["price_age_days"] = _days_since(r.get("price_asof") or r.get("data_asof"))
        r["financial_age_days"] = _days_since(r.get("financial_period_end"))
        financial_age = r["financial_age_days"]
        r["data_freshness_status"] = (
            "fresh" if financial_age is not None and financial_age <= 180 else
            "stale" if financial_age is not None and financial_age <= 365 else
            "unknown"
        )
        freshness_factor = 1.0 if financial_age is not None and financial_age <= 180 else 0.75 if financial_age is not None and financial_age <= 365 else 0.5
        r["valuation_confidence"] = round(r["data_completeness"] * r["method_agreement"] * freshness_factor, 2)

    return records


def print_valuation_report(records, market, top_n=20):
    """输出估值报告"""
    _configure_console()
    if not records:
        print("\n  没有有效的估值数据")
        return

    records.sort(key=lambda r: (bool(r.get("eligible")), r.get("value_score", 0)), reverse=True)

    label = "港股" if market == "HK" else "美股"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    print(f"\n{'='*95}")
    print(f"  📊 {label}估值选股报告")
    print(f"  {now}")
    print(f"{'='*95}")

    # 统计概要
    n_valued = sum(1 for r in records if r.get("value_score", 0) > 0)
    n_eligible = sum(1 for r in records if r.get("eligible"))
    n_graham = sum(1 for r in records if r.get("graham_fv"))
    n_fcf = sum(1 for r in records if r.get("fcf_fv"))
    n_pe_comp = sum(1 for r in records if r.get("comparable_pe_fv"))
    n_peg = sum(1 for r in records if r.get("peg"))
    print(f"  评估: {len(records)} 只 | 估值有效: {n_valued} 只 | 安全边际合格: {n_eligible} 只")
    print(f"  方法覆盖率: Graham={n_graham} ComparablePE={n_pe_comp} FCF={n_fcf} PEG={n_peg}")
    print(f"  估值方法权重: {', '.join(f'{k}={v:.0%}' for k,v in METHOD_WEIGHTS.items())}")

    # Top N 排名
    print(f"  {'#':>3} {'代码':>10} {'名称':>6} {'现价':>8} {'估值分':>6} {'Graham':>8} {'PE估值':>8} {'FCF估值':>8} {'PEG' :>5} {'折/溢%':>6}")
    print(f"  {'-'*80}")
    for i, r in enumerate(records[:top_n]):
        score = r.get("value_score", 0)
        price = r.get("price", 0)
        if price is None or price <= 0:
            continue

        # 计算平均折价（多方法平均）
        discounts = []
        for fld in ["graham_discount", "comparable_pe_discount", "fcf_discount", "pb_discount"]:
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

        status = "OK" if r.get("eligible") else "WATCH"
        confidence = r.get("valuation_confidence", 0)
        print(f"  {i+1:>3} {r['ticker']:>10} {name:>6} ${price:>6.1f} {score_str:>6} {graham_str:>8} {pe_str:>8} {fcf_str:>8} {peg_str:>5} {discount_str:>6} {confidence:>4.0%} {status:>5}")

    # ─── 深度个股分析 (顶部 3 只) ───
    print(f"\n{'='*95}")
    print(f"  详细分析 Top 3")
    print(f"{'='*95}")
    for r in records[:3]:
        t = r["ticker"]
        name = r.get("name", t)
        sector = r.get("sector", "N/A")
        price = r.get("price", 0)
        if price is None or price <= 0:
            continue

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
            print(f"     DCF估值: ${r['fcf_fv']:.1f} ({sign} {abs(d):.1f}%, 要求回报率{r.get('required_return', REQUIRED_RETURN):.1%})")

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

        average_discount = r.get("average_discount")
        average_discount = float(average_discount) if average_discount is not None else 0.0
        print(f"     安全边际: {average_discount:.1f}% | 覆盖方法: {r.get('valuation_coverage', 0)} | 置信度: {r.get('valuation_confidence', 0):.0%} | 分歧: {r.get('method_dispersion', 'N/A')} | {('可进入候选池' if r.get('eligible') else '仅供观察')}")
        print(f"     数据状态: {r.get('data_freshness_status', 'unknown')} | 财务数据年龄: {r.get('financial_age_days', 'N/A')}天 | PIT: {r.get('point_in_time_status', 'unknown')}")

        if fundamentals:
            print(f"     基本面: {' | '.join(fundamentals)}")

        if r.get("target_mean"):
            diff = (r["target_mean"] - price) / price * 100
            print(f"     分析师目标价: ${r['target_mean']:.1f} (潜在{'上涨' if diff > 0 else '下跌'}{abs(diff):.1f}%)")


def save_results(records, market, errors=None):
    """保存估值结果到 JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    safe = [r for r in records if r.get("eligible")]
    with open(os.path.join(os.path.dirname(__file__), "v5_config.json"), "rb") as config_file:
        config_hash = hashlib.sha256(config_file.read()).hexdigest()
    path = os.path.join(CACHE_DIR, f"valuation_{market}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "market": market,
            "generated_at": datetime.now().isoformat(),
            "total": len(records),
            "valued": len(safe),
            "errors": errors or [],
            "config_sha256": config_hash,
            "assumptions": MARKET_ASSUMPTIONS.get(market, MARKET_ASSUMPTIONS["US"]),
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
    parser.add_argument("--force-refresh", action="store_true", help="强制刷新免费数据源")
    parser.add_argument("--save", action="store_true", help="保存结果到 JSON")
    args = parser.parse_args()

    tickers = US_WATCHLIST if args.market == "US" else HK_WATCHLIST
    label = "港股" if args.market == "HK" else "美股"

    print(f"📊 {label}估值分析 — {len(tickers)} 只标的")
    assumptions = _market_assumptions(args.market)
    required_return = assumptions["risk_free_rate"] + assumptions["equity_risk_premium"]
    print(f"  请求回报率: {required_return:.0%} = 无风险{assumptions['risk_free_rate']:.1%} + ERP{assumptions['equity_risk_premium']:.1%}")

    records = fetch_fundamentals(tickers, force_refresh=args.force_refresh, market=args.market)
    if not records:
        print("❌ 没有获取到基本面数据")
        sys.exit(1)

    records = compute_valuation_scores(records, market=args.market)
    print_valuation_report(records, args.market, top_n=args.top)

    errors = getattr(fetch_fundamentals, "last_errors", [])
    if errors:
        print(f"  数据获取失败: {len(errors)} 只，详情已写入结果 metadata")
    if args.save:
        save_results(records, args.market, errors=errors)
