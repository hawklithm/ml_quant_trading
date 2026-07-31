"""Free market-data providers used by the valuation workflow.

US fundamentals come from SEC EDGAR companyfacts. HK fundamentals use the
free AkShare public endpoint when installed; prices use the existing Tencent /
Sina adapter. Every returned record carries its source and reporting dates.
"""

import os
import time
from datetime import datetime

import numpy as np
import requests


SEC_HEADERS = {
    "User-Agent": os.getenv("SEC_USER_AGENT", "ml-quant-trading research contact@example.com"),
    "Accept-Encoding": "gzip, deflate",
}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
SEC_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _ratio(value):
    value = _number(value)
    if value is None:
        return None
    return value / 100 if abs(value) > 1 else value


def _request_json(url, headers=None, timeout=20):
    response = requests.get(url, headers=headers or {}, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _latest_fact(facts, concepts, unit=None):
    """Select the latest filed fact, preferring annual/quarterly facts."""
    candidates = []
    for concept in concepts:
        for namespace in ("us-gaap", "dei"):
            item = facts.get(namespace, {}).get(concept, {})
            units = item.get("units", {})
            for unit_name, rows in units.items():
                if unit and unit_name != unit:
                    continue
                for row in rows:
                    value = _number(row.get("val"))
                    if value is None or not row.get("end"):
                        continue
                    candidates.append((row.get("filed", ""), row.get("end", ""), value, unit_name, row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    filed, period_end, value, unit_name, row = candidates[-1]
    return {
        "value": value,
        "filed_at": filed,
        "period_end": period_end,
        "unit": unit_name,
        "form": row.get("form"),
        "accession": row.get("accn"),
    }


def _sec_record(ticker, price, facts, company):
    company_name = company if isinstance(company, str) else (company or {}).get("name", ticker)
    eps = _latest_fact(facts, ["EarningsPerShareDiluted", "EarningsPerShareBasic"], "USD/shares")
    shares = _latest_fact(facts, ["EntityCommonStockSharesOutstanding"], "shares")
    equity = _latest_fact(facts, ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"])
    assets = _latest_fact(facts, ["Assets"])
    revenue = _latest_fact(facts, ["Revenues", "SalesRevenueNet", "RevenueFromContractWithCustomerExcludingAssessedTax"])
    net_income = _latest_fact(facts, ["NetIncomeLoss", "ProfitLoss"])
    cash = _latest_fact(facts, ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"])
    debt = _latest_fact(facts, ["LongTermDebtCurrent", "LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsCurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"])
    operating_cashflow = _latest_fact(facts, ["NetCashProvidedByUsedInOperatingActivities"])
    capex = _latest_fact(facts, ["PaymentsToAcquirePropertyPlantAndEquipment"])

    shares_value = shares["value"] if shares else None
    eps_value = eps["value"] if eps else None
    equity_value = equity["value"] if equity else None
    price = _number(price)
    fcf_value = None
    if operating_cashflow and capex:
        fcf_value = operating_cashflow["value"] + capex["value"]
    latest_dates = [item for item in (eps, shares, equity, revenue, net_income, operating_cashflow) if item]
    latest = max(latest_dates, key=lambda item: item.get("filed_at", "")) if latest_dates else {}
    record = {
        "ticker": ticker,
        "name": company_name,
        "sector": "Other",
        "industry": "",
        "valuation_group": "Other",
        "price": price,
        "trailing_eps": eps_value,
        "forward_eps": None,
        "eps": eps_value,
        "bvps": equity_value / shares_value if equity_value and shares_value else None,
        "fcf_ps": fcf_value / shares_value if fcf_value is not None and shares_value else None,
        "fcf_source": "sec_xbrl_annual",
        "cash_flow_type": "fcfe_proxy",
        "trailing_pe": price / eps_value if price and eps_value and eps_value > 0 else None,
        "forward_pe": None,
        "comparable_pe": price / eps_value if price and eps_value and eps_value > 0 else None,
        "comparable_eps": eps_value,
        "pb": price / (equity_value / shares_value) if price and equity_value and shares_value else None,
        "ps": price / (revenue["value"] / shares_value) if price and revenue and shares_value and revenue["value"] > 0 else None,
        "ev_ebitda": None,
        "dividend_yield": None,
        "earnings_growth": None,
        "revenue_growth": None,
        "roe": net_income["value"] / equity_value if net_income and equity_value else None,
        "profit_margin": net_income["value"] / revenue["value"] if net_income and revenue and revenue["value"] else None,
        "debt_equity": debt["value"] / equity_value * 100 if debt and equity_value else None,
        "market_cap": price * shares_value if price and shares_value else None,
        "total_debt": debt["value"] if debt else None,
        "cash_total": cash["value"] if cash else None,
        "target_mean": None,
        "fcf": fcf_value,
        "shares": shares_value,
        "data_asof": datetime.now().isoformat(),
        "price_asof": datetime.now().isoformat(),
        "financial_period_end": latest.get("period_end"),
        "filed_at": latest.get("filed_at"),
        "filing_accession": latest.get("accession"),
        "point_in_time_ready": bool(latest.get("filed_at")),
        "point_in_time_status": "sec_filed_snapshot" if latest.get("filed_at") else "missing_filing_date",
        "source": "sec_edgar",
        "currency": "USD",
    }
    return record


class FreeDataProvider:
    """Provider facade with no paid API dependency."""

    def __init__(self):
        self._sec_tickers = None

    def _load_sec_tickers(self):
        if self._sec_tickers is None:
            payload = _request_json(SEC_TICKERS_URL, headers=SEC_HEADERS)
            self._sec_tickers = {}
            for row in payload.get("data", []):
                if len(row) >= 3:
                    self._sec_tickers[str(row[2]).upper()] = str(row[0]).zfill(10)
        return self._sec_tickers

    def _fetch_us(self, tickers):
        from tencent_data import fetch_sina_realtime

        quotes = fetch_sina_realtime(tickers)
        records, errors = [], []
        try:
            mapping = self._load_sec_tickers()
        except Exception as exc:
            return records, [{"ticker": ticker, "status": "provider_unavailable", "reason": str(exc), "source": "sec_edgar"} for ticker in tickers]
        for ticker in tickers:
            try:
                cik = mapping.get(ticker.upper())
                if not cik:
                    raise ValueError("ticker not found in SEC company_tickers_exchange")
                facts_payload = _request_json(SEC_FACTS_URL.format(cik=cik), headers=SEC_HEADERS)
                quote = quotes.get(ticker, {})
                record = _sec_record(ticker, quote.get("price"), facts_payload.get("facts", {}), facts_payload.get("entityName", {}))
                if not record.get("price"):
                    raise ValueError("free quote provider returned no price")
                records.append(record)
                time.sleep(0.12)
            except Exception as exc:
                errors.append({"ticker": ticker, "status": "error", "reason": str(exc), "source": "sec_edgar"})
        return records, errors

    def _fetch_hk(self, tickers):
        from tencent_data import fetch_sina_realtime

        quotes = fetch_sina_realtime(tickers)
        records, errors = [], []
        try:
            import akshare as ak
        except ImportError:
            ak = None
        for ticker in tickers:
            code = ticker.replace(".HK", "").zfill(5)
            quote = quotes.get(ticker, {})
            try:
                if ak is None:
                    raise RuntimeError("akshare is required for free HK fundamentals; install requirements.txt")
                frame = ak.stock_hk_financial_indicator_em(symbol=code)
                if frame is None or frame.empty:
                    raise ValueError("HK free fundamentals returned no rows")
                row = frame.iloc[0].to_dict()
                get = lambda *names: next((_number(row.get(name)) for name in names if _number(row.get(name)) is not None), None)
                eps = get("基本每股收益(元)")
                bvps = get("每股净资产(元)")
                shares = get("已发行股本(股)")
                record = {
                    "ticker": ticker, "name": str(row.get("公司名称") or ticker), "sector": "Other", "industry": "", "valuation_group": "Other",
                    "price": quote.get("price"), "trailing_eps": eps, "forward_eps": None, "eps": eps, "bvps": bvps,
                    # Operating cash flow is not free cash flow; do not feed it into DCF.
                    "fcf_ps": None, "fcf_source": "operating_cashflow_only", "cash_flow_type": "unavailable",
                    "trailing_pe": get("市盈率"), "forward_pe": None, "comparable_pe": get("市盈率"), "comparable_eps": eps,
                    "pb": get("市净率"), "ps": None, "ev_ebitda": None, "dividend_yield": get("股息率TTM(%)"),
                    "earnings_growth": None, "revenue_growth": _ratio(get("营业总收入滚动环比增长(%)")), "roe": _ratio(get("股东权益回报率(%)")),
                    "profit_margin": _ratio(get("销售净利率(%)")), "debt_equity": None, "market_cap": get("总市值(港元)"),
                    "total_debt": None, "cash_total": None, "target_mean": None, "fcf": None, "shares": shares,
                    "data_asof": datetime.now().isoformat(), "price_asof": datetime.now().isoformat(), "financial_period_end": None,
                    "filed_at": None, "point_in_time_ready": False, "point_in_time_status": "hk_indicator_snapshot",
                    "source": "akshare_hk_public", "currency": "HKD",
                }
                records.append(record)
            except Exception as exc:
                errors.append({"ticker": ticker, "status": "error", "reason": str(exc), "source": "akshare_hk_public"})
        return records, errors

    def fetch(self, tickers, market):
        market = str(market).upper()
        return self._fetch_us(tickers) if market == "US" else self._fetch_hk(tickers)


def fetch_free_fundamentals(tickers, market):
    return FreeDataProvider().fetch(tickers, market)
