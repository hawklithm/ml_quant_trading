import math

import valuation_screener as valuation
from valuation_screener import _dcf_per_share, compute_valuation_scores


def _record(ticker, price, pe, eps, growth=0.10):
    return {
        "ticker": ticker,
        "sector": "Technology",
        "industry": "Software",
        "valuation_group": "Software",
        "price": price,
        "eps": eps,
        "bvps": 10.0,
        "fcf_ps": 2.0,
        "trailing_pe": pe,
        "forward_pe": pe,
        "earnings_growth": growth,
        "revenue_growth": growth,
        "roe": 0.15,
        "profit_margin": 0.10,
        "debt_equity": 80.0,
        "graham_fv": math.sqrt(22.5 * eps * 10.0),
        "graham_discount": (math.sqrt(22.5 * eps * 10.0) - price) / math.sqrt(22.5 * eps * 10.0) * 100,
        "fcf_fv": _dcf_per_share(2.0, growth, 0.10, 0.02),
        "fcf_discount": 10.0,
        "peg": pe / (growth * 100),
    }


def test_dcf_is_finite_and_growth_bounded():
    value = _dcf_per_share(2.0, 0.10, 0.10, 0.02)
    assert value is not None and math.isfinite(value) and value > 0
    assert _dcf_per_share(2.0, 0.10, 0.02, 0.02) is None
    assert _dcf_per_share(2.0, 0.10, 0.10, 0.02, net_debt_ps=5.0) < value
    assert _dcf_per_share(2.0, 0.10, 0.10, 0.02, dilution_rate=0.10) < value


def test_valuation_requires_absolute_margin_and_quality():
    records = [_record("AAA", 15, 20, 2.0), _record("BBB", 30, 25, 2.0), _record("CCC", 45, 30, 2.0)]
    result = compute_valuation_scores(records, market="US")
    assert all("eligible" in record and "valuation_coverage" in record for record in result)
    assert all(record["valuation_coverage"] >= 2 for record in result)
    assert any(record["eligible"] for record in result)


def test_negative_quality_is_not_eligible():
    records = [_record("AAA", 40, 20, 2.0), _record("BBB", 50, 25, 2.0), _record("CCC", 60, 30, 2.0)]
    records[0]["debt_equity"] = 800.0
    result = compute_valuation_scores(records, market="US")
    assert result[0]["quality_pass"] is False
    assert result[0]["eligible"] is False


def test_incomplete_quality_and_expensive_valuation_are_not_eligible():
    records = [_record("AAA", 15, 100, 2.0), _record("BBB", 30, 25, 2.0), _record("CCC", 45, 30, 2.0)]
    records[0]["roe"] = None
    records[0]["profit_margin"] = None
    records[0]["debt_equity"] = None
    result = compute_valuation_scores(records, market="US")
    assert result[0]["quality_status"] == "incomplete"
    assert result[0]["hard_valuation_pass"] is False
    assert result[0]["eligible"] is False


def test_cache_schema_invalidates_old_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(valuation, "CACHE_DIR", str(tmp_path))
    valuation._write_fundamentals_cache("US", [{"ticker": "AAA"}])
    assert valuation._read_fundamentals_cache("US", ["AAA"])[0]["ticker"] == "AAA"
    original_version = valuation.CACHE_SCHEMA_VERSION
    try:
        valuation.CACHE_SCHEMA_VERSION += 1
        assert valuation._read_fundamentals_cache("US", ["AAA"]) is None
    finally:
        valuation.CACHE_SCHEMA_VERSION = original_version
