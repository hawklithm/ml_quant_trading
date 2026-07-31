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
    assert _dcf_per_share(2.0, 0.10, 0.10, 0.02, net_debt_ps=5.0, cash_flow_type="fcfe") > _dcf_per_share(2.0, 0.10, 0.10, 0.02, net_debt_ps=5.0, cash_flow_type="fcff")


def test_normalized_fcf_uses_median_and_fallback():
    import pandas as pd

    cashflow = pd.DataFrame([[100.0, 120.0, 80.0]], index=["Free Cash Flow"], columns=["a", "b", "c"])
    value, source = valuation._normalized_fcf_per_share(cashflow, 999.0, 10.0)
    assert value == 10.0
    assert source == "annual_median"
    value, source = valuation._normalized_fcf_per_share(None, 50.0, 10.0)
    assert value == 5.0
    assert source == "latest"
    negative = pd.DataFrame([[-100.0, -120.0]], index=["Free Cash Flow"])
    value, source = valuation._normalized_fcf_per_share(50.0, 50.0, 10.0)
    assert value == 5.0 and source == "latest"
    value, source = valuation._normalized_fcf_per_share(negative, 50.0, 10.0)
    assert value is None and source == "annual_median_nonpositive"
    assert valuation._growth_percent(0.055) == 5.5
    assert valuation._growth_percent(5.5) == 5.5
    assert valuation._growth_percent(150) is None


def test_missing_methods_do_not_reduce_to_neutral_weight():
    records = [_record("AAA", 15, 20, 2.0), _record("BBB", 30, 25, 2.0), _record("CCC", 45, 30, 2.0)]
    for record in records:
        record["graham_discount"] = None
        record["peg"] = None
    result = compute_valuation_scores(records)
    assert all(record["valuation_confidence"] < 1 for record in result)
    assert all("method_agreement" in record and "data_freshness_status" in record for record in result)


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
