import valuation_screener as valuation
from quant.data.free_provider import _ratio, _sec_record


def test_sec_record_normalizes_core_fields_and_dates():
    facts = {
        "us-gaap": {
            "EarningsPerShareDiluted": {"units": {"USD/shares": [{"val": 4.0, "end": "2025-12-31", "filed": "2026-02-01", "form": "10-K", "accn": "x"}]}},
            "EntityCommonStockSharesOutstanding": {"units": {"shares": [{"val": 100.0, "end": "2025-12-31", "filed": "2026-02-01"}]}},
            "StockholdersEquity": {"units": {"USD": [{"val": 500.0, "end": "2025-12-31", "filed": "2026-02-01"}]}},
            "NetIncomeLoss": {"units": {"USD": [{"val": 50.0, "end": "2025-12-31", "filed": "2026-02-01"}]}},
            "Revenues": {"units": {"USD": [{"val": 250.0, "end": "2025-12-31", "filed": "2026-02-01"}]}},
            "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [{"val": 80.0, "end": "2025-12-31", "filed": "2026-02-01"}]}},
            "PaymentsToAcquirePropertyPlantAndEquipment": {"units": {"USD": [{"val": -20.0, "end": "2025-12-31", "filed": "2026-02-01"}]}},
        }
    }
    record = _sec_record("AAA", 10.0, facts, "AAA Inc")
    assert record["source"] == "sec_edgar"
    assert record["eps"] == 4.0
    assert record["bvps"] == 5.0
    assert record["fcf_ps"] == 0.6
    assert record["filed_at"] == "2026-02-01"
    assert record["point_in_time_ready"] is True


def test_hk_percentages_are_normalized():
    assert _ratio(15.0) == 0.15
    assert _ratio(0.15) == 0.15


def test_free_records_can_enter_the_existing_valuation_pipeline():
    record = _sec_record("AAA", 10.0, {"us-gaap": {}}, "AAA Inc")
    valuation.compute_valuation_scores([record], market="US")
    assert "valuation_confidence" in record
    assert record["source"] == "sec_edgar"
