#!/usr/bin/env python3
"""Re-process old saved valuation data through enhanced scoring logic."""
import sys, json
sys.path.insert(0, '.')
from valuation_screener import _valuation_policy, compute_valuation_scores, print_valuation_report, save_results

FILES = {
    "US": "/home/hawky/.cache/hermes-quant/valuation_US_20260731_1053.json",
    "HK": "/home/hawky/.cache/hermes-quant/valuation_HK_20260731_1056.json",
}

for market, fpath in FILES.items():
    data = json.load(open(fpath))
    records = data["results"]
    
    for r in records:
        r.setdefault("industry", "")
        r.setdefault("valuation_group", r.get("sector", "Other"))
        r.setdefault("total_debt", None)
        r.setdefault("cash_total", None)
        r["dcf_fv"] = r.get("fcf_fv")
        r.setdefault("required_return", 0.10)
        r.setdefault("terminal_growth", 0.02)
        r.setdefault("data_asof", "2026-07-31")
        r.setdefault("market", market)
        pol, methods = _valuation_policy(r.get("industry", ""), r.get("sector", "Other"))
        r["valuation_policy"] = pol
        r["allowed_valuation_methods"] = methods

    records = compute_valuation_scores(records, market=market)
    print_valuation_report(records, market, top_n=15)
    save_results(records, market)

    eligible = [r for r in records if r.get("eligible")]
    print(f"\n  ✅ 安全边际合格标的: {len(eligible)} 只")
    for r in eligible:
        print(f"    {r['ticker']:>10} | 估值分 {r['value_score']:.0f} | 折价 {r['average_discount']:.1f}%")

    if not eligible:
        by_discount = sorted(records, key=lambda r: -(r.get('average_discount') or -999))
        print(f"\n  接近门槛标的:")
        for r in by_discount[:5]:
            ad = r.get('average_discount')
            if ad is not None:
                print(f"    {r['ticker']:>10} | 估值分 {r['value_score']:.0f} | 折价 {ad:+.1f}% | 覆盖 {r.get('valuation_coverage',0)} 方法")
