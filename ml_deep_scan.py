#!/usr/bin/env python3
"""
ML 量化选股深度扫描 — 美股+港股
集成 ml_optimized_picker v4 的全部功能：自适应窗口/横截面排名/多模型ensemble
"""
import numpy as np, pandas as pd, warnings, json, sys, os
warnings.filterwarnings('ignore')
from datetime import datetime

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

# 先用系统自带的跑美股
sys.path.insert(0, '/home/hawky/projects/quant-trading')
from ml_optimized_picker import score_stock_by_ml, get_ticker_sector, print_ml_report, run_ml_picking
from ml_optimized_picker import US_WATCHLIST, TOP_N, MODELS, ADAPTIVE_WINDOWS

print("="*80)
print("  ML 量化选股深度扫描 v4 — 全量运行")
print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("="*80)
print()

# ═══════════════════════════════════════
# Phase 1: 美股 Run ML (全量: RF+XGB+LGB, 自适应窗口, 横截面排名)
# ═══════════════════════════════════════
print("【Phase 1】美股全量 ML 选股 (RF+XGB+LGB+自适应窗口+横截面排名)")
print("="*60)

us_results, us_errors = run_ml_picking(
    tickers=US_WATCHLIST,
    models_to_use=["xgb", "lgb", "rf"],
    quick=False,
    use_adaptive=True,
    use_cross_section=True,
    top_n=TOP_N,
)

# 输出排名
if us_results:
    top = print_ml_report(us_results, top_n=min(30, len(us_results)))
else:
    print("❌ 美股全量运行失败")

# 保存
us_csv = None
if us_results:
    df = pd.DataFrame(us_results)
    us_csv = f"ml_us_picks_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(us_csv, index=False, encoding='utf-8-sig')
    print(f"  已保存: {us_csv}")

print()
print("="*80)

# ═══════════════════════════════════════
# Phase 2: 港股 ML 扫描 (需要逐一用 score_stock_by_ml)
# ═══════════════════════════════════════
print("【Phase 2】港股 ML 扫描")
print("="*60)

HK_TICKERS = [
    '0700.HK',  # 腾讯
    '9988.HK',  # 阿里巴巴
    '9999.HK',  # 网易
    '1810.HK',  # 小米
    '3690.HK',  # 美团
    '0941.HK',  # 中移动
    '0883.HK',  # 中海油
    '0388.HK',  # 港交所
    '0005.HK',  # 汇丰
    '1299.HK',  # 友邦
    '2269.HK',  # 药明生物
    '2382.HK',  # 舜宇光学
    '9618.HK',  # 京东
    '1024.HK',  # 快手
    '0123.HK',  # 越秀地产
    '0011.HK',  # 恒生银行
    '0939.HK',  # 建行
    '3988.HK',  # 中行
    '2628.HK',  # 国寿
    '0857.HK',  # 中石油
    '0027.HK',  # 银河娱乐
    '1928.HK',  # 金沙中国
    '1177.HK',  # 中国生物制药
    '0032.HK',  # 港通控股
]

hk_results = []
hk_errors = []
total = len(HK_TICKERS)

for i, t in enumerate(HK_TICKERS):
    try:
        print(f"  [{i+1}/{total}] {t} ... ", end="", flush=True)
        sr = score_stock_by_ml(t, period="2y", models_to_use=["xgb", "lgb", "rf"], use_adaptive=True)
        if sr is not None:
            hk_results.append(sr)
            print(f"评分={sr['score']:.3f} R²={sr['walk_forward_r2']:.3f} w={sr['adaptive_window']}")
        else:
            hk_errors.append(t)
            print(f"数据不足")
    except Exception as e:
        hk_errors.append(t)
        print(f"失败: {e}")

if hk_results:
    hk_results.sort(key=lambda x: x['score'], reverse=True)

    print()
    print(f"{'='*100}")
    print(f"  📊 港股 ML 选股结果 Top {len(hk_results)}")
    print(f"{'='*100}")
    hdr = (f" {'#':>3} {'代码':>10} {'ML评分':>7} {'可信度':>5} {'R²':>6} "
           f"{'预测5d':>8} {'价格':>8} {'1日':>7} {'1月':>8} {'3月':>8} "
           f"{'模型':>12} {'窗口':>5}")
    print(hdr)
    print(f" {'-'*95}")
    for i, s in enumerate(hk_results[:15]):
        models_str = "+".join(s["models_used"])
        print(f" {i+1:>3} {s['ticker']:>10} {s['score']:>7.3f} {s['confidence']:>5.2f} "
              f"{s['walk_forward_r2']:>6.3f} {s['pred_return']:>+6.2f}% "
              f"${s['price']:>6.2f} {s['chg_1d_pct']:>+5.1f}% {s['mom_1m']:>+5.1f}% {s['mom_3m']:>+5.1f}% "
              f"{models_str:>12} {s['adaptive_window']:>5}")

    # 保存港股结果
    hk_csv = f"ml_hk_picks_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    pd.DataFrame(hk_results).to_csv(hk_csv, index=False, encoding='utf-8-sig')
    print(f"  已保存: {hk_csv}")

print()
print(f"  ⚠️ 港股失败: {len(hk_errors)} 只: {', '.join(hk_errors[:8])}")

# ═══════════════════════════════════════
# Phase 3: 综合推荐
# ═══════════════════════════════════════
print()
print("="*80)
print(f"  🏆 综合推荐 (ML评分+技术面双验证)")
print("="*80)

# 定义评级函数
def get_rating(s, threshold_high=0.30, threshold_mid=0.15):
    if s['score'] > threshold_high and s['confidence'] > 0.3:
        return '★★★★★'
    elif s['score'] > threshold_high:
        return '★★★★'
    elif s['score'] > threshold_mid and s['confidence'] > 0.3:
        return '★★★'
    elif s['score'] > threshold_mid:
        return '★★'
    elif s['score'] > 0.08:
        return '★'
    else:
        return '☆'

print()
print("【美股 Top 10】")
print(f"{'评分':<4} {'代码':<8} {'ML分':<7} {'可信度':<6} {'R²':<7} {'预测5d':<9} {'现价':<10} {'1月':<8} {'3月':<8}")
print('-'*70)
for i, s in enumerate(us_results[:10]):
    stars = get_rating(s)
    print(f"{stars:<4} {s['ticker']:<8} {s['score']:<7.3f} {s['confidence']:<6.2f} {s['walk_forward_r2']:<7.3f} {s['pred_return']:<+6.2f}%  ${s['price']:<7.2f} {s['mom_1m']:<+5.1f}% {s['mom_3m']:<+5.1f}%")

print()
print("【港股 Top 10】")
print(f"{'评分':<4} {'代码':<10} {'ML分':<7} {'可信度':<6} {'R²':<7} {'预测5d':<9} {'现价':<10} {'1月':<8} {'3月':<8}")
print('-'*75)
for i, s in enumerate(hk_results[:10]):
    stars = get_rating(s)
    print(f"{stars:<4} {s['ticker']:<10} {s['score']:<7.3f} {s['confidence']:<6.2f} {s['walk_forward_r2']:<7.3f} {s['pred_return']:<+6.2f}%  ${s['price']:<7.2f} {s['mom_1m']:<+5.1f}% {s['mom_3m']:<+5.1f}%")

print()
print("✅ ML 量化选股深度扫描完成!")
