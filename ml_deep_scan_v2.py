#!/usr/bin/env python3
"""ML 量化选股深度扫描 — 美股+港股"""
import sys, os, warnings
warnings.filterwarnings('ignore')
from datetime import datetime

sys.path.insert(0, '/home/hawky/projects/quant-trading')
from ml_optimized_picker import score_stock_by_ml, run_ml_picking, print_ml_report
from ml_optimized_picker import US_WATCHLIST
import pandas as pd, numpy as np

print('=' * 80)
print('  ML 量化选股深度扫描 v4')
print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 80)

# ─── Phase 1: 美股 ───
print()
print('【Phase 1】美股全量 ML 选股 (RF+XGB+LGB+自适应窗口+横截面)')
print('=' * 60)

us_r, us_e = run_ml_picking(
    tickers=US_WATCHLIST,
    models_to_use=['xgb', 'lgb', 'rf'],
    use_adaptive=True,
    use_cross_section=True,
)

if us_r:
    print()
    print('【美股 ML Top 15】')
    print(f"{'#':>3} {'代码':>6} {'ML分':>8} {'可信':>5} {'R²':>6} {'预测5d':>8} {'价格':>8} {'1月':>7} {'3月':>7}")
    print('-' * 65)
    for i, s in enumerate(us_r[:15]):
        print(f"{i+1:>3} {s['ticker']:>6} {s['score']:>8.4f} {s['confidence']:>5.2f} {s['walk_forward_r2']:>6.3f} {s['pred_return']:>+6.2f}% ${s['price']:>7.2f} {s['mom_1m']:>+5.1f}% {s['mom_3m']:>+5.1f}%")

    pd.DataFrame(us_r).to_csv(f'/home/hawky/projects/quant-trading/ml_us_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False, encoding='utf-8-sig')
print(f'美股失败: {len(us_e)} 只')

# ─── Phase 2: 港股 ───
print()
print('【Phase 2】港股 ML 扫描')
print('=' * 60)

HK_LIST = ['0700.HK','9988.HK','9999.HK','1810.HK','3690.HK',
           '0941.HK','0883.HK','0388.HK','0005.HK','1299.HK',
           '2269.HK','2382.HK','9618.HK','1024.HK',
           '0939.HK','3988.HK','0857.HK','0027.HK','1928.HK','1177.HK']

NAMES = {'0700.HK':'腾讯','9988.HK':'阿里','9999.HK':'网易','1810.HK':'小米',
         '3690.HK':'美团','0941.HK':'中移动','0883.HK':'中海油',
         '0388.HK':'港交所','0005.HK':'汇丰','1299.HK':'友邦',
         '2269.HK':'药明','2382.HK':'舜宇','9618.HK':'京东'}

hk_r = []
hk_e = []
for i, t in enumerate(HK_LIST):
    sys.stdout.write(f'  [{i+1}/{len(HK_LIST)}] {t} ... ')
    sys.stdout.flush()
    try:
        sr = score_stock_by_ml(t, period='2y', models_to_use=['xgb','lgb','rf'], use_adaptive=True)
        if sr is not None:
            hk_r.append(sr)
            print(f"评分={sr['score']:.4f} R²={sr['walk_forward_r2']:.3f}")
        else:
            hk_e.append(t)
            print('数据不足')
    except Exception as e:
        hk_e.append(t)
        print(f'失败')

if hk_r:
    hk_r.sort(key=lambda x: x['score'], reverse=True)
    print()
    print('【港股 ML Top 15】')
    print(f"{'#':>3} {'代码':>10} {'名称':>6} {'ML分':>8} {'可信':>5} {'R²':>6} {'预测5d':>8} {'价格':>8} {'1月':>7} {'3月':>7}")
    print('-' * 75)
    for i, s in enumerate(hk_r[:15]):
        n = NAMES.get(s['ticker'], '')
        print(f"{i+1:>3} {s['ticker']:>10} {n:>6} {s['score']:>8.4f} {s['confidence']:>5.2f} {s['walk_forward_r2']:>6.3f} {s['pred_return']:>+6.2f}% ${s['price']:>7.2f} {s['mom_1m']:>+5.1f}% {s['mom_3m']:>+5.1f}%")

    pd.DataFrame(hk_r).to_csv(f'/home/hawky/projects/quant-trading/ml_hk_{datetime.now().strftime("%Y%m%d_%H%M")}.csv', index=False, encoding='utf-8-sig')

print(f'港股失败: {len(hk_e)} 只')

# ─── Phase 3: 综合报告 ───
print()
print('=' * 80)
print('  🏆 综合推荐报告')
print('=' * 80)

if us_r:
    print()
    print('【🟢 美股 Top 5 推荐】')
    for i, s in enumerate(us_r[:5]):
        stars = '★★★★★' if s['score'] > 0.35 else '★★★★' if s['score'] > 0.25 else '★★★' if s['score'] > 0.15 else '★★'
        print(f'  {stars} {s["ticker"]}')
        print(f'     ML评分: {s["score"]:.4f} | 可信度: {s["confidence"]:.2f} | R²: {s["walk_forward_r2"]:.3f}')
        print(f'     预测5d收益: {s["pred_return"]:+.2f}% | 现价: ${s["price"]:.2f}')
        print(f'     1月动量: {s["mom_1m"]:+.1f}% | 3月动量: {s["mom_3m"]:+.1f}%')
        if 'top_features' in s:
            print(f'     关键特征: {", ".join(s["top_features"][:5])}')
        print()

if hk_r:
    print('【🟢 港股 Top 5 推荐】')
    for i, s in enumerate(hk_r[:5]):
        n = NAMES.get(s['ticker'], s['ticker'])
        stars = '★★★★★' if s['score'] > 0.35 else '★★★★' if s['score'] > 0.25 else '★★★' if s['score'] > 0.15 else '★★'
        print(f'  {stars} {n}({s["ticker"]})')
        print(f'     ML评分: {s["score"]:.4f} | 可信度: {s["confidence"]:.2f} | R²: {s["walk_forward_r2"]:.3f}')
        print(f'     预测5d收益: {s["pred_return"]:+.2f}% | 现价: ${s["price"]:.2f}')
        print(f'     1月动量: {s["mom_1m"]:+.1f}% | 3月动量: {s["mom_3m"]:+.1f}%')
        print()

# 关联分析: ML分数 vs 当前价格位置
if us_r:
    print('【📊 ML评分 vs 动量散点】')
    print(f"{'代码':>6} {'ML分':>7} {'R²':>6} {'预测5d':>9} {'价格位置':>9}")
    print('-' * 40)
    for s in us_r[:10]:
        # 价格位置: 根据1月动量判断
        pos = '高位' if s['mom_1m'] > 3 else '中性' if s['mom_1m'] > -3 else '低位'
        print(f"{s['ticker']:>6} {s['score']:>7.4f} {s['walk_forward_r2']:>6.3f} {s['pred_return']:>+6.2f}% {pos:>9}")

print()
print('=' * 80)
print('  ✅ ML 量化选股扫描完成!')
print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 80)
