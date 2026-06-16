#!/usr/bin/env python3
"""ML 深度扫描 — 分批下载防限速, 聚焦推荐池"""
import yfinance as yf, pandas as pd, numpy as np, warnings, sys, os, time
warnings.filterwarnings('ignore')
from datetime import datetime

sys.path.insert(0, '/home/hawky/projects/quant-trading')
from ml_optimized_picker import score_stock_by_ml, build_features_v3, get_ticker_sector
from ml_optimized_picker import MODELS, ADAPTIVE_WINDOWS, MIN_TRADING_DAYS, FORECAST_HORIZON
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

print('=' * 80)
print('  ML 量化选股深度扫描 v4 — 分批模式')
print(f'  {datetime.now().strftime("%Y-%m-%d %H:%M")}')
print('=' * 80)

# ─── 推荐股票池（基于前一轮技术面筛选的优质标的）───
TICKERS = {
    'US': ['JPM','LLY','CAT','GOOGL','COST','V','NVDA','META','MSFT','AMZN',
           'WMT','PG','XOM','CVX','UNH','JNJ','BAC','GS','MA','BLK',
           'KO','PEP','MCD','HD','LOW','TMO','ABBV','MRK'],
    'HK': ['0700.HK','9988.HK','9999.HK','1810.HK','3690.HK',
           '0941.HK','0883.HK','0388.HK','0005.HK','1299.HK',
           '2269.HK','2382.HK','9618.HK','0857.HK','0027.HK']
}
NAMES_HK = {'0700.HK':'腾讯','9988.HK':'阿里','9999.HK':'网易','1810.HK':'小米',
            '3690.HK':'美团','0941.HK':'中移动','0883.HK':'中海油',
            '0388.HK':'港交所','0005.HK':'汇丰','1299.HK':'友邦',
            '2269.HK':'药明','2382.HK':'舜宇','9618.HK':'京东'}

def score_single_stock(ticker, period='2y'):
    """简化的单股票评分 — 更健壮"""
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty or len(df) < MIN_TRADING_DAYS:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        
        features, target = build_features_v3(df)
        if len(features) < 100:
            return None
        
        X = features.dropna()
        y = target.loc[X.index].dropna()
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common]
        
        if len(X) < 100:
            return None
        
        # 训练 RF (最快)
        tscv = TimeSeriesSplit(n_splits=5)
        scaler = StandardScaler()
        preds = []
        test_ys = []
        
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            model = RandomForestRegressor(n_estimators=200, max_depth=6,
                                          min_samples_leaf=10, n_jobs=-1, random_state=42)
            model.fit(X_tr_s, y_tr)
            preds.extend(model.predict(X_te_s))
            test_ys.extend(y_te)
        
        wf_r2 = r2_score(test_ys, preds) if len(preds) > 5 else 0
        
        # 最新预测
        X_all_s = scaler.fit_transform(X)
        model_all = RandomForestRegressor(n_estimators=200, max_depth=6,
                                          min_samples_leaf=10, n_jobs=-1, random_state=42)
        model_all.fit(X_all_s, y)
        latest_pred = float(model_all.predict(X_all_s[-1:].reshape(1, -1))[0])
        
        ref_returns = y.values
        pctl = np.searchsorted(np.sort(ref_returns), latest_pred) / len(ref_returns) if len(ref_returns) > 50 else 0.5
        ml_score = max(0, min(1, (pctl - 0.4) / 0.5))
        confidence = min(max(wf_r2 * 3, 0.1), 1.0)
        final_score = ml_score * confidence
        
        closes = df['Close'].values
        mom_1m = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
        mom_3m = (closes[-1] / closes[-63] - 1) * 100 if len(closes) >= 63 else 0
        
        # Feature importance
        imp = model_all.feature_importances_
        top_feats = [X.columns[i] for i in np.argsort(imp)[::-1][:5]]
        
        return {
            'ticker': ticker,
            'price': round(float(closes[-1]), 2),
            'ml_score': round(ml_score, 4),
            'confidence': round(confidence, 4),
            'pred_return': round(latest_pred * 100, 2),
            'walk_forward_r2': round(wf_r2, 4),
            'mom_1m': round(mom_1m, 2),
            'mom_3m': round(mom_3m, 2),
            'final_score': round(final_score, 4),
            'top_features': top_feats,
        }
    except Exception as e:
        return None

# ─── 分批运行 ───
all_results = []

for market, tickers in TICKERS.items():
    print(f"\n【{market}】扫描 {len(tickers)} 只...")
    print('-' * 60)
    
    for i, t in enumerate(tickers):
        sys.stdout.write(f'  [{i+1}/{len(tickers)}] {t:>6} ... ')
        sys.stdout.flush()
        sr = score_single_stock(t)
        if sr:
            all_results.append(sr)
            tag = '★' if sr['final_score'] > 0.15 else '·'
            print(f"{tag} 评分={sr['final_score']:.4f} R²={sr['walk_forward_r2']:.3f} 预测{sr['pred_return']:+.2f}%")
        else:
            print('  数据不足')
        time.sleep(0.5)  # 防限速

# ─── 排序输出 ───
if all_results:
    all_results.sort(key=lambda x: x['final_score'], reverse=True)
    
    print()
    print('=' * 90)
    print('  🏆 ML 量化选股完整排名')
    print('=' * 90)
    print(f"{'#':>3} {'市场':>4} {'代码':>10} {'名称':>6} {'ML分':>7} {'可信':>6} {'R²':>6} {'预测5d':>8} {'现价':>8} {'1月':>7} {'3月':>7}")
    print('-' * 72)
    
    for i, s in enumerate(all_results):
        market = '美股' if s['ticker'] not in [t for sub in TICKERS.values() for t in sub[:0]] else \
                 '港股' if s['ticker'].endswith('.HK') else '美股'
        name = NAMES_HK.get(s['ticker'], '')
        print(f"{i+1:>3} {market:>4} {s['ticker']:>10} {name:>6} {s['final_score']:>7.4f} {s['confidence']:>6.2f} {s['walk_forward_r2']:>6.3f} {s['pred_return']:>+6.2f}% ${s['price']:>7.2f} {s['mom_1m']:>+5.1f}% {s['mom_3m']:>+5.1f}%")
    
    # 保存
    csv_path = f"/home/hawky/projects/quant-trading/ml_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    pd.DataFrame(all_results).to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  已保存: {csv_path}")
    
    print()
    print('=' * 80)
    print('  🏆 综合推荐 (ML评分降序)')
    print('=' * 80)
    
    for i, s in enumerate(all_results[:8]):
        name = NAMES_HK.get(s['ticker'], '')
        ticker_label = f"{name}({s['ticker']})" if name else s['ticker']
        stars = '★★★★★' if s['final_score'] > 0.35 else \
                '★★★★' if s['final_score'] > 0.25 else \
                '★★★' if s['final_score'] > 0.15 else \
                '★★' if s['final_score'] > 0.08 else '★'
        
        verdict = '🟢 强烈推荐' if s['final_score'] > 0.30 else \
                  '🟢 推荐' if s['final_score'] > 0.20 else \
                  '🟡 关注' if s['final_score'] > 0.10 else \
                  '⚪ 观察' if s['final_score'] > 0.05 else '🔴 回避'
        
        print(f"\n{stars} {ticker_label}")
        print(f"    {verdict}")
        print(f"    ML评分: {s['final_score']:.4f} | 可信度: {s['confidence']:.2f} | R²: {s['walk_forward_r2']:.3f}")
        print(f"    预测5d收益: {s['pred_return']:+.2f}% | 现价: ${s['price']:.2f}")
        print(f"    1月动量: {s['mom_1m']:+.1f}% | 3月动量: {s['mom_3m']:+.1f}%")
        print(f"    关键特征: {', '.join(str(f) for f in s['top_features'][:5])}")

print(f"\n{'='*80}")
print(f"  ✅ ML 深度扫描完成! 共评分 {len(all_results)} 只股票")
print(f"{'='*80}")
