#!/usr/bin/env python3
"""Debug: test ML pipeline on one stock"""
import yfinance as yf, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

# 1. 下载
d = yf.download("AAPL", period="2y", auto_adjust=True, progress=False)
print(f"1. 下载: {len(d)} 行, MultiIndex={isinstance(d.columns, pd.MultiIndex)}")

# 2. 展平
if isinstance(d.columns, pd.MultiIndex):
    d.columns = [c[0] for c in d.columns]
print(f"2. 展平后 len={len(d)}, cols={d.columns[:3].tolist()}")

# 3. build_features
from ml_optimized_picker import build_features_v3
feat, target = build_features_v3(d)
print(f"3. 特征: {feat.shape}, 目标非空: {target.notna().sum()}")

# 4. 去NA
X = feat.dropna()
y = target.loc[X.index].dropna()
common = X.index.intersection(y.index)
print(f"4. X={len(X)}, y={len(y)}, common={len(common)}")
print(f"5. 满足>=100: {len(X) >= 100}")

# 5. 打印X的列数
print(f"6. 特征列数: {X.shape[1]}")
print(f"7. X前几行: {X.head(2)}")
