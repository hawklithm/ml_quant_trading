#!/usr/bin/env python3
"""
机器学习多因子选股演示 — 简化版

用 SPY 单标的做因子预测演示 (避免批量获取性能问题)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (14, 8)

print("=" * 60)
print("ML 因子预测模型 (单标的演示)")
print("=" * 60)

TICKER = "SPY"
print(f"\n获取 {TICKER} 5年数据...")
df = yf.download(TICKER, period="5y", auto_adjust=True, progress=False)
df.columns = [c[0] for c in df.columns]
close = df["Close"]
vol = df["Volume"]

print(f"数据行数: {len(close)}")

# ──────────── 构建时序特征 ────────────
print("构建特征...")

def make_features(close):
    f = pd.DataFrame(index=close.index)
    ret = close.pct_change()

    # 动量特征 (过去N日收益)
    for p in [1, 5, 10, 21, 42, 63]:
        f[f"ret_{p}d"] = close.pct_change(p)

    # 波动率特征
    for p in [5, 21, 63]:
        f[f"vol_{p}d"] = ret.rolling(p).std()

    # 价格偏离
    for p in [20, 50, 200]:
        f[f"sma{p}_dev"] = close / close.rolling(p).mean() - 1

    # 最高最低比
    for p in [10, 20]:
        f[f"high_low_ratio_{p}"] = close.rolling(p).max() / close.rolling(p).min() - 1

    # 日内波动
    f["daily_range"] = (df["High"] - df["Low"]) / close

    # RSI (简化版)
    delta = ret
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    f["rsi_14"] = 100 - (100 / (1 + rs))

    # 成交量变化
    if vol is not None:
        f["volume_change"] = vol.pct_change(5)
        f["volume_sma_ratio"] = vol / vol.rolling(20).mean()

    return f

features = make_features(close)

# 目标: 未来 5 日收益
target = close.pct_change(5).shift(-5)
TARGET_NAME = "未来5日收益"

# 对齐
data = features.join(target.rename("target")).dropna()

X = data.drop(columns=["target"]).values
y = data["target"].values
names = data.drop(columns=["target"]).columns.tolist()

print(f"特征数: {X.shape[1]}")
print(f"样本数: {X.shape[0]}")

# ──────────── 训练 ────────────
print("\n训练 RandomForest...")
scaler = StandardScaler()
X_s = scaler.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=10, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)

train_r2 = r2_score(y_tr, rf.predict(X_tr))
test_r2 = r2_score(y_te, rf.predict(X_te))
print(f"  训练 R²: {train_r2:.3f}")
print(f"  测试 R²: {test_r2:.3f}")

# ──────────── 特征重要性 ────────────
print("\nTop 10 重要特征:")
imp = rf.feature_importances_
order = np.argsort(imp)[::-1][:10]
for i, idx in enumerate(order):
    print(f"  {i+1}. {names[idx]:<20} {imp[idx]:.4f}")

# ──────────── 信号回测 ────────────
print("\n信号回测 (阈值开仓)...")
pred = rf.predict(X_s)
signal = pd.Series(0, index=data.index)
# 预测收益 > +1.5σ 做多, < -1.5σ 做空
threshold = pred.std() * 1.5
signal[pd.Series(pred, index=data.index) > threshold] = 1
signal[pd.Series(pred, index=data.index) < -threshold] = -1

daily_ret = close.pct_change()
strat_ret = signal.shift(1) * daily_ret.reindex(signal.index)
strat_cum = (1 + strat_ret.dropna()).cumprod()
buy_hold = (1 + daily_ret.reindex(signal.index).dropna()).cumprod()

# 指标
def calc_metrics(ret_series):
    sr = (ret_series.mean() / ret_series.std()) * np.sqrt(252) if ret_series.std() > 0 else 0
    cum = (1 + ret_series).cumprod()
    dd = (cum / cum.cummax() - 1).min()
    return sr, dd

strat_sr, strat_dd = calc_metrics(strat_ret.dropna())
bh_sr, bh_dd = calc_metrics(daily_ret.reindex(signal.index).dropna())

print(f"  做多阈值: > +{threshold:.4f}")
print(f"  做空阈值: < {threshold:.4f}")
print(f"  策略总收益: {strat_cum.iloc[-1]-1:.2%}")
print(f"  买入持有:    {buy_hold.iloc[-1]-1:.2%}")
print(f"  策略夏普:    {strat_sr:.2f}")
print(f"  BH 夏普:     {bh_sr:.2f}")
print(f"  策略最大回撤:{strat_dd:.2%}")
print(f"  BH 最大回撤: {bh_dd:.2%}")

# ──────────── 可视化 ────────────
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# 1. 特征重要性
ax = axes[0, 0]
top_idx = order[:10]
ax.barh([names[i] for i in top_idx][::-1], imp[top_idx][::-1], color="#2196F3")
ax.set_title("Top 10 特征重要性", fontweight="bold")
ax.set_xlabel("重要性")

# 2. 预测 vs 实际
ax = axes[0, 1]
ax.scatter(y_te, rf.predict(X_te), alpha=0.4, s=10, c="#2196F3")
lims = [min(y_te.min(), rf.predict(X_te).min()), max(y_te.max(), rf.predict(X_te).max())]
ax.plot(lims, lims, "r--", lw=1, alpha=0.6)
ax.set_xlabel("实际收益")
ax.set_ylabel("预测收益")
ax.set_title(f"预测 vs 实际 (测试集 R²={test_r2:.3f})", fontweight="bold")
ax.grid(True, alpha=0.25)

# 3. 净值曲线
ax = axes[1, 0]
ax.plot(strat_cum.index, strat_cum, label="ML策略", color="#4CAF50", lw=1.5)
ax.plot(buy_hold.index, buy_hold, label="买入持有", color="#999", lw=1, alpha=0.6)
ax.legend(loc="best")
ax.set_title("净值曲线", fontweight="bold")
ax.set_ylabel("净值")
ax.grid(True, alpha=0.25)

# 4. 信号分布
ax = axes[1, 1]
ax.hist(pred, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
ax.axvline(threshold, color="green", ls="--", lw=1, label=f"做多阈值 (+{threshold:.3f})")
ax.axvline(-threshold, color="red", ls="--", lw=1, label=f"做空阈值 ({-threshold:.3f})")
ax.legend(fontsize=9)
ax.set_title("模型预测值分布", fontweight="bold")
ax.set_xlabel("预测收益")

plt.tight_layout()
plt.savefig("ml_factor_model.png", dpi=150)
print(f"\n图表: ml_factor_model.png")

print("\n" + "=" * 60)
print("说明")
print("=" * 60)
print(f"""
  标的: {TICKER}
  特征: 动量(6种) + 波动率(3种) + 均价偏离(3种)
        + 最高最低比(2种) + 日内波动 + RSI + 成交量(2种) = 18个特征
  模型: RandomForest (200棵, max_depth=6)
  目标: {TARGET_NAME}

  特征工程 > 模型选择:
    - 多因子选股的核心是找到有效的因子
    - 不同市场环境适合不同因子
    - 需要持续监控因子衰减

  下一步步:
    - 用 AkShare 换 A股数据做多因子选股
    - 加入基本面因子 (PE/PB/ROE)
    - 做 Walk-Forward 滚动验证
""")
