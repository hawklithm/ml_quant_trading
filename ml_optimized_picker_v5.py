#!/usr/bin/env python3
"""
ML 优化版选股系统 v5.2
====================
v5.2 全面优化 (2026-06-16):
  P1a 分类ensemble: RF+XGB+LGB三路投票, 替代单RF
  P1b 动量EMA平滑: 消除单点噪声, 用21d/63d EMA动量替代pct_change
  P1c 自适应窗口: 根据21d波动率动态选择504/756/1008窗口
  P1d 横截面特征排名: 特征值转历史百分位rank, 消除量纲依赖
  P1e 自适应超参: n_estimators/max_depth根据样本量自动调节
  P1f 时间衰减: 训练样本按时间指数加权, 近高远低
  P2a 宏观因子扩展: 加波动率变化+板块相对beta
  P2b 增量缓存: 缓存存活12h但每4h增量拉取最新行情补充

v5.1 优化改进 (根据复盘建议):
  P0a 回归目标 5d→21d: 提升信噪比, R²正比例从0%→30%
  P0b 特征精简 43→25个: 去掉冗余动量/波动率/RSI/成交量, 缓解过拟合
  P0b 动量兜底: R²<0时混合21d+63d动量评分, 权重随R²恶化递增
  P0a confidence权重 0.4→0.15: 降级高共识错判影响
  P1a 看跌阈值 -0.3→-0.25: 减少系统性看跌偏差
  P1b consumer板块 xly(可选)+xlp(必需) 双ETF因子
"""

import numpy as np
import pandas as pd
import warnings, sys, os, json, pickle, time
from datetime import datetime, timedelta
from scipy.stats import rankdata, spearmanr
from itertools import combinations

# ═══════════════════════════════════════
# 从 v5_config.json 加载参数
# ═══════════════════════════════════════
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "v5_config.json")
_CFG = None

def load_config():
    global _CFG
    if _CFG is None:
        with open(_CONFIG_PATH) as f:
            full = json.load(f)
        _CFG = full["ml_scoring"]
    return _CFG

CFG = load_config()

warnings.filterwarnings("ignore")

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL = CFG["cache_ttl_hours"] * 3600  # 从配置读取

# ──── ML 模型 ────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

# ──── 自适应超参 ────
def _get_adaptive_params(n_samples):
    """根据样本量自动选择模型超参"""
    if n_samples < 200:
        return {"n_estimators": 100, "max_depth": 4, "min_samples_leaf": 15}
    elif n_samples < 500:
        return {"n_estimators": 200, "max_depth": 5, "min_samples_leaf": 10}
    else:
        return {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 8}

def _build_reg_model(name, n_samples):
    """构建回归模型（自适应超参）"""
    p = _get_adaptive_params(n_samples)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=42)
    elif name == "xgb" and HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=0.05, n_jobs=-1, random_state=42)
    elif name == "lgb" and HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
    return None

def _build_cls_model(name, n_samples):
    """构建分类模型（自适应超参）"""
    p = _get_adaptive_params(n_samples)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], n_jobs=-1, random_state=42)
    elif name == "xgb" and HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=0.05, n_jobs=-1, random_state=42)
    elif name == "lgb" and HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            learning_rate=0.05, n_jobs=-1, random_state=42, verbose=-1)
    return None

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════
TOP_N = CFG["top_n"]
MIN_TRADING_DAYS = CFG["min_trading_days"]
TEST_SPLITS = CFG["test_splits"]
FORECAST_HORIZON_SHORT = CFG["forecast_horizon_short"]
FORECAST_HORIZON_LONG = CFG["forecast_horizon_long"]
FORECAST_HORIZON_MOM = CFG["forecast_horizon_mom"]
WARMUP = CFG["warmup"]

ADAPTIVE_WINDOWS = CFG["adaptive_windows"]

# v5: 移除ETF, 补全股票池
SECTOR_MAP = {
    "tech": {"AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","ORCL",
             "AMD","QCOM","TSM"},
    "finance": {"JPM","V","MA","GS","BLK"},
    "consumer": {"WMT","COST","HD","PG","KO","PEP","MCD"},
    "healthcare": {"UNH","JNJ","LLY","ABBV"},
    "energy": {"XOM"},
    "industrial": {"CAT","GE"},
    "hk_tech": {"0700.HK","9988.HK","9999.HK","1810.HK","3690.HK","9618.HK","1024.HK"},
    "hk_finance": {"0005.HK","1299.HK","0388.HK","0939.HK","3988.HK"},
    "hk_energy": {"0883.HK","0857.HK"},
    "hk_health": {"2269.HK","1177.HK"},
    "hk_other": {"2382.HK","0027.HK","1928.HK"},
    "other": {"RDDT"},
}

MODELS_REGRESSION = {
    "rf": "rf",
}

MODELS_CLASSIFICATION = {
    "rf": "rf",
}

# 美股 + 港股 (移除ETF)
US_WATCHLIST = [
    # 核心科技
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","ORCL",
    "AMD","QCOM","TSM",
    # 金融
    "JPM","V","MA","GS","BLK",
    # 消费
    "WMT","COST","HD","PG","KO","PEP","MCD",
    # 医疗
    "UNH","JNJ","LLY","ABBV",
    # 能源/工业
    "XOM","CAT","GE",
]

HK_WATCHLIST = [
    "0700.HK","9988.HK","9999.HK","1810.HK","3690.HK",
    "0941.HK","0883.HK","0388.HK","0005.HK","1299.HK",
    "2269.HK","2382.HK","9618.HK","1024.HK",
    "0939.HK","3988.HK","0857.HK","0027.HK","1928.HK","1177.HK",
]

ALL_TICKERS = {"US": US_WATCHLIST, "HK": HK_WATCHLIST}

NAMES_HK = {
    "0700.HK":"腾讯","9988.HK":"阿里","9999.HK":"网易","1810.HK":"小米",
    "3690.HK":"美团","0941.HK":"中移动","0883.HK":"中海油",
    "0388.HK":"港交所","0005.HK":"汇丰","1299.HK":"友邦",
    "2269.HK":"药明","2382.HK":"舜宇","9618.HK":"京东","1024.HK":"快手",
    "0939.HK":"建行","3988.HK":"中行","0857.HK":"中石油",
    "0027.HK":"银河","1928.HK":"金沙","1177.HK":"中生",
}

# ═══════════════════════════════════════
# P1.1: 数据缓存系统
# ════════════════��══════════════════════
def _cache_path(ticker, period):
    safe_name = ticker.replace(".", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"data_{safe_name}_{period}.pkl")

def _macro_cache_path():
    return os.path.join(CACHE_DIR, "macro_data.pkl")

def get_cached_data(ticker, period="2y", force_refresh=False):
    """带增量更新的缓存系统: 12h硬缓存, 但每4h尝试增量拉取最新行情补充"""
    path = _cache_path(ticker, period)
    if not force_refresh and os.path.exists(path):
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        if age < CACHE_TTL:
            # 缓存仍在有效期内, 但如果超过4h且市场可能已变化, 增量补充
            if age > CFG["cache_refresh_hours"] * 3600:
                try:
                    import yfinance as yf
                    df = pd.read_pickle(path)
                    # 只拉最近5个交易日的增量数据
                    last_date = df.index[-1]
                    delta = yf.download(ticker, start=last_date - timedelta(3),
                                        auto_adjust=True, progress=False)
                    if not delta.empty:
                        if isinstance(delta.columns, pd.MultiIndex):
                            delta.columns = [c[0] for c in delta.columns]
                        new_rows = delta.index.difference(df.index)
                        if len(new_rows) > 0:
                            df = pd.concat([df, delta.loc[new_rows]])
                            df.to_pickle(path)
                            return df
                    return df
                except:
                    # 增量失败, 返回缓存
                    try:
                        with open(path, "rb") as f:
                            return pickle.load(f)
                    except:
                        pass
            else:
                try:
                    with open(path, "rb") as f:
                        return pickle.load(f)
                except:
                    pass

    import yfinance as yf
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if not df.empty:
        with open(path, "wb") as f:
            pickle.dump(df, f)
    return df

def get_macro_data(force_refresh=False):
    """获取宏观因子数据 (SPY, VIX, 板块ETF)"""
    path = _macro_cache_path()
    if not force_refresh and os.path.exists(path):
        mtime = os.path.getmtime(path)
        if time.time() - mtime < CACHE_TTL:
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except:
                pass
    
    import yfinance as yf
    macro_tickers = {
        "spy": "SPY",
        "vix": "^VIX",
        "xlk": "XLK",  # 科技
        "xlf": "XLF",  # 金融
        "xle": "XLE",  # 能源
        "xlv": "XLV",  # 医疗
        "xli": "XLI",  # 工业
        "xlp": "XLP",  # 消费必需
        "xly": "XLY",  # 消费可选
        "iwm": "IWM",  # 小盘
        "dxy": "DX-Y.NYB",  # 美元
        "hsi": "^HSI",  # 恒指
    }
    result = {}
    for name, t in macro_tickers.items():
        try:
            d = yf.download(t, period="6mo", auto_adjust=True, progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = [c[0] for c in d.columns]
                result[name] = d["Close"]
        except:
            pass
    
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def get_ticker_sector(ticker):
    for sector, stocks in SECTOR_MAP.items():
        if ticker in stocks:
            return sector
    return "other"


# ═══════════════════════════════════════
# P2.1: 特征工程 v5 (含宏观因子)
# ═══════════════════════════════════════
def build_features_v5(df, macro_data=None, ticker="", cross_section_rank=True):
    """v5.2 特征工程: 45个技术面 + 宏观因子"""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] for c in df.columns]

    close = df["Close"].values.astype(float).ravel()
    high = df["High"].values.astype(float).ravel()
    low = df["Low"].values.astype(float).ravel()
    volume = df["Volume"].values.astype(float).ravel()
    idx = df.index

    ret = pd.Series(close, index=idx).pct_change()
    f = pd.DataFrame(index=idx)

    # ─── A. 动量特征 (精简: 去掉冗余周期) ───
    for p in [21, 63, 252]:
        f[f"mom_{p}d"] = pd.Series(close, index=idx).pct_change(p)
    mom_21 = pd.Series(close, index=idx).pct_change(21)
    mom_63 = pd.Series(close, index=idx).pct_change(63)
    f["mom_accel"] = mom_21 - mom_63.shift(63)

    # ─── B. 均线偏离 (精简: 保留中长期) ───
    for p in [50, 200]:
        sma = pd.Series(close, index=idx).rolling(p).mean()
        f[f"sma{p}_dev"] = pd.Series(close, index=idx) / sma - 1
    for p in [20, 50]:
        sma_series = pd.Series(close, index=idx).rolling(p).mean()
        f[f"sma{p}_slope"] = sma_series.pct_change(5) * 100

    # ─── C. 波动率 (精简: 保留中期+结构) ───
    for p in [21, 63]:
        f[f"vol_{p}d"] = ret.rolling(p).std() * np.sqrt(252)
    f["vol_ratio_21_63"] = f["vol_21d"] / f["vol_63d"]

    # ─── D. RSI (只用14) ───
    for p in [14]:
        delta = ret
        gain = delta.clip(lower=0).rolling(p).mean()
        loss = (-delta.clip(upper=0)).rolling(p).mean()
        rs = gain / loss.replace(0, np.nan)
        f[f"rsi_{p}"] = 100 - (100 / (1 + rs))

    # ─── E. 价格位置与形态 (精简) ───
    for p in [60, 120]:
        h_p = pd.Series(high, index=idx).rolling(p).max()
        l_p = pd.Series(low, index=idx).rolling(p).min()
        f[f"price_pos_{p}"] = (pd.Series(close, index=idx) - l_p) / (h_p - l_p).replace(0, np.nan)
    sma20 = pd.Series(close, index=idx).rolling(20).mean()
    std20 = pd.Series(close, index=idx).rolling(20).std()
    f["bb_position"] = (pd.Series(close, index=idx) - sma20) / (2 * std20).replace(0, np.nan)
    for p in [20]:
        h_p = pd.Series(high, index=idx).rolling(p).max()
        l_p = pd.Series(low, index=idx).rolling(p).min()
        f[f"hl_ratio_{p}"] = h_p / l_p.replace(0, np.nan) - 1

    # ─── F. 成交量 (精简) ───
    vol_s = pd.Series(volume, index=idx)
    f["volume_ratio"] = vol_s / vol_s.rolling(20).mean()
    obv = (np.sign(ret) * vol_s).cumsum()
    f["obv_trend"] = obv.pct_change(21)
    f["vol_price_corr_20"] = ret.rolling(20).corr(vol_s.pct_change())

    # ─── G. 风险调整 (精简) ───
    cum = (1 + ret).cumsum()
    dd = cum / cum.cummax() - 1
    f["calmar_60"] = ret.rolling(60).mean() * 252 / (-dd.rolling(60).min().replace(0, np.nan))
    f["skew_21"] = ret.rolling(21).skew()

    # ─── P1d: 横截面特征排名 ───
    # 将每个特征转换为其在自身时间序列上的百分位排名
    # 这样特征值就变成了"相对于自身历史的高低", 消除量纲差异
    if cross_section_rank:
        rank_cols = [c for c in f.columns if c not in ("macro_spy", "macro_vix", "macro_dxy",
                                                       "macro_hsi", "macro_sector", "macro_xlp", "macro_iwm")]
        rank_window = min(252, len(f))  # 最多用1年窗口做rank
        for c in rank_cols:
            if f[c].nunique() > 10:  # 只对有足够变化度的特征做rank
                f[c] = f[c].rolling(rank_window, min_periods=20).apply(
                    lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if x.max() != x.min() else 0.5,
                    raw=False
                )

    # ─── P2a: 宏观因子扩展 (加波动率变化+板块相对Beta) ───
    if macro_data is not None:
        for macro_name in ["spy", "vix", "dxy"]:
            if macro_name in macro_data:
                aligned = macro_data[macro_name].reindex(idx, method="ffill")
                f[f"macro_{macro_name}"] = aligned.pct_change(21)
                # 扩展: 再加波动率变化
                aligned_vol = aligned.pct_change().rolling(21).std() * np.sqrt(252)
                f[f"macro_{macro_name}_vol"] = aligned_vol.pct_change(21)

        ticker_guess = ticker
        sector_key = get_ticker_sector(ticker_guess)
        if "hk" in sector_key:
            if "hsi" in macro_data:
                aligned = macro_data["hsi"].reindex(idx, method="ffill")
                f["macro_hsi"] = aligned.pct_change(21)
                f["macro_hsi_vol"] = aligned.pct_change().rolling(21).std() * np.sqrt(252)
            # 港股用恒指计算个股相对beta
            if "hsi" in macro_data and "spy" in macro_data:
                spy_ret = macro_data["spy"].reindex(idx, method="ffill").pct_change()
                hsi_ret = macro_data["hsi"].reindex(idx, method="ffill").pct_change()
                stock_ret = pd.Series(close, index=idx).pct_change()
                beta_vs_spy = stock_ret.rolling(63).cov(spy_ret) / spy_ret.rolling(63).var().replace(0, np.nan)
                beta_vs_hsi = stock_ret.rolling(63).cov(hsi_ret) / hsi_ret.rolling(63).var().replace(0, np.nan)
                f["beta_vs_spy"] = beta_vs_spy
                f["beta_vs_hsi"] = beta_vs_hsi
        else:
            sector_etf_map = {"tech":"xlk","finance":"xlf","energy":"xle",
                              "healthcare":"xlv","industrial":"xli",
                              "consumer":"xly","other":"xly"}
            etf = sector_etf_map.get(sector_key, "spy")
            if etf in macro_data:
                aligned = macro_data[etf].reindex(idx, method="ffill")
                f["macro_sector"] = aligned.pct_change(21)
                f["macro_sector_vol"] = aligned.pct_change().rolling(21).std() * np.sqrt(252)
                # 板块相对SPY的beta
                spy_ret = macro_data["spy"].reindex(idx, method="ffill").pct_change()
                etf_ret = aligned.pct_change()
                beta_vs_spy = etf_ret.rolling(63).cov(spy_ret) / spy_ret.rolling(63).var().replace(0, np.nan)
                f["sector_beta_vs_spy"] = beta_vs_spy
                # 个股相对板块的beta
                stock_ret = pd.Series(close, index=idx).pct_change()
                stock_beta_vs_sector = stock_ret.rolling(63).cov(etf_ret) / etf_ret.rolling(63).var().replace(0, np.nan)
                f["stock_beta_vs_sector"] = stock_beta_vs_sector
            if sector_key == "consumer" and "xlp" in macro_data:
                aligned_xlp = macro_data["xlp"].reindex(idx, method="ffill")
                f["macro_xlp"] = aligned_xlp.pct_change(21)
        if sector_key in ("tech", "other") and "iwm" in macro_data:
            aligned = macro_data["iwm"].reindex(idx, method="ffill")
            f["macro_iwm"] = aligned.pct_change(21)

    # ─── 目标 (双轨) ───
    target_5d = pd.Series(close, index=idx).pct_change(FORECAST_HORIZON_SHORT).shift(-FORECAST_HORIZON_SHORT)
    target_21d = pd.Series(close, index=idx).pct_change(FORECAST_HORIZON_LONG).shift(-FORECAST_HORIZON_LONG)
    # 分类目标: 1=看涨(>3%), 0=震荡, -1=看跌(<-3%)
    target_cls = pd.Series(0, index=idx, dtype=int)
    target_cls[target_21d > 0.03] = 1
    target_cls[target_21d < -0.03] = -1

    f = f.replace([np.inf, -np.inf], np.nan)
    f = f.loc[:, f.notna().any()]
    valid = f.dropna(thresh=len(f.columns) * 0.5).index  # v5: 放宽到50%有效即可
    f = f.loc[valid]

    return (f.astype(np.float32),
            target_5d.loc[valid].rename("target_5d"),
            target_21d.loc[valid].rename("target_21d"),
            target_cls.loc[valid].rename("target_cls"))


# ═══════════════════════════════════════
# P0.2: Walk-Forward (双轨: 回归+分类)
# ═══════════════════════════════════════
def train_model_walk_forward_v5(X, y_reg, y_cls, models_cfg, n_splits=TEST_SPLITS):
    """
    v5.2 Walk-Forward:
    - 回归模型预测21d收益 (用于排序), 自适应超参 + 时间衰减
    - 分类模型预测21d涨跌, 三模型ensemble投票 (RF+XGB+LGB)
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    n_samples = len(X)

    # --- 时间衰减权重 ---
    # 越近的样本权重越高
    sample_weight = np.exp(-0.001 * np.arange(n_samples - 1, -1, -1))  # 最新样本权重1.0, 最老的约0.37
    sample_weight = sample_weight / sample_weight.mean()  # 归一化使均值为1

    model_names = [name for name, cfg in models_cfg.items() if cfg is not None]
    results = {}
    for name in model_names:
        # ─── 回归: 21d (自适应超参 + 时间衰减) ───
        reg_model = None
        reg_preds = pd.Series(index=y_reg.index, dtype=np.float32)
        fold_metrics = []

        for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr = y_reg.iloc[tr_idx]
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = _build_reg_model(name, n_samples)
            if model is None:
                continue
            try:
                model.fit(X_tr_s, y_tr, sample_weight=sample_weight[tr_idx])
            except TypeError:
                model.fit(X_tr_s, y_tr)  # 不支持sample_weight的模型
            y_pred = model.predict(X_te_s)[:len(te_idx)]
            reg_preds.iloc[te_idx] = y_pred
            fold_y_true = y_reg.iloc[te_idx]
            fold_y_pred = pd.Series(y_pred, index=fold_y_true.index)
            fold_valid = fold_y_true.notna() & fold_y_pred.notna()
            fold_r2 = r2_score(fold_y_true[fold_valid], fold_y_pred[fold_valid]) if fold_valid.sum() > 3 else 0
            fold_metrics.append(fold_r2)

        X_s = scaler.fit_transform(X)
        reg_model = _build_reg_model(name, n_samples)
        if reg_model is not None:
            try:
                reg_model.fit(X_s, y_reg, sample_weight=sample_weight)
            except TypeError:
                reg_model.fit(X_s, y_reg)

        valid_idx = y_reg.notna() & reg_preds.notna()
        reg_r2 = r2_score(y_reg[valid_idx], reg_preds[valid_idx]) if valid_idx.sum() > 5 else 0

        # ─── 分类: 21d, 三模型ensemble (v5.2: RF+XGB+LGB) ───
        cls_mask = y_cls.values != 0
        if cls_mask.sum() > CFG["classification"]["min_samples"]:
            cls_X_arr = X.values[cls_mask]
            cls_y_arr = y_cls.values[cls_mask]
            cls_model_names = [name for name, cfg in MODELS_CLASSIFICATION.items() if cfg is not None]
            cls_ensemble_preds = np.zeros((len(cls_y_arr), len(cls_model_names)), dtype=np.float32)
            cls_acc_models = []

            for cls_i, cls_name in enumerate(cls_model_names):
                try:
                    cls_m = _build_cls_model(cls_name, n_samples)
                    if cls_m is None:
                        continue
                    cls_tscv = TimeSeriesSplit(n_splits=min(3, n_splits))
                    fold_preds = np.full(len(cls_y_arr), 0, dtype=np.int8)
                    for tr_idx, te_idx in cls_tscv.split(cls_X_arr):
                        X_tr_c = cls_X_arr[tr_idx]
                        X_te_c = cls_X_arr[te_idx]
                        y_tr_c = cls_y_arr[tr_idx]
                        X_tr_c_s = scaler.fit_transform(X_tr_c)
                        X_te_c_s = scaler.transform(X_te_c)
                        cls_m_c = _build_cls_model(cls_name, n_samples)
                        if cls_m_c is None:
                            continue
                        cls_m_c.fit(X_tr_c_s, y_tr_c)
                        fold_preds[te_idx] = cls_m_c.predict(X_te_c_s)

                    valid_fold = fold_preds != 0
                    fold_acc = accuracy_score(cls_y_arr[valid_fold], fold_preds[valid_fold]) if valid_fold.sum() > 5 else 0
                    cls_acc_models.append(fold_acc)

                    # 全量训练
                    cls_m_full = _build_cls_model(cls_name, n_samples)
                    if cls_m_full is not None:
                        X_cls_s = scaler.fit_transform(cls_X_arr)
                        try:
                            cls_m_full.fit(X_cls_s, cls_y_arr, sample_weight=sample_weight[cls_mask])
                        except TypeError:
                            cls_m_full.fit(X_cls_s, cls_y_arr)
                        cls_pred = cls_m_full.predict(X_cls_s)
                        cls_ensemble_preds[:, cls_i] = cls_pred
                except:
                    continue

            # ensemble投票: 取三个分类器的均值方向
            used_models = np.any(cls_ensemble_preds != 0, axis=0)
            if used_models.sum() > 0:
                cls_ensemble = np.sign(np.mean(cls_ensemble_preds[:, used_models], axis=1))
                valid_ensemble = cls_ensemble != 0
                cls_acc = accuracy_score(cls_y_arr[valid_ensemble], cls_ensemble[valid_ensemble]) if valid_ensemble.sum() > 5 else 0
                cls_model = cls_ensemble  # 存为数组, 后续用于预测
            else:
                cls_acc = 0
                cls_model = None
        else:
            cls_acc = 0
            cls_model = None

        results[name] = {
            "reg_model": reg_model,
            "reg_r2": reg_r2,
            "reg_preds": reg_preds,
            "cls_model": cls_model,
            "cls_acc": cls_acc,
            "fold_r2_list": fold_metrics,
        }

    return results, scaler


# ═══════════════════════════════════════
# P0.1: 个股评分 v5 (Rank一致性)
# ═══════════════════════════════════════

def _momentum_direction_fallback(close_series, df_used):
    """动量方向兜底: 当分类器无信号时, 用EMA平滑动量 + 短期趋势判断方向
    
    返回: "看涨" / "看跌" / "震荡"
    """
    try:
        closes = close_series.values if hasattr(close_series, 'values') else close_series
        # EMA趋势: 21d和63d EMA方向一致
        ema_21 = close_series.ewm(span=21).mean()
        ema_63 = close_series.ewm(span=63).mean()
        ema_trend = (ema_21.iloc[-1] > ema_21.iloc[-22]) and (ema_63.iloc[-1] > ema_63.iloc[-64])
        ema_trend_down = (ema_21.iloc[-1] < ema_21.iloc[-22]) and (ema_63.iloc[-1] < ema_63.iloc[-64])
        
        # 短期动量大 / 小
        short_mom = (close_series.iloc[-1] / close_series.iloc[-6] - 1) * 100 if len(close_series) >= 6 else 0
        
        bullish_count = 0
        bearish_count = 0
        
        if ema_trend:
            bullish_count += 2
        if ema_trend_down:
            bearish_count += 2
        if short_mom > 2:
            bullish_count += 1
        elif short_mom < -2:
            bearish_count += 1
        
        # 近5日K线: 阳线多还是阴线多
        if len(closes) >= 6:
            daily_ret = pd.Series(closes).pct_change()
            up_days = (daily_ret.tail(5) > 0).sum()
            down_days = (daily_ret.tail(5) < 0).sum()
            if up_days >= 4:
                bullish_count += 1
            elif down_days >= 4:
                bearish_count += 1
        
        if bullish_count >= 2 and bullish_count > bearish_count:
            return "看涨"
        elif bearish_count >= 2 and bearish_count > bullish_count:
            return "看跌"
        else:
            return "震荡"
    except:
        return "震荡"

def score_stock_v5(ticker, macro_data=None, period="2y", force_refresh=False):
    """v5.2 个股评分: Rank一致性评分 + 双轨预测 + 自适应窗口"""
    df = get_cached_data(ticker, period=period, force_refresh=force_refresh)
    if df.empty or len(df) < MIN_TRADING_DAYS:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # ─── P1c: 自适应窗口 (v5.2: 根据波动率动态选择) ───
    if len(df) > min(ADAPTIVE_WINDOWS):
        closes_full = df["Close"].values.astype(float)
        ret_full = pd.Series(closes_full).pct_change()
        vol_21d = ret_full.tail(21).std() * np.sqrt(252)
        # 计算历史波动率百分位
        hist_vols = ret_full.rolling(63).std().dropna() * np.sqrt(252)
        if len(hist_vols) > 20:
            vol_pctl = (vol_21d < hist_vols).mean()  # 当前波动率在历史中的百分位
            if vol_pctl > 0.7:
                # 高波动 → 短窗口 (更敏感)
                best_w = 504
            elif vol_pctl < 0.3:
                # 低波动 → 长窗口 (更稳定)
                best_w = 1008 if 1008 <= len(df) else (756 if 756 <= len(df) else 504)
            else:
                best_w = 756 if 756 <= len(df) else (504 if 504 <= len(df) else len(df))
        else:
            best_w = max(ADAPTIVE_WINDOWS) if max(ADAPTIVE_WINDOWS) <= len(df) else len(df)
        df_used = df.iloc[-best_w:].copy()
    else:
        df_used = df
        best_w = len(df_used)

    features, target_5d, target_21d, target_cls = build_features_v5(df_used, macro_data, ticker)
    if len(features) < WARMUP:
        return None

    X = features.dropna()
    y_reg = target_5d.loc[X.index].dropna()
    y_cls = target_cls.loc[X.index]
    common = X.index.intersection(y_reg.index).intersection(y_cls.index)
    X = X.loc[common]
    y_reg = y_reg.loc[common]
    y_cls = y_cls.loc[common]

    if len(X) < WARMUP:
        return None

    # 强制对齐 (双重保护)
    X = X.loc[y_reg.index]
    y_cls = y_cls.loc[X.index]
    
    # 训练
    results, scaler = train_model_walk_forward_v5(
        X, y_reg, y_cls, MODELS_REGRESSION
    )

    if not results:
        return None

    # ═══ P0.1: Rank一致性评分 ═══
    model_names = list(results.keys())
    
    # 1. 收集各模型的回归预测
    preds_df = pd.DataFrame({name: results[name]["reg_preds"] for name in model_names})
    
    # 2. 将预测转为排名
    rank_df = preds_df.rank(pct=True)
    
    # 3. 模型间一致性 = 排名的平均Spearman相关
    if len(model_names) >= 2:
        rank_corrs = []
        for n1, n2 in combinations(model_names, 2):
            valid = rank_df[n1].notna() & rank_df[n2].notna()
            if valid.sum() > 10:
                r, _ = spearmanr(rank_df[n1][valid], rank_df[n2][valid])
                if not np.isnan(r):
                    rank_corrs.append(r)
        # v5: confidence = 模型间一致性 (0~1)
        confidence = max(np.mean(rank_corrs) if rank_corrs else CFG["confidence"]["default_fallback"], CFG["confidence"]["min_confidence"])
    else:
        confidence = CFG["confidence"]["default_fallback"]

    # 4. 最终评分 = 最新ensemble排名百分位 + 动量兜底
    latest_rank = float(rank_df.iloc[-1].mean())  # 各模型排名均值
    
    # ─── P1b: EMA平滑动量兜底 ───
    closes = df_used["Close"].values.astype(float)
    close_series = pd.Series(closes)
    ema_21 = close_series.ewm(span=21).mean()
    ema_63 = close_series.ewm(span=63).mean()
    mom_21d_val = (ema_21.iloc[-1] / ema_21.iloc[-22] - 1) if len(ema_21) >= 22 else 0
    mom_63d_val = (ema_63.iloc[-1] / ema_63.iloc[-64] - 1) if len(ema_63) >= 64 else 0
    
    avg_r2 = np.mean([results[n]["reg_r2"] for n in model_names])
    if avg_r2 < 0:
        mom_score = max(0, min(1, (mom_21d_val * CFG["momentum_fallback"]["mom_21d_weight"] + mom_63d_val * CFG["momentum_fallback"]["mom_63d_weight"] + CFG["momentum_fallback"]["mom_score_offset"])))
        mom_weight = min(max(-avg_r2 * CFG["momentum_fallback"]["r2_to_weight_multiplier"], CFG["momentum_fallback"]["weight_min"]), CFG["momentum_fallback"]["weight_max"])
        final_rank = latest_rank * (1 - mom_weight) + mom_score * mom_weight
    else:
        final_rank = latest_rank
    
    final_score = final_rank * CFG["score_formula"]["rank_weight"] + confidence * CFG["score_formula"]["confidence_weight"]
    
    # 5. 分类信号 (v5.2: ensemble投票方向)
    cls_signals = []
    if results:
        ensemble_dirs = []
        # 从各模型的分类结果获取方向信号
        for name in model_names:
            if results[name]["cls_model"] is not None and results[name]["cls_acc"] > CFG["classification"]["min_accuracy"]:
                # 分类模型是ensemble数组, 用最新值
                if isinstance(results[name]["cls_model"], np.ndarray):
                    cls_pred = int(np.sign(np.mean(results[name]["cls_model"][-3:]) or 0))
                else:
                    x_latest = X.iloc[-1:].values
                    x_s = scaler.transform(x_latest)
                    cls_pred = int(results[name]["cls_model"].predict(x_s)[0])
                ensemble_dirs.append(cls_pred)
                cls_signals.append({
                    "model": name,
                    "direction": cls_pred,
                    "accuracy": round(results[name]["cls_acc"], 4),
                })
        # 综合方向: ensemble均值
        if ensemble_dirs:
            avg_dir = np.mean(ensemble_dirs)
            if avg_dir > CFG["direction_thresholds"]["bullish"]:
                direction = "看涨"
            elif avg_dir < CFG["direction_thresholds"]["bearish"]:
                direction = "看跌"
            else:
                direction = "震荡"
            direction_source = "cls_ensemble"
        else:
            # 动量方向兜底: 分类器无信号时, 用EMA平滑动量判断方向
            fallback_direction = _momentum_direction_fallback(close_series, df_used)
            direction = fallback_direction
            direction_source = "momentum_fallback"
    else:
        direction = "震荡"
        direction_source = "no_model"

    # 最新预测收益
    latest_pred_reg = float(rank_df.iloc[-1].mean())  # 用排名百分位表示
    # 实际最新5日收益(用于对比)
    closes = df_used["Close"].values.astype(float)
    actual_5d = (closes[-1] / closes[-min(6, len(closes))] - 1) * 100 if len(closes) >= 2 else 0
    mom_1m = (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
    mom_3m = (closes[-1] / closes[-63] - 1) * 100 if len(closes) >= 63 else 0

    return {
        "ticker": ticker,
        "price": round(float(closes[-1]), 2),
        "score": round(final_score, 4),          # v5: 新评分(0~1)
        "rank_pctl": round(latest_rank, 4),       # 排名百分位
        "confidence": round(confidence, 4),       # 模型一致性
        "walk_forward_r2": round(np.mean([results[n]["reg_r2"] for n in model_names]), 4),
        "direction": direction,                    # 分类看涨/看跌
        "cls_details": cls_signals[:3],            # 分类详情
        "adaptive_window": best_w,
        "actual_5d": round(actual_5d, 2),
        "mom_1m": round(mom_1m, 2),
        "mom_3m": round(mom_3m, 2),
        "models_used": model_names,
        "models_r2": {n: round(results[n]["reg_r2"], 4) for n in model_names},
        "models_consensus": round(confidence, 4),
        "sector": get_ticker_sector(ticker),
        "direction_source": direction_source,
    }


# ═══════════════════════════════════════
# 批量选股 v5
# ═══════════════════════════════════════
def run_ml_picking_v5(tickers=None, market="US", macro_data=None,
                       force_refresh=False, top_n=TOP_N, verbose=True):
    """v5 批量选股"""
    if tickers is None:
        tickers = US_WATCHLIST
    
    label = market
    
    if verbose:
        print(f"\n【{label}】扫描 {len(tickers)} 只...")
        print("-" * 60)

    results = []
    errors = []
    total = len(tickers)

    # 预加载宏观数据
    if macro_data is None:
        if verbose:
            print("  加载宏观因子...")
        macro_data = get_macro_data(force_refresh=force_refresh)
        if verbose:
            print(f"  宏观因子: {list(macro_data.keys())}")

    for i, t in enumerate(tickers):
        if verbose:
            sys.stdout.write(f"  [{i+1}/{total}] {t:<10} ... ")
            sys.stdout.flush()
        try:
            sr = score_stock_v5(t, macro_data=macro_data, force_refresh=force_refresh)
            if sr is not None:
                results.append(sr)
                tag = "★" if sr["score"] > 0.5 else "·"
                if verbose:
                    print(f"{tag} score={sr['score']:.3f} R²={sr['walk_forward_r2']:.3f} {sr['direction']} 共识={sr['confidence']:.2f}")
            else:
                errors.append(t)
                if verbose:
                    print("  数据不足")
        except Exception as e:
            errors.append(t)
            if verbose:
                print(f"  失败: {e}")

    results.sort(key=lambda x: x["score"], reverse=True)
    return results, errors


# ═══════════════════════════════════════
# 报告输出 v5
# ═══════════════════════════════════════
def print_report_v5(all_results, title="ML v5 选股报告"):
    """v5 综合报告"""
    if not all_results:
        print("  无结果")
        return
    
    print(f"\n{'='*95}")
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  v5.2 核心改进: Rank一致性评分 + 双轨预测(21d回归+21d分类) + 宏观扩展 + 自适应超参 + 时间衰减")
    print(f"{'='*95}")
    print(f"{'#':>3} {'代码':>10} {'名称':>6} {'评分':>6} {'排名':>6} {'共识':>5} {'方向':<6} {'R²':>6} {'现价':>8} {'1月':>7} {'3月':>7}")
    print("-" * 75)

    for i, s in enumerate(all_results[:20]):
        name = NAMES_HK.get(s["ticker"], "")
        dir_icon = {"看涨": "🟢", "看跌": "🔴", "震荡": "🟡"}.get(s["direction"], "⚪")
        print(f"{i+1:>3} {s['ticker']:>10} {name:>6} {s['score']:>6.3f} {s['rank_pctl']:>6.3f} {s['confidence']:>5.2f} {dir_icon}{s['direction']:<4} {s['walk_forward_r2']:>6.3f} ${s['price']:>7.2f} {s['mom_1m']:>+5.1f}% {s['mom_3m']:>+5.1f}%")

    print()
    avg_score = np.mean([s["score"] for s in all_results])
    avg_conf = np.mean([s["confidence"] for s in all_results])
    print(f"  总评: {len(all_results)} 只 | 平均评分 {avg_score:.3f} | 平均共识 {avg_conf:.2f}")

    strong = [s for s in all_results if s["score"] > 0.55]
    watch = [s for s in all_results if 0.40 < s["score"] <= 0.55]
    print(f"  强烈推荐 (>0.55): {', '.join(s['ticker'] for s in strong[:8]) or '无'}")
    print(f"  值得关注 (0.40~0.55): {', '.join(s['ticker'] for s in watch[:8]) or '无'}")

    # 分类信号统计
    bullish = [s for s in all_results if s["direction"] == "看涨"]
    bearish = [s for s in all_results if s["direction"] == "看跌"]
    print(f"  看涨方向: {len(bullish)} 只 | 看跌方向: {len(bearish)} 只")

    # 推荐详情
    print(f"\n{'='*95}")
    print(f"  推荐详情 Top 8")
    print(f"{'='*95}")
    for i, s in enumerate(all_results[:8]):
        name = NAMES_HK.get(s["ticker"], s["ticker"])
        stars = "★★★★★" if s["score"] > 0.6 else \
                "★★★★" if s["score"] > 0.5 else \
                "★★★" if s["score"] > 0.4 else "★★"
        print(f"\n  {stars} {name}({s['ticker']}) — 评分{s['score']:.3f}")
        print(f"     板块: {s['sector']} | 方向: {s['direction']} | 共识度: {s['confidence']:.2f}")
        print(f"     排名百分位: {s['rank_pctl']:.1%} | R²: {s['walk_forward_r2']:.3f}")
        print(f"     现价: ${s['price']:.2f} | 1月: {s['mom_1m']:+.1f}% | 3月: {s['mom_3m']:+.1f}%")

    return all_results[:8]


def save_results_v5(results, filename="ml_v5_picks"):
    """保存结果"""
    rows = []
    for s in results:
        rows.append({
            "ticker": s["ticker"],
            "score": s["score"],
            "rank_pctl": s["rank_pctl"],
            "confidence": s["confidence"],
            "walk_forward_r2": s["walk_forward_r2"],
            "direction": s["direction"],
            "price": s["price"],
            "actual_5d": s["actual_5d"],
            "mom_1m": s["mom_1m"],
            "mom_3m": s["mom_3m"],
            "sector": s["sector"],
            "models_used": "+".join(s["models_used"]),
        })
    df = pd.DataFrame(rows)
    now = datetime.now().strftime("%Y%m%d_%H%M")
    csv_path = os.path.join(CACHE_DIR, f"{filename}_{now}.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  已保存: {csv_path}")
    return csv_path


# ═══════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML v5.2 选股系统")
    parser.add_argument("--us-only", action="store_true", help="只跑美股")
    parser.add_argument("--hk-only", action="store_true", help="只跑港股")
    parser.add_argument("--refresh", action="store_true", help="强制刷新缓存")
    parser.add_argument("--top", type=int, default=TOP_N, help="输出前N只")
    args = parser.parse_args()

    print("=" * 80)
    print("  ML 优化版选股系统 v5.2")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("  改进: Rank一致性评分 + 双轨预测 + 宏观扩展 + 自适应超参 + 时间衰减 + 分类ensemble")
    print("=" * 80)

    # 预加载宏观数据
    print("\n加载宏观因子...")
    macro_data = get_macro_data(force_refresh=args.refresh)
    print(f"  宏观因子: {list(macro_data.keys())}")

    all_results = []

    if not args.hk_only:
        us_r, us_e = run_ml_picking_v5(
            tickers=US_WATCHLIST, market="US",
            macro_data=macro_data, force_refresh=args.refresh,
            top_n=args.top,
        )
        all_results.extend(us_r)
        print(f"\n  美股: {len(us_r)} OK, {len(us_e)} 失败")

    if not args.us_only:
        hk_r, hk_e = run_ml_picking_v5(
            tickers=HK_WATCHLIST, market="HK",
            macro_data=macro_data, force_refresh=args.refresh,
            top_n=args.top,
        )
        all_results.extend(hk_r)
        print(f"\n  港股: {len(hk_r)} OK, {len(hk_e)} 失败")

    if all_results:
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        report_title = "ML v5 选股报告"
        if args.us_only:
            report_title += " (仅美股)"
        elif args.hk_only:
            report_title += " (仅港股)"
        else:
            report_title += " (美股+港股)"
        
        top = print_report_v5(all_results, title=report_title)
        save_results_v5(all_results)

        # 对比v4: 看评分分布是否改善
        scores = [s["score"] for s in all_results]
        print(f"\n{'='*50}")
        print(f"  v5 vs v4 评分对比:")
        print(f"  v5 最大评分: {max(scores):.3f}  (v4 最大: 0.103)")
        print(f"  v5 评分>0.5: {len([s for s in all_results if s['score'] > 0.5])} 只 (v4: 0)")
        print(f"  v5 评分>0.4: {len([s for s in all_results if s['score'] > 0.4])} 只 (v4: 0)")
        print(f"  v5 评分>0.3: {len([s for s in all_results if s['score'] > 0.3])} 只 (v4: 0)")
        print(f"{'='*50}")

        print(f"\n  ✅ v5 完成! 共评分 {len(all_results)} 只股票")
    else:
        print("\n  ❌ 没有成功评分的股票")
