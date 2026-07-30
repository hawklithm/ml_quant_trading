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
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            full = json.load(f)
        _CFG = full["ml_scoring"]
    return _CFG

CFG = load_config()

warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_TTL = CFG["cache_ttl_hours"] * 3600  # 从配置读取

# ──── 全局限流控制 ────
# 一旦 YF 返回 rate-limit 错误，本运行周期内跳过所有后续下载，直接回退缓存
_YF_GLOBAL_RATELIMITED = [False]  # 用 list 包装使可被函数内修改
_YF_SHARED_SESSION = None

def _get_yf_session():
    """获取/创建共享 requests.Session，带重试策略"""
    global _YF_SHARED_SESSION
    if _YF_SHARED_SESSION is None:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        _YF_SHARED_SESSION = requests.Session()
        retry_strategy = Retry(
            total=2, backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        _YF_SHARED_SESSION.mount('https://', adapter)
        _YF_SHARED_SESSION.mount('http://', adapter)
    return _YF_SHARED_SESSION

def _is_yf_ratelimited():
    return _YF_GLOBAL_RATELIMITED[0]

def _set_yf_ratelimited():
    _YF_GLOBAL_RATELIMITED[0] = True

def _yf_quick_probe():
    """快速探测 YF 是否可用 — 用轻量 HTTP 请求检查, 不 yf.download
    已标记限流则跳过探测"""
    if _is_yf_ratelimited():
        return False
    try:
        import requests
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/SPY?interval=1d&range=1d",
            timeout=3,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        ok = r.status_code == 200
        if not ok:
            _set_yf_ratelimited()
        return ok
    except Exception:
        _set_yf_ratelimited()
        return False

def _download_yf(tickers, **kwargs):
    """带 session 复用和全局限流检测的 yf.download 封装
    单只股票用 Ticker.history() 避免 yf.download() 的 curl_cffi 兼容问题
    批量下载请呼叫我时带上 _batch=True """
    if _is_yf_ratelimited():
        return pd.DataFrame()
    kwargs.pop('session', None)  # 不传 session，yfinance 1.4.1 用 curl_cffi 而非 requests
    kwargs.setdefault('timeout', 10)
    try:
        import yfinance as yf
        import time as _ytime
        _ytime.sleep(0.3)

        if kwargs.pop('_batch', False) or isinstance(tickers, list):
            # 批量多只 → 用 yf.download（不带自定义 session）
            df = yf.download(tickers, **kwargs)
        else:
            # 单只 → 用 Ticker.history（100% 兼容港股长周期）
            tk = yf.Ticker(tickers)
            # 提取兼容参数
            hist_kw = {}
            for k in ('period', 'start', 'end', 'interval', 'auto_adjust',
                       'back_adjust', 'round', 'actions'):
                if k in kwargs:
                    hist_kw[k] = kwargs[k]
            df = tk.history(**hist_kw)
            # 统一列名（Ticker.history 返回小写列名, 但代码用大写 "Close"）
            if df is not None and not df.empty and hasattr(df, 'columns'):
                if df.columns[0].islower():
                    cap_map = {c: c.title() for c in df.columns}
                    df = df.rename(columns=cap_map)
        # 统一时区: yf.download / Ticker.history 都可能返回带时区的时间戳
        if df is not None and not df.empty and hasattr(df, 'index'):
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        err_str = str(e)
        if any(kw in err_str for kw in ['Rate', '429', 'limit',
                                         'Connection', 'Timeout', 'Remote']):
            _set_yf_ratelimited()
        return pd.DataFrame()

# ──── ML 模型 ────
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
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
PREDICTION_HORIZONS = tuple(CFG.get("prediction_horizons", [FORECAST_HORIZON_SHORT, FORECAST_HORIZON_LONG]))
PURGE_GAP_DAYS = int(CFG.get("purge_gap_days", max(PREDICTION_HORIZONS)))
BENCHMARK_BY_MARKET = CFG.get("benchmark_by_market", {"US": "spy", "HK": "hsi"})
# 新增配置
MARKET_REGIME_CFG = CFG.get("market_regime", {})
STOCK_PENALTY_CFG = CFG.get("stock_penalty", {})
SCORE_SPREAD_CFG = CFG.get("score_spread", {})
SENTIMENT_V2_CFG = CFG.get("sentiment_v2", {})

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
    "xgb": "xgb",
    "lgb": "lgb",
}

MODELS_CLASSIFICATION = {
    "rf": "rf",
    "xgb": "xgb",
    "lgb": "lgb",
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


def get_cache_metadata(ticker, period="2y"):
    """Return cache age metadata without changing the DataFrame API."""
    path = _cache_path(ticker, period)
    if not os.path.exists(path):
        return {"cache_path": path, "cache_age_hours": None, "stale": None}
    age_hours = max(0.0, (time.time() - os.path.getmtime(path)) / 3600.0)
    return {
        "cache_path": path,
        "cache_age_hours": round(age_hours, 3),
        "stale": age_hours * 3600 > CACHE_TTL,
    }

def _macro_cache_path():
    return os.path.join(CACHE_DIR, "macro_data.pkl")

def get_cached_data(ticker, period="2y", force_refresh=False):
    """带增量更新的缓存系统: 12h硬缓存, 但每8h尝试增量拉取最新行情补充
    港股走腾讯财经, 美股走 YF/YF限流回退缓存"""
    path = _cache_path(ticker, period)
    
    # ─── 港股 → 走腾讯财经 ───
    if ticker.endswith(".HK"):
        from tencent_data import get_hk_kline_data
        return get_hk_kline_data(ticker, force_refresh=force_refresh, cache_ttl_hours=CFG["cache_ttl_hours"])
    
    # ─── 美股 → 走 YF，但有缓存优先逻辑 ───
    # 先检查缓存
    if os.path.exists(path) and not force_refresh:
        try:
            with open(path, "rb") as f:
                cached = pickle.load(f)
            mtime = os.path.getmtime(path)
            age = time.time() - mtime
            
            # 缓存12小时内 → 直接返回
            if age < CACHE_TTL:
                return cached
            
            # 缓存超过12小时但不超过5天 → 尝试快速增量
            if age < 5 * 24 * 3600:
                # 先做快速探测，探测快速失败就回退缓存
                if not _yf_quick_probe():
                    return cached  # YF不可用，直接回退
                # YF可用 → 尝试增量刷新
                try:
                    last_date = cached.index[-1]
                    delta = _download_yf(ticker, start=last_date - timedelta(3),
                                         auto_adjust=True, progress=False)
                    if not delta.empty:
                        if isinstance(delta.columns, pd.MultiIndex):
                            delta.columns = [c[0] for c in delta.columns]
                        new_rows = delta.index.difference(cached.index)
                        if len(new_rows) > 0:
                            df = pd.concat([cached, delta.loc[new_rows]])
                            with open(path, "wb") as f:
                                pickle.dump(df, f)
                            return df
                    return cached
                except (OSError, ValueError, EOFError, pickle.UnpicklingError):
                    return cached  # 增量失败，还是返回缓存
            
            # 缓存超过5天 → 尝试实时拉取，失败才回退
            pass  # 继续下面的实时拉取逻辑
        except Exception:
            pass
    
    # 实时拉取 — 先快速探测
    if not _yf_quick_probe():
        # YF不可用 → 尝试akShare备用
        try:
            import akshare as ak
            from datetime import datetime
            df_ak = ak.stock_us_daily(symbol=ticker, adjust='')
            if not df_ak.empty and len(df_ak) > 10:
                # 转换列名: date→Date, open→Open 等
                cols_map = {'date': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
                df_ak = df_ak.rename(columns=cols_map)
                df_ak['Date'] = pd.to_datetime(df_ak['Date'])
                df_ak = df_ak.set_index('Date').sort_index()
                # 只保留2年数据
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=730)
                df_ak = df_ak[df_ak.index >= cutoff]
                if not df_ak.empty:
                    with open(path, "wb") as f:
                        pickle.dump(df_ak, f)
                    return df_ak
        except Exception:
            pass
        
        # 有缓存就回退
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except (OSError, EOFError, pickle.UnpicklingError) as exc:
                print(f"  cache read failed: {exc}")
        return pd.DataFrame()
    
    df = _download_yf(ticker, period=period, auto_adjust=True, progress=False)
    if not df.empty:
        with open(path, "wb") as f:
            pickle.dump(df, f)
        return df
    
    # 实时拉取失败 → 回退到过期缓存
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except Exception:
            pass
    return pd.DataFrame()

def get_macro_data(force_refresh=False):
    """获取宏观因子数据 (SPY, VIX, 板块ETF) — 批量下载优化"""
    path = _macro_cache_path()
    if not force_refresh and os.path.exists(path):
        mtime = os.path.getmtime(path)
        if time.time() - mtime < CACHE_TTL:
            try:
                with open(path, "rb") as f:
                    result = pickle.load(f)
                # 统一时区: 确保所有宏观因子Series是tz-naive
                for k in result:
                    if hasattr(result[k], 'index') and hasattr(result[k].index, 'tz') and result[k].index.tz is not None:
                        result[k].index = result[k].index.tz_localize(None)
                return result
            except (OSError, EOFError, pickle.UnpicklingError) as exc:
                print(f"  macro cache read failed: {exc}")
    
    macro_tickers = {
        "spy": "SPY",
        "vix": "^VIX",
        "xlk": "XLK",  # 科技
        "xlv": "XLV",  # 医疗
        "xli": "XLI",  # 工业
        "xlp": "XLP",  # 消费必需
        "xly": "XLY",  # 消费可选
        "iwm": "IWM",  # 小盘
        "dxy": "DX-Y.NYB",  # 美元
        "hsi": "^HSI",  # 恒指（走腾讯财经）
    }
    
    result = {}
    symbols_for_yf = list(macro_tickers.values())
    
    # 快速探测: 先试一次是否全局限流, 避免12个ticker逐个超时
    if not _yf_quick_probe() and _is_yf_ratelimited():
        print(f"  ⚠️  YF 不可用，快速回退到过期缓存")
        # 至少尝试从腾讯财经获取HSI
        hsi = None
        try:
            from tencent_data import fetch_hsi_kline
            hsi_series = fetch_hsi_kline(500)
            if not hsi_series.empty:
                print(f"  📡 已从腾讯财经获取HSI替代 ({len(hsi_series)}行)")
                hsi = hsi_series
        except (ImportError, OSError, ValueError) as exc:
            print(f"  HSI fallback unavailable: {exc}")
        
        if os.path.exists(path):
            try:
                cached = pickle.load(open(path, "rb"))
                # 统一时区
                for k in cached:
                    if hasattr(cached[k], 'index') and hasattr(cached[k].index, 'tz') and cached[k].index.tz is not None:
                        cached[k].index = cached[k].index.tz_localize(None)
                # 回退前的NaN清洗: ffill避免尾行NaN传播
                for k in list(cached.keys()):
                    s = cached[k]
                    if s.notna().sum() < 5:
                        del cached[k]
                    else:
                        cached[k] = s.ffill()
                # 如果腾讯HSI可用，替换或补充缓存中的HSI
                try:
                    from tencent_data import fetch_hsi_kline
                    if not hsi.empty:
                        cached['hsi'] = hsi['Close']
                        print(f"  ✅ HSI已刷新 ({len(hsi)}行)")
                        with open(path, "wb") as f:
                            pickle.dump(cached, f)
                except (ImportError, OSError, ValueError) as exc:
                    print(f"  HSI cache refresh failed: {exc}")
                # 同时尝试akShare刷新SPY/XLK等ETF宏观因子
                try:
                    import akshare as ak
                    refreshed = 0
                    for name, sym in macro_tickers.items():
                        if name in ('hsi', 'vix', 'dxy') or name in cached:
                            continue
                        try:
                            df = ak.stock_us_daily(symbol=sym, adjust='')
                            if not df.empty:
                                s = df.set_index('date')['close']
                                s.index = pd.to_datetime(s.index)
                                s = s.sort_index()
                                if hasattr(s.index, 'tz') and s.index.tz is not None:
                                    s.index = s.index.tz_localize(None)
                                cached[name] = s
                                refreshed += 1
                        except (ImportError, OSError, ValueError) as exc:
                            print(f"  ETF macro refresh failed for {name}: {exc}")
                    if refreshed > 0:
                        print(f"  ✅ akShare刷新{refreshed}个ETF因子")
                        with open(path, "wb") as f:
                            pickle.dump(cached, f)
                except (OSError, ValueError, pickle.PicklingError) as exc:
                    print(f"  macro cache write failed: {exc}")
                return cached
            except (ImportError, OSError, ValueError) as exc:
                print(f"  akShare macro fallback failed: {exc}")
        # 无缓存但有腾讯HSI
        try:
            if not hsi.empty:
                result = {'hsi': hsi['Close']}
                with open(path, "wb") as f:
                    pickle.dump(result, f)
                print(f"  ✅ 纯腾讯HSI因子 ({len(hsi)}行)")
                return result
        except (OSError, ValueError, KeyError) as exc:
            print(f"  HSI macro cache write failed: {exc}")
        return {}
    
    # 批量下载 — 使用共享 session
    batch = _download_yf(symbols_for_yf, period="6mo", auto_adjust=True,
                         progress=False, group_by="ticker", _batch=True)
    if not batch.empty:
        for name, sym in macro_tickers.items():
            try:
                if isinstance(batch.columns, pd.MultiIndex) and batch.columns.nlevels > 1:
                    close_series = batch.xs("Close", axis=1, level=1)[sym]
                else:
                    continue
                if close_series is not None and not close_series.empty:
                    result[name] = close_series
            except Exception:
                pass
    
    if result:
        # 统一时区: 确保所有宏观因子Series是tz-naive
        for k in result:
            if hasattr(result[k], 'index') and hasattr(result[k].index, 'tz') and result[k].index.tz is not None:
                result[k].index = result[k].index.tz_localize(None)
        # 清洗NaN: YF在非交易日返回NaN, ffill确保尾行有效
        # 如果整个series全NaN(如已退市ETF), 丢弃
        for k in list(result.keys()):
            s = result[k]
            if s.notna().sum() < 5:
                del result[k]
                print(f"  ⚠️ 丢弃 {k}: 有效数据不足5行")
            else:
                result[k] = s.ffill()
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result
    
    # 批量下载失败 → 尝试akShare备用
    if not _is_yf_ratelimited():
        # YF不是限流引起的失败，可能只是批量下载超时，尝试akShare
        pass
    try:
        import akshare as ak
        print(f"  📡 尝试akShare备用获取宏观因子...")
        ak_ok = 0
        for name, sym in macro_tickers.items():
            if name in result:
                continue
            if sym.startswith("^"):
                continue  # VIX/恒指不走akShare
            if sym == "DX-Y.NYB":
                continue  # DXY不走akShare
            try:
                df = ak.stock_us_daily(symbol=sym, adjust='')
                if not df.empty:
                    close_series = df.set_index('date')['close']
                    close_series.index = pd.to_datetime(close_series.index)
                    close_series = close_series.sort_index()
                    if hasattr(close_series.index, 'tz') and close_series.index.tz is not None:
                        close_series.index = close_series.index.tz_localize(None)
                    result[name] = close_series
                    ak_ok += 1
            except (ImportError, OSError, ValueError) as exc:
                print(f"  akShare fallback failed for {name}: {exc}")
        
        if result:
            print(f"  ✅ akShare备用获取 {ak_ok}/{len(macro_tickers)} 个因子")
            for k in result:
                if hasattr(result[k], 'index') and hasattr(result[k].index, 'tz') and result[k].index.tz is not None:
                    result[k].index = result[k].index.tz_localize(None)
            with open(path, "wb") as f:
                pickle.dump(result, f)
            return result
    except (ImportError, OSError, ValueError) as exc:
        print(f"  macro data refresh failed: {exc}")
    
    # 实时拉取完全失败 → 立即回退过期缓存，不再逐个重试
    if os.path.exists(path):
        try:
            cached = pickle.load(open(path, "rb"))
            # 统一时区: 确保所有宏观因子Series是tz-naive
            for k in cached:
                if hasattr(cached[k], 'index') and hasattr(cached[k].index, 'tz') and cached[k].index.tz is not None:
                    cached[k].index = cached[k].index.tz_localize(None)
            # 过期缓存也需要NaN清洗
            for k in list(cached.keys()):
                s = cached[k]
                if s.notna().sum() < 5:
                    del cached[k]
                else:
                    cached[k] = s.ffill()
            print(f"  ⚠️  YF 实时拉取失败，回退到过期缓存 ({len(cached)}个因子)")
            return cached
        except (OSError, ValueError, KeyError) as exc:
            print(f"  stale macro cache invalid: {exc}")
    return {}

def get_ticker_sector(ticker):
    for sector, stocks in SECTOR_MAP.items():
        if ticker in stocks:
            return sector
    return "other"


# ═══════════════════════════════════════
# P2.1: 特征工程 v5 (含宏观因子)
# ═══════════════════════════════════════
def build_features_v5(df, macro_data=None, ticker="", market="US", cross_section_rank=True):
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

    # ─── A. 动量特征 (精简: 去掉冗余周期) — v5.4: 相对SPY Alpha化 ───
    # 之前改了目标(y=alpha)但特征还是绝对动量 → 特征-目标体系错配
    # 现在动量特征也改为相对SPY的Alpha版本，与目标体系一致
    spy_close = None
    if macro_data is not None and "spy" in macro_data:
        spy_close = macro_data["spy"].reindex(idx, method="ffill")
    
    for p in [21, 63, 252]:
        stock_pct = pd.Series(close, index=idx).pct_change(p)
        if spy_close is not None:
            spy_pct = spy_close.pct_change(p)
            # SPY数据若覆盖不到股票早期，退回到绝对动量
            alpha = stock_pct - spy_pct
            alpha_nan = alpha.isna()
            stock_nan = stock_pct.isna()
            # 只有SPY引入额外NaN时才退回
            extra_nan = alpha_nan & ~stock_nan
            if extra_nan.any():
                alpha = alpha.copy()
                alpha[extra_nan] = stock_pct[extra_nan]
            f[f"mom_{p}d"] = alpha
        else:
            f[f"mom_{p}d"] = stock_pct  # 退回到绝对动量
    
    # mom_accel也用Alpha版本
    mom_21 = f["mom_21d"].copy()
    mom_63 = f["mom_63d"].copy()
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
            # 优化4: 港股金融板块加dxy/vix因子反映利率/汇率风险
            import re as _re
            if sector_key == "hk_finance" or _re.match(r"0005|1299|0939|3988", ticker_guess.split(".")[0]):
                if "dxy" in macro_data:
                    f["macro_dxy_hkf"] = macro_data["dxy"].reindex(idx, method="ffill").pct_change(21)
                if "vix" in macro_data:
                    f["macro_vix_hkf"] = macro_data["vix"].reindex(idx, method="ffill").pct_change(21)
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

    # ─── 目标 (双轨) — v5.3: 改为相对SPY超额收益(Alpha) ───
    # 个股绝对收益80%是大盘贡献的，去掉大盘因素后R²有望转正
    benchmark_name = BENCHMARK_BY_MARKET.get(str(market).upper(), "spy")
    benchmark = macro_data.get(benchmark_name) if macro_data else None
    stock_21d = pd.Series(close, index=idx).pct_change(FORECAST_HORIZON_LONG)
    if macro_data and "spy" in macro_data:
        spy_21d = macro_data["spy"].reindex(idx, method="ffill").pct_change(FORECAST_HORIZON_LONG)
        alpha_21d = stock_21d - spy_21d  # 相对收益
    else:
        alpha_21d = stock_21d  # 退回到绝对收益
    target_21d = alpha_21d.shift(-FORECAST_HORIZON_LONG)  # 前移作为预测目标
    
    stock_5d = pd.Series(close, index=idx).pct_change(FORECAST_HORIZON_SHORT)
    if macro_data and "spy" in macro_data:
        spy_5d = macro_data["spy"].reindex(idx, method="ffill").pct_change(FORECAST_HORIZON_SHORT)
        alpha_5d = stock_5d - spy_5d
    else:
        alpha_5d = stock_5d
    target_5d = alpha_5d.shift(-FORECAST_HORIZON_SHORT)

    # Recompute relative targets with the market-specific benchmark.
    if benchmark is not None and benchmark_name != "spy":
        benchmark_21d = benchmark.reindex(idx, method="ffill").pct_change(FORECAST_HORIZON_LONG)
        benchmark_5d = benchmark.reindex(idx, method="ffill").pct_change(FORECAST_HORIZON_SHORT)
        target_21d = (stock_21d - benchmark_21d).shift(-FORECAST_HORIZON_LONG)
        target_5d = (stock_5d - benchmark_5d).shift(-FORECAST_HORIZON_SHORT)
    elif benchmark_name != "spy" and benchmark is None:
        # Do not silently use a US benchmark for a non-US market.
        target_21d = stock_21d.shift(-FORECAST_HORIZON_LONG)
        target_5d = stock_5d.shift(-FORECAST_HORIZON_SHORT)
    
    # 分类目标: 基于相对收益（个股是否跑赢/跑输大盘±3%）
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
def make_purged_time_splits(n_samples, n_splits=TEST_SPLITS, gap=PURGE_GAP_DAYS):
    """Create expanding time splits with a purge gap for overlapping labels."""
    if n_samples <= 0 or n_splits <= 0:
        return []
    test_size = n_samples // (n_splits + 1)
    if test_size <= 0:
        return []
    splits = []
    for fold in range(n_splits):
        train_end = test_size * (fold + 1)
        test_start = train_end + max(0, int(gap))
        test_end = min(test_start + test_size, n_samples)
        if test_start < test_end:
            splits.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return splits


def train_model_walk_forward_v5(X, y_reg, y_cls, models_cfg, n_splits=TEST_SPLITS):
    """
    v5.2 Walk-Forward:
    - 回归模型预测21d收益 (用于排序), 自适应超参 + 时间衰减
    - 分类模型预测21d涨跌, 三模型ensemble投票 (RF+XGB+LGB)
    """
    split_indices = make_purged_time_splits(len(X), n_splits=n_splits)
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
        reg_preds = pd.Series(index=y_reg.index, dtype=np.float64)
        fold_metrics = []
        cls_artifact = None

        for fold, (tr_idx, te_idx) in enumerate(split_indices):
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
            y_pred = np.asarray(model.predict(X_te_s)).ravel()
            reg_preds.iloc[te_idx] = y_pred[:len(te_idx)]
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
            cls_models = []

            for cls_i, cls_name in enumerate(cls_model_names):
                try:
                    cls_m = _build_cls_model(cls_name, n_samples)
                    if cls_m is None:
                        continue
                    fold_preds = np.full(len(cls_y_arr), 0, dtype=np.int8)
                    cls_splits = make_purged_time_splits(len(cls_X_arr), n_splits=min(3, n_splits))
                    for tr_idx, te_idx in cls_splits:
                        X_tr_c = cls_X_arr[tr_idx]
                        X_te_c = cls_X_arr[te_idx]
                        y_tr_c = cls_y_arr[tr_idx]
                        cls_scaler_fold = StandardScaler()
                        X_tr_c_s = cls_scaler_fold.fit_transform(X_tr_c)
                        X_te_c_s = cls_scaler_fold.transform(X_te_c)
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
                        cls_scaler = StandardScaler()
                        X_cls_s = cls_scaler.fit_transform(cls_X_arr)
                        try:
                            cls_m_full.fit(X_cls_s, cls_y_arr, sample_weight=sample_weight[cls_mask])
                        except TypeError:
                            cls_m_full.fit(X_cls_s, cls_y_arr)
                        # Keep final models separately; validation uses OOS fold predictions.
                        cls_ensemble_preds[:, cls_i] = fold_preds
                        cls_models.append((cls_m_full, cls_scaler))
                except (ValueError, TypeError, RuntimeError) as exc:
                    print(f"  classification fold failed: {exc}")
                    continue

            # ensemble投票: 取三个分类器的均值方向
            cls_artifact = None
            used_models = np.any(cls_ensemble_preds != 0, axis=0)
            if used_models.sum() > 0:
                cls_ensemble = np.sign(np.mean(cls_ensemble_preds[:, used_models], axis=1))
                valid_ensemble = cls_ensemble != 0
                cls_acc = accuracy_score(cls_y_arr[valid_ensemble], cls_ensemble[valid_ensemble]) if valid_ensemble.sum() > 5 else 0
                cls_artifact = {
                    "models": cls_models,
                    "oos_accuracy": cls_acc,
                    "oof_pred": cls_ensemble,
                }
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
            "cls_artifact": cls_artifact,
            "cls_acc": cls_acc,
            "fold_r2_list": fold_metrics,
        }

    return results, scaler


# ═══════════════════════════════════════
# P0.1: 市场状态检测 + 自适应评分权重 (动置-均值回归切换)
# ═══════════════════════════════════════

def compute_adx(high, low, close, period=14):
    """计算ADX（平均趋向指数）"""
    n = len(high)
    if n < period:
        return 0
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]),
                               np.abs(low[1:] - close[:-1])))
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
    plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0
    return dx  # ADX值

def detect_market_regime(close, high, low):
    """检测市态: 'trend' / 'strong_trend' / 'sideways' / 'high_vol', 返回dict"""
    ret = np.diff(np.log(close))
    adx = compute_adx(high, low, close)
    vol_21d = float(np.std(ret[-21:]) * np.sqrt(252) * 100) if len(ret) >= 21 else 0
    vol_63d = float(np.std(ret[-63:]) * np.sqrt(252) * 100) if len(ret) >= 63 else vol_21d
    
    state = "neutral"
    if adx > MARKET_REGIME_CFG.get("adx_threshold_strong_trend", 35):
        state = "strong_trend"
    elif adx > MARKET_REGIME_CFG.get("adx_threshold_trend", 25):
        state = "trend"
    else:
        state = "sideways"
    
    vol_ratio = vol_21d / vol_63d if vol_63d > 0 else 1.0
    if vol_ratio > 1.5:
        high_vol = True
    else:
        high_vol = False
    
    return {"state": state, "high_vol": high_vol, "adx": adx, "vol_21d": vol_21d}

def adjust_scores_by_regime(results, macro_data=None):
    """根据市态调整评分集合的权重结构"""
    if not results:
        return results
    cfg = MARKET_REGIME_CFG
    if not cfg.get("enabled", True):
        return results
    
    # 检测市态（使用第一只股票的数据作参考）
    sample = results[0]
    regime_info = sample.get("_regime_info", {})
    state = regime_info.get("state", "trend")
    high_vol = regime_info.get("high_vol", False)
    
    is_sideways = state == "sideways"
    is_strong_trend = state == "strong_trend"
    
    for r in results:
        if is_sideways:
            # 震荡市：压缩动量权重，加大RSI/均值回归信号
            r["_mom_mult"] = cfg.get("mean_reversion_mom_weight_mult", 0.5)
            r["_trend_mult"] = cfg.get("mean_reversion_trend_weight_mult", 0.5)
        elif is_strong_trend:
            # 强趋势：加大动量/趋势权重
            r["_mom_mult"] = cfg.get("strong_trend_mom_weight_mult", 1.3)
            r["_trend_mult"] = cfg.get("strong_trend_trend_weight_mult", 1.5)
        else:
            r["_mom_mult"] = 1.0
            r["_trend_mult"] = 1.0
        
        # 高波动环境：降低动量权重
        if high_vol:
            r["_rsi_mult"] = cfg.get("high_vol_rsi_weight_mult", 1.2)
            r["_mom_mult"] = r.get("_mom_mult", 1.0) * cfg.get("high_vol_mom_weight_mult", 0.6)
    
    return results

# ═══════════════════════════════════════
# P0.2: 高频失败股惩罚机制
# ═══════════════════════════════════════

def load_stock_penalties():
    """从文件加载失败股惩罚记录"""
    path = os.path.expanduser(STOCK_PENALTY_CFG.get("penalty_file", "~/.cache/hermes-quant/stock_penalties.json"))
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"  penalty cache read failed: {exc}")
    return {"penalties": {}, "updated": datetime.now().isoformat()}

def save_stock_penalties(data):
    """保存失败股惩罚记录"""
    path = os.path.expanduser(STOCK_PENALTY_CFG.get("penalty_file", "~/.cache/hermes-quant/stock_penalties.json"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def get_penalty_for_ticker(ticker, stock_penalties):
    """获取某只股票的当前惩罚系数 (0=无惩罚, 最大值来自配置)"""
    cfg = STOCK_PENALTY_CFG
    if not cfg.get("enabled", True):
        return 0
    pdata = stock_penalties.get("penalties", {}).get(ticker, {})
    if not pdata:
        return 0
    failures = pdata.get("consecutive_high_conf_failures", 0)
    max_penalty = cfg.get("max_penalty", 0.15)
    penalty_per = cfg.get("penalty_per_failure", 0.04)
    threshold = cfg.get("consecutive_failures_threshold", 2)
    if failures < threshold:
        return 0
    penalty = min(failures * penalty_per, max_penalty)
    return penalty

def apply_score_spread(scores):
    """评分展宽: 拉大评分间距"""
    cfg = SCORE_SPREAD_CFG
    if not cfg.get("enabled", True):
        return scores
    method = cfg.get("method", "power")
    exponent = cfg.get("power_exponent", 1.3)
    min_s = cfg.get("min_score", 0.0)
    max_s = cfg.get("max_score", 1.0)
    if not scores:
        return scores
    arr = np.array(scores)
    # 归一化到0-1
    lo, hi = arr.min(), arr.max()
    if hi - lo < 0.01:
        return scores
    norm = (arr - lo) / (hi - lo)
    if method == "power":
        expanded = norm ** exponent
    else:
        expanded = norm
    # 缩放回原始区间
    result = expanded * (hi - lo) + lo
    return result.tolist()

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
    except (KeyError, TypeError, ValueError):
        return "震荡"

def score_stock_v5(ticker, macro_data=None, period="2y", force_refresh=False, market="US", data=None):
    """v5.2 个股评分: Rank一致性评分 + 双轨预测 + 自适应窗口 + 市态检测"""
    df = data.copy() if data is not None else get_cached_data(ticker, period=period, force_refresh=force_refresh)
    if df.empty or len(df) < MIN_TRADING_DAYS:
        return None
    cache_metadata = get_cache_metadata(ticker, period) if data is None else {
        "cache_age_hours": None,
        "stale": False,
    }

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    # 统一时区: 缓存文件可能来自 Ticker.history() (带时区) 或 yf.download (无时区)
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # ─── P1c: 自适应窗口 (v5.2: 根据波动率动态选择) ───
    if len(df) > min(ADAPTIVE_WINDOWS):
        closes_full = df["Close"].values.astype(float)
        ret_full = pd.Series(closes_full).pct_change()
        vol_21d = ret_full.tail(21).std() * np.sqrt(252)
        # 计算历史波动率百分位
        hist_vols = ret_full.rolling(63).std().dropna() * np.sqrt(252)
        if len(hist_vols) > 20:
            vol_pctl = (vol_21d < hist_vols).mean()
            if vol_pctl > 0.7:
                best_w = 504
            elif vol_pctl < 0.3:
                best_w = 1008 if 1008 <= len(df) else (756 if 756 <= len(df) else 504)
            else:
                best_w = 756 if 756 <= len(df) else (504 if 504 <= len(df) else len(df))
        else:
            best_w = max(ADAPTIVE_WINDOWS) if max(ADAPTIVE_WINDOWS) <= len(df) else len(df)
        df_used = df.iloc[-best_w:].copy()
    else:
        df_used = df
        best_w = len(df_used)

    # ─── 市态检测 (优化1) ───
    close_arr = df_used["Close"].values.astype(float)
    high_arr = df_used["High"].values.astype(float)
    low_arr = df_used["Low"].values.astype(float)
    regime_info = detect_market_regime(close_arr, high_arr, low_arr)

    features, target_5d, target_21d, target_cls = build_features_v5(
        df_used, macro_data, ticker, market=market
    )
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
    
    # ─── 捕获特征值快照（供 data_archiver 回测归档） ───
    _latest_features = {}
    if len(X) > 0:
        latest = X.iloc[-1]
        for col in X.columns:
            val = latest[col]
            if pd.notna(val) and not isinstance(val, (np.ndarray, pd.Series)):
                try:
                    _latest_features[col] = round(float(val), 6) if isinstance(val, (np.floating, float)) else val
                except (ValueError, TypeError):
                    pass
    
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
    
    # ─── P1b: EMA平滑动量兜底 + 市态调整 ───
    closes = df_used["Close"].values.astype(float)
    close_series = pd.Series(closes)
    ema_21 = close_series.ewm(span=21).mean()
    ema_63 = close_series.ewm(span=63).mean()
    mom_21d_val = (ema_21.iloc[-1] / ema_21.iloc[-22] - 1) if len(ema_21) >= 22 else 0
    mom_63d_val = (ema_63.iloc[-1] / ema_63.iloc[-64] - 1) if len(ema_63) >= 64 else 0
    avg_r2 = np.mean([results[n]["reg_r2"] for n in model_names])

    # 市态信息（供adjust_scores_by_regime使用）
    mom_mult = 1.0
    trend_mult = 1.0
    state = regime_info.get("state", "trend")
    high_vol = regime_info.get("high_vol", False)
    if state == "sideways":
        mom_mult = MARKET_REGIME_CFG.get("mean_reversion_mom_weight_mult", 0.5)
        trend_mult = MARKET_REGIME_CFG.get("mean_reversion_trend_weight_mult", 0.5)
    elif state == "strong_trend":
        mom_mult = MARKET_REGIME_CFG.get("strong_trend_mom_weight_mult", 1.3)
        trend_mult = MARKET_REGIME_CFG.get("strong_trend_trend_weight_mult", 1.5)
    if high_vol:
        mom_mult *= MARKET_REGIME_CFG.get("high_vol_mom_weight_mult", 0.6)

    # 市态调整后的动量兜底
    if avg_r2 < 0:
        adjusted_mom_21d = mom_21d_val * mom_mult
        adjusted_mom_63d = mom_63d_val * mom_mult
        mom_score = max(0, min(1, (adjusted_mom_21d * CFG["momentum_fallback"]["mom_21d_weight"] + adjusted_mom_63d * CFG["momentum_fallback"]["mom_63d_weight"] + CFG["momentum_fallback"]["mom_score_offset"])))
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
            cls_artifact = results[name].get("cls_artifact")
            if cls_artifact and results[name]["cls_acc"] > CFG["classification"]["min_accuracy"]:
                # 分类模型是ensemble数组, 用最新值
                x_latest = X.iloc[-1:].values
                model_preds = []
                for cls_model, cls_scaler in cls_artifact["models"]:
                    x_s = cls_scaler.transform(x_latest)
                    model_preds.append(int(cls_model.predict(x_s)[0]))
                cls_pred = int(np.sign(np.mean(model_preds))) if model_preds else 0
                if cls_pred == 0:
                    continue
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
            
            # ─── Fix4: 低信噪比保守模式 — 如果R²严重为负且置信度低, 记录状态不强制震荡
            # (注意: 批次级的保守模式会在run_ml_picking_v5()中用动量方向整体覆盖)
            if avg_r2 < -0.5 and confidence < 0.3 and direction != "震荡":
                # 标记为低信噪比，批次级处理时会整体用动量覆盖
                direction_source = "cls_low_snr"
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
    # ─── P2c: 防NaN兜底——yfinance偶发最后一天Close=NaN（未结算）───
    closes_raw = df_used["Close"].values.astype(float)
    # 找最后一个有效收盘价
    valid_mask = ~np.isnan(closes_raw) & (closes_raw > 0)
    if valid_mask.any():
        last_valid_idx = np.where(valid_mask)[0][-1]
        last_close = closes_raw[last_valid_idx]
        closes = closes_raw[:last_valid_idx + 1]  # 截断到有效位置
    else:
        last_close = 0.0
        closes = closes_raw

    trailing_return_5d = (last_close / closes[-min(6, len(closes))] - 1) * 100 if len(closes) >= 2 else 0
    trailing_return_21d = (last_close / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
    mom_1m = (last_close / closes[-21] - 1) * 100 if len(closes) >= 21 else 0
    mom_3m = (last_close / closes[-63] - 1) * 100 if len(closes) >= 63 else 0

    return {
        "ticker": ticker,
        "market": str(market).upper(),
        "benchmark": BENCHMARK_BY_MARKET.get(str(market).upper(), "spy"),
        "benchmark_status": "available" if (
            macro_data is not None and
            BENCHMARK_BY_MARKET.get(str(market).upper(), "spy") in macro_data
        ) else "missing",
        "cache_age_hours": cache_metadata.get("cache_age_hours"),
        "data_stale": cache_metadata.get("stale"),
        "price": round(float(last_close), 2) if last_close > 0 else 0.0,
        "score": round(final_score, 4),          # v5: 新评分(0~1)
        "rank_pctl": round(latest_rank, 4),       # 排名百分位
        "confidence": round(confidence, 4),       # 模型一致性
        "walk_forward_r2": round(np.mean([results[n]["reg_r2"] for n in model_names]), 4),
        "direction": direction,                    # 分类看涨/看跌
        "cls_details": cls_signals[:3],            # 分类详情
        "adaptive_window": best_w,
        # The future realized return is unavailable at signal time. Keep the
        # legacy field only for schema compatibility and expose trailing data
        # under an explicitly non-predictive name.
        "actual_5d": None,
        "trailing_return_5d": round(trailing_return_5d, 2),
        "trailing_return_21d": round(trailing_return_21d, 2),
        "signal_date": str(df_used.index[last_valid_idx].date()) if valid_mask.any() else None,
        "data_asof": str(df_used.index[last_valid_idx].date()) if valid_mask.any() else None,
        "target_horizon_days": FORECAST_HORIZON_SHORT,
        "direction_horizon_days": FORECAST_HORIZON_LONG,
        "mom_1m": round(mom_1m, 2),
        "mom_3m": round(mom_3m, 2),
        "models_used": model_names,
        "models_r2": {n: round(results[n]["reg_r2"], 4) for n in model_names},
        "models_consensus": round(confidence, 4),
        "sector": get_ticker_sector(ticker),
        "direction_source": direction_source,
        "_regime_info": regime_info,
        "_mom_mult": mom_mult,
        "_trend_mult": trend_mult,
        "_avg_r2": float(avg_r2) if isinstance(avg_r2, (int, float)) else 0,
        "_latest_features": _latest_features,
    }


# ═══════════════════════════════════════
# 批量选股 v5
# ═══════════════════════════════════════
def run_ml_picking_v5(tickers=None, market="US", macro_data=None,
                       force_refresh=False, top_n=TOP_N, verbose=True):
    """v5 批量选股"""
    if tickers is None:
        if market == "HK":
            tickers = HK_WATCHLIST
        else:
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
            sr = score_stock_v5(
                t, macro_data=macro_data, force_refresh=force_refresh, market=market
            )
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
        # 防限流: 每只股票之间短暂间隔, 给YF缓存喘息时间
        if i < total - 1:
            time.sleep(0.15)

    results.sort(key=lambda x: x["score"], reverse=True)

    # ─── 计算全体R²指标 ───
    all_r2 = [r.get("walk_forward_r2", 0) or r.get("_avg_r2", 0) for r in results]
    neg_r2_ratio = sum(1 for r2 in all_r2 if r2 < 0) / len(all_r2) if all_r2 else 0
    avg_all_r2 = np.mean(all_r2) if all_r2 else 0
    low_snr_mode = neg_r2_ratio > 0.8 or avg_all_r2 < -0.8

    # ─── R²全负保守模式: 方向脱钩分类器, 改用动量兜底 ───
    if low_snr_mode:
        if verbose:
            print(f"  🛡️  R²全负({neg_r2_ratio:.0%} avg_R²={avg_all_r2:.2f}) 保守模式")
        for r in results:
            direction_source = r.get("direction_source", "")
            # 只覆盖来自分类器的方向（cls_ensemble和cls_low_snr），
            # momentum_fallback和no_model等非分类器来源保留不变
            if direction_source in ("cls_ensemble", "cls_low_snr"):
                m1 = r.get("mom_1m", 0) or 0
                m3 = r.get("mom_3m", 0) or 0
                orig_dir = r.get("direction", "震荡")
                mom_accel = m1 - m3 * (abs(m3) / max(abs(m3), 5)) if abs(m3) > 1 else m1  # 动量加速度
                # 规则1: 动量陷阱检测
                if m3 > 15 and m1 < -8 and orig_dir != "看跌":
                    r["direction"] = "看跌"
                    r["direction_source"] = "low_snr_momentum_trap"
                    if verbose:
                        print(f"    🔻 {r['ticker']}: 动量陷阱({m1:+.0f}%1m/{m3:+.0f}%3m) {orig_dir}→看跌")
                # 规则2: 动量修复检测
                elif m3 < -15 and m1 > 8 and orig_dir != "看涨":
                    r["direction"] = "看涨"
                    r["direction_source"] = "low_snr_momentum_recovery"
                    if verbose:
                        print(f"    🔺 {r['ticker']}: 动量修复({m1:+.0f}%1m/{m3:+.0f}%3m) {orig_dir}→看涨")
                # 规则3: 强看涨(双月均>5%)却看跌 → 改为看涨
                elif orig_dir == "看跌" and m1 > 5 and m3 > 5:
                    r["direction"] = "看涨"
                    r["direction_source"] = "low_snr_correct_bull"
                # 规则4: 强看跌(1m<-8%且3m<-8%)却看涨 → 改为看跌
                elif orig_dir == "看涨" and m1 < -8 and m3 < -8:
                    r["direction"] = "看跌"
                    r["direction_source"] = "low_snr_correct_bear"
                else:
                    # 保留原方向，但标记为低信噪比来源
                    r["direction_source"] = "low_snr_cls"

    # ─── 后处理：评分展宽 (优化3) — R²全负时禁用 ───
    score_spread_enabled = SCORE_SPREAD_CFG.get("enabled", True)
    if score_spread_enabled and len(results) > 1:
        if neg_r2_ratio > 0.8 or avg_all_r2 < -1.0:
            # R²全负时禁用评分展宽——展宽在噪声环境下只会放大错误信号的区分度
            if verbose:
                print(f"  ⚠️  R²全负率{neg_r2_ratio:.0%} avg_R²={avg_all_r2:.2f}，评分展宽已禁用")
        else:
            scores = [r["score"] for r in results]
            expanded = apply_score_spread(scores)
            for i, r in enumerate(results):
                r["score"] = expanded[i]

    # ─── 后处理：市场状态批量调整 (优化1) + 改进(优化6) ───
    results = adjust_scores_by_regime(results, macro_data)

    # 改进: 震荡市+高波动 → 反转因子加强（额外调低动量权重）
    if results:
        sample = results[0]
        state = sample.get("_regime_info", {}).get("state", "trend")
        high_vol = sample.get("_regime_info", {}).get("high_vol", False)
        if state == "sideways" and high_vol:
            if verbose:
                print("  🔄 震荡+高波动市态: 强化反转因子(动量权重额外×0.7)")
            for r in results:
                r["_mom_mult"] = r.get("_mom_mult", 1.0) * 0.7

    # ─── 后处理：失败股惩罚 (优化2) ───
    if STOCK_PENALTY_CFG.get("enabled", True):
        stock_penalties = load_stock_penalties()
        for r in results:
            penalty = get_penalty_for_ticker(r["ticker"], stock_penalties)
            if penalty > 0:
                r["score"] = max(0, r["score"] - penalty)
                r["_penalty"] = penalty

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

    # ─── 数据质量报告 ───
    r2s = [s["walk_forward_r2"] for s in all_results]
    avg_r2 = np.mean(r2s)
    neg_r2_ratio = sum(1 for r in r2s if r < 0) / len(r2s) * 100
    # mom指标可用性
    mom1m_ok = sum(1 for s in all_results if not np.isnan(s.get("mom_1m", np.nan)))
    mom3m_ok = sum(1 for s in all_results if not np.isnan(s.get("mom_3m", np.nan)))
    price_ok = sum(1 for s in all_results if s.get("price", 0) > 0)
    print(f"\n  📊 数据质量:")
    print(f"     平均 R²: {avg_r2:.3f} | 负R²比例: {neg_r2_ratio:.0f}%")
    print(f"     有价格: {price_ok}/{len(all_results)} | mom_1m有效: {mom1m_ok}/{len(all_results)} | mom_3m有效: {mom3m_ok}/{len(all_results)}")
    if avg_r2 < -0.1:
        print(f"     ⚠️  R²均值 {avg_r2:.2f}（模型预测力偏弱），评分已启用动量兜底")
    if mom1m_ok < len(all_results) * 0.8:
        missing_price = [s["ticker"] for s in all_results if s.get("price", 0) <= 0]
        print(f"     ⚠️  部分股票最新收盘价异常（可能当日未结算），已回退到上一个交易日数据 → {', '.join(missing_price[:5])}")

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
        # 异常事件预警（如果有）
        sf = s.get("sentiment_factors", {})
        if sf.get("events"):
            evt_str = " + ".join(sf.get("event_labels", []))
            discount = sf.get("event_discount", 1.0)
            print(f"     ⚠️  异常事件: {evt_str} (折扣 {discount:.3f})")

    return all_results[:8]


def save_results_v5(results, filename="ml_v5_picks"):
    """保存结果（含情绪因子列，如有）"""
    rows = []
    for s in results:
        sf = s.get("sentiment_factors", {})
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
            "sentiment_score": sf.get("sentiment_score", ""),
            "news_count": sf.get("news_count", ""),
            "events": "+".join(sf.get("event_labels", [])),
            "event_discount": sf.get("event_discount", ""),
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
    parser.add_argument("--sentiment", action="store_true", help="融合新闻情绪因子和异常事件检测")
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

        # ─── 融合新闻情绪因子 + 异常事件检测 (在打印报告之前) ───
        if args.sentiment:
            print(f"\n{'='*50}")
            print(f"  📰 加载新闻情绪 + 异常事件检测...")
            print(f"{'='*50}")
            try:
                from finbert_sentiment import build_sentiment_factors, sentiment_boost

                # 收集所有 ticker
                all_tickers = list(set(r["ticker"] for r in all_results))
                sentiment_factors = build_sentiment_factors(
                    all_tickers,
                    signal_time=datetime.now(),
                    archive_root=os.path.join(CACHE_DIR, "backtest"),
                )

                llm_count = sum(1 for f in sentiment_factors.values() if f.get("method") == "llm")
                finbert_count = sum(1 for f in sentiment_factors.values() if f.get("method") == "finbert")
                keyword_count = len(all_tickers) - llm_count - finbert_count
                print(f"  DeepSeek-V4-Flash: {llm_count} 只 | 关键词: {keyword_count} 只")

                event_stocks = []
                for r in all_results:
                    t = r["ticker"]
                    sf = sentiment_factors.get(t, {})
                    if sf and sf.get("news_count", 0) > 0:
                        original = r["score"]
                        fused, adj, evt_adj = sentiment_boost(original, sf)
                        r["score"] = fused
                        r["ml_score_raw"] = original
                        r["sentiment_adj"] = adj
                        r["event_adj"] = evt_adj
                        r["sentiment_factors"] = sf
                        if sf.get("events"):
                            event_stocks.append((t, sf["event_labels"], sf.get("event_discount", 1.0)))
                    else:
                        r["sentiment_adj"] = 0
                        r["event_adj"] = 0
                        r["ml_score_raw"] = r["score"]
                        r["sentiment_factors"] = {"sentiment_score": 0, "events": []}

                # 重排序
                all_results.sort(key=lambda x: x["score"], reverse=True)

                # 异常事件预警
                if event_stocks:
                    print(f"\n  ⚠️  异常事件预警 ({len(event_stocks)} 只):")
                    for t, labels, discount in event_stocks:
                        label_str = " + ".join(labels)
                        print(f"    🔴 {t}: {label_str} (折扣系数 {discount:.3f})")

                # 打印融合详情 Top
                print(f"\n  {'='*100}")
                print(f"  情绪因子 + 异常事件 调整详情 (Top 10)")
                print(f"  {'='*100}")
                print(f"  {'代码':>6} {'ML原始':>8} {'情绪调':>8} {'事件调':>8} {'最终分':>8} {'新闻':>5} {'事件'}")
                for r in all_results[:10]:
                    sf = r.get("sentiment_factors", {})
                    adj = r.get("sentiment_adj", 0)
                    evt_adj = r.get("event_adj", 0)
                    orig = r.get("ml_score_raw", r["score"] - adj - evt_adj)
                    events_str = " + ".join(sf.get("event_labels", [])) or "—"
                    print(f"  {r['ticker']:>6} {orig:>8.4f} {adj:>+8.4f} {evt_adj:>+8.4f} {r['score']:>8.4f} {sf.get('news_count',0):>5} {events_str}")

                print(f"\n  ✅ 情绪融合完成! ({len(all_tickers)} 只)")
            except ImportError as e:
                print(f"  ❌ 情绪模块加载失败: {e}")
                print(f"     请确保 finbert_sentiment.py 与当前脚本在同一目录")
            except Exception as e:
                print(f"  ❌ 情绪融合失败: {e}")

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
