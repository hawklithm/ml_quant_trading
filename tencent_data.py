"""
腾讯财经数据源模块 — 替代 Yahoo Finance
=============================================
港股日K 500条前复权 via web.ifzq.gtimg.cn
新浪实时报价 via hq.sinajs.cn
"""

import pandas as pd
import numpy as np
import requests
import os, time, pickle, json
from datetime import datetime, timedelta

# ─── 代码映射 ───
# 港股: 0700.HK → hk00700
# 美股: AAPL → gb_aapl (新浪) 
# 指数: HSI → hkHSI

TENCENT_TICKER_MAP = {}
SINA_TICKER_MAP_HK = {}
SINA_TICKER_MAP_US = {}

_HK_TICKERS = [
    "0700.HK","9988.HK","9999.HK","1810.HK","3690.HK",
    "0941.HK","0883.HK","0388.HK","0005.HK","1299.HK",
    "2269.HK","2382.HK","9618.HK","1024.HK",
    "0939.HK","3988.HK","0857.HK","0027.HK","1928.HK","1177.HK",
]
for t in _HK_TICKERS:
    code = t.replace(".HK", "").zfill(5)  # 补齐5位: 0700→00700
    TENCENT_TICKER_MAP[t] = f"hk{code}"
    SINA_TICKER_MAP_HK[t] = f"hk{code}"

# 美股
_US_TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","AVGO","ORCL",
    "AMD","QCOM","TSM",
    "JPM","V","MA","GS","BLK",
    "WMT","COST","HD","PG","KO","PEP","MCD",
    "UNH","JNJ","LLY","ABBV",
    "XOM","CAT","GE",
]
for t in _US_TICKERS:
    SINA_TICKER_MAP_US[t] = f"gb_{t.lower()}"

# 宏观代码映射
MACRO_TICKERS = {
    "hsi": "hkHSI",       # 恒生指数 via 腾讯
    "dxy": "gb_dxy",      # 美元指数 via 新浪
}

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
os.makedirs(CACHE_DIR, exist_ok=True)

_HEADERS = {'User-Agent': 'Mozilla/5.0'}
_SINA_HEADERS = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finance.sina.com.cn'}


def _to_dataframe(days, ticker=""):
    """将腾讯财经日K数组转为标准DataFrame"""
    rows = []
    for item in days:
        try:
            date = item[0]
            o = float(item[1])
            c = float(item[2])
            h_val = float(item[3])
            l_val = float(item[4])
            v = float(item[5]) if isinstance(item[5], (int, float, str)) else float(item[5][0])
            rows.append([date, o, h_val, l_val, c, v])
        except (IndexError, ValueError, TypeError):
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df


def fetch_hk_kline(ticker, count=500):
    """从腾讯财经获取港股日K线 (前复权)"""
    tc = TENCENT_TICKER_MAP.get(ticker)
    if not tc:
        # 尝试直接构建
        tc = ticker.replace(".HK", "")
        tc = f"hk{tc}" if not tc.startswith("hk") else tc
    
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tc},day,,,{count},qfq"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        code_key = tc
        if data.get('data') and data['data'].get(code_key) and data['data'][code_key].get('day'):
            days = data['data'][code_key]['day']
            return _to_dataframe(days, ticker)
    except Exception:
        pass
    return pd.DataFrame()


def fetch_hsi_kline(count=500):
    """从腾讯财经获取恒生指数日K"""
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=hkHSI,day,,,{count},qfq"
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if data.get('data'):
            for k, v in data['data'].items():
                if isinstance(v, dict) and 'day' in v:
                    days = v['day']
                    rows = []
                    for item in days:
                        try:
                            date = item[0]
                            c = float(item[2])
                            rows.append([date, c])
                        except:
                            continue
                    if rows:
                        df = pd.DataFrame(rows, columns=['Date', 'Close'])
                        df['Date'] = pd.to_datetime(df['Date'])
                        df = df.set_index('Date').sort_index()
                        return df
    except Exception:
        pass
    return pd.DataFrame()


def fetch_sina_realtime(tickers):
    """从新浪获取批量实时报价 (港股+美股)
    返回: {ticker: {'price': float, 'change_pct': float, 'open': float, 'high': float, 'low': float, 'volume': float}}
    """
    sina_codes = []
    code_to_ticker = {}
    for t in tickers:
        if t in SINA_TICKER_MAP_HK:
            sc = SINA_TICKER_MAP_HK[t]
            sina_codes.append(sc)
            code_to_ticker[sc] = t
        elif t in SINA_TICKER_MAP_US:
            sc = SINA_TICKER_MAP_US[t]
            sina_codes.append(sc)
            code_to_ticker[sc] = t
        # 宏观因子
        elif t == "^HSI":
            sina_codes.append("hkHSI")
            code_to_ticker["hkHSI"] = t
        elif t == "^DXY" or t == "DX-Y.NYB":
            sina_codes.append("gb_dxy")
            code_to_ticker["gb_dxy"] = t
        else:
            # 其他美股/ETF尝试
            sc = f"gb_{t.lower()}"
            sina_codes.append(sc)
            code_to_ticker[sc] = t
    
    if not sina_codes:
        return {}
    
    codes_str = ",".join(sina_codes)
    url = f"https://hq.sinajs.cn/list={codes_str}"
    try:
        r = requests.get(url, headers=_SINA_HEADERS, timeout=10)
        if r.status_code != 200:
            return {}
        
        result = {}
        for line in r.text.strip().split('\n'):
            if 'hq_str' not in line:
                continue
            # 提取 code 和数据
            try:
                # var hq_str_gb_aapl="...";
                code = line.split('hq_str_')[1].split('=')[0].strip()
                raw = line.split('"')[1]
                parts = raw.split(',')
                
                ticker = code_to_ticker.get(code)
                if not ticker:
                    continue
                
                # 港股格式: name_en, name_cn, open, last_close, price, high, low, change, change_pct, ...
                if code.startswith('hk'):
                    # market_state: 0=盘中 1=休市 2=收市
                    market_state = parts[10] if len(parts) > 10 else ''
                    price = float(parts[4]) if len(parts) > 4 and parts[4] else 0
                    change_pct = parts[2]  # 绝对值
                    open_p = float(parts[5]) if len(parts) > 5 and parts[5] else price
                    high = float(parts[6]) if len(parts) > 6 and parts[6] else price
                    low = float(parts[7]) if len(parts) > 7 and parts[7] else price
                    volume = float(parts[8]) if len(parts) > 8 and parts[8] else 0
                # 美股格式: name,price,change_pct,time,change_amount,open,high,low,...
                elif code.startswith('gb'):
                    price = float(parts[1]) if parts[1] else 0
                    change_pct = float(parts[2]) if parts[2] else 0.0  # 已经是百分比
                    open_p = float(parts[6]) if len(parts) > 6 and parts[6] else price
                    high = float(parts[7]) if len(parts) > 7 and parts[7] else price
                    low = float(parts[8]) if len(parts) > 8 and parts[8] else price
                    volume = float(parts[11]) if len(parts) > 11 and parts[11] else 0
                else:
                    continue
                
                result[ticker] = {
                    'price': price,
                    'change_pct': change_pct,
                    'open': open_p,
                    'high': high,
                    'low': low,
                    'volume': volume,
                }
            except (IndexError, ValueError):
                continue
        return result
    except Exception:
        return {}


def _cache_path_local(ticker):
    safe_name = ticker.replace(".", "_").replace("^", "_")
    return os.path.join(CACHE_DIR, f"data_{safe_name}_2y.pkl")

def _macro_cache_path_local():
    return os.path.join(CACHE_DIR, "macro_data.pkl")

def get_hk_kline_data(ticker, force_refresh=False, cache_ttl_hours=12):
    """港股K线数据：腾讯财经 + 缓存系统"""
    path = _cache_path_local(ticker)
    
    # 缓存有效则直接返回
    if not force_refresh and os.path.exists(path):
        mtime = os.path.getmtime(path)
        age = time.time() - mtime
        if age < cache_ttl_hours * 3600:
            try:
                return pd.read_pickle(path)
            except:
                pass
    
    # 从腾讯财经获取
    df = fetch_hk_kline(ticker, count=500)
    if not df.empty:
        with open(path, "wb") as f:
            pickle.dump(df, f)
        return df
    
    # 失败则回退缓存
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except:
            pass
    return pd.DataFrame()


def get_hk_macro_data(force_refresh=False, cache_ttl_hours=12):
    """获取宏观因子: HSI via 腾讯, 其他新浪实时"""
    path = _macro_cache_path_local()
    
    # 缓存
    if not force_refresh and os.path.exists(path):
        mtime = os.path.getmtime(path)
        if time.time() - mtime < cache_ttl_hours * 3600:
            try:
                return pd.read_pickle(path)
            except:
                pass
    
    result = {}
    
    # HSI 恒生指数
    hsi = fetch_hsi_kline(500)
    if not hsi.empty:
        result['hsi'] = hsi['Close']
    
    if result:
        with open(path, "wb") as f:
            pickle.dump(result, f)
        return result
    
    # 回退
    if os.path.exists(path):
        try:
            return pd.read_pickle(path)
        except:
            pass
    return {}


def batch_fetch_hk_kline(tickers):
    """批量获取港股K线，逐个请求"""
    results = {}
    for t in tickers:
        df = fetch_hk_kline(t, count=500)
        if not df.empty:
            results[t] = df
            time.sleep(0.15)  # 礼貌间隔
    return results


if __name__ == "__main__":
    # 测试港股
    import sys
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "0700.HK"
    df = fetch_hk_kline(test_ticker)
    print(f"{test_ticker}: {len(df)} rows")
    if not df.empty:
        print(f"  范围: {df.index[0]} ~ {df.index[-1]}")
        print(f"  列: {list(df.columns)}")
        print(f"  最新 Close: {df['Close'].iloc[-1]:.2f}")
    
    # 测试HSI
    if len(sys.argv) <= 1:
        hsi = fetch_hsi_kline()
        print(f"\nHSI: {len(hsi)} rows")
        if not hsi.empty:
            print(f"  最新: {hsi.index[-1]} Close={hsi['Close'].iloc[-1]:.2f}")

    # 测试新浪实时
    print(f"\n新浪实时 {test_ticker}:")
    rt = fetch_sina_realtime([test_ticker, "^HSI", "AAPL"])
    for t, v in rt.items():
        print(f"  {t}: ${v['price']} ({v['change_pct']})")
