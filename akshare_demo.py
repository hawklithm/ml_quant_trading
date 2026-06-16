#!/usr/bin/env python3
"""
A股数据源 — AkShare 接入演示

AkShare 是国内最全的免费金融数据源, 无需注册。
支持: A股/港股/期货/基金/指数/宏观经济

用法:
  python akshare_demo.py

注意事项:
  - 首次调用需要下载数据, 速度取决于网络
  - 部分接口在交易时间外可能返回空数据
  - 如需更高频数据, 可切换至 Tushare (需注册)
"""

import pandas as pd
import datetime
import warnings
warnings.filterwarnings("ignore")

print("=" * 60)
print("AkShare 数据源演示")
print("=" * 60)

# ──────────── 1. A股日线行情 ────────────
print("\n1) 沪深300 日线行情 (最近20个交易日)")
import akshare as ak
try:
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df.index = pd.to_datetime(df.index)
    print(f"  数据: {df.shape[0]} 行 | {df.index[0].date()} ~ {df.index[-1].date()}")
    print(df.tail(3).to_string())
except Exception as e:
    print(f"  错误: {e}")

# ──────────── 2. 个股日线 ────────────
print("\n2) 贵州茅台 日线行情 (最近1年)")
try:
    df = ak.stock_zh_a_hist(symbol="600519", period="daily",
                             start_date=(datetime.date.today() - datetime.timedelta(days=365)).strftime("%Y%m%d"),
                             end_date=datetime.date.today().strftime("%Y%m%d"),
                             adjust="qfq")  # 前复权
    print(f"  数据: {df.shape[0]} 行 | 最新价: {df['收盘'].iloc[-1]} | 最旧: {df['日期'].iloc[0]}")
    print(df.tail(3).to_string())
except Exception as e:
    print(f"  错误: {e}")

# ──────────── 3. 实时行情 ────────────
print("\n3) 实时行情 (A股当前交易日)")
try:
    df = ak.stock_zh_a_spot_em()
    print(f"  全市场股票数: {df.shape[0]}")
    # 按成交额排序取前10
    top10 = df.sort_values("成交额", ascending=False).head(10)
    print("  成交额Top10:")
    print(top10[["代码", "名称", "最新价", "涨跌幅", "成交额"]].to_string(index=False))
except Exception as e:
    print(f"  错误: {e}")

# ──────────── 4. ETF/基金 ────────────
print("\n4) ETF 实时行情")
try:
    df = ak.fund_etf_spot_em()
    print(f"  ETF数量: {df.shape[0]}")
    top_etf = df.sort_values("成交额", ascending=False).head(5)
    print("  成交额Top5 ETF:")
    print(top_etf[["代码", "名称", "最新价", "涨跌幅", "成交额"]].to_string(index=False))
except Exception as e:
    print(f"  错误: {e}")

# ──────────── 5. 北向资金 ────────────
print("\n5) 北向资金 (沪深港通)")
try:
    df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    df.index = pd.to_datetime(df.index)
    print(f"  数据: {df.shape[0]} 行 | 最近: {df.index[-1].date()}")
    print(df.tail(3).to_string())
except Exception as e:
    print(f"  错误: {e}")

# ──────────── 6. 对比 yfinance ────────────
print("\n6) 数据对比: 沪深300 yfinance vs AkShare")
try:
    import yfinance as yf
    yf_df = yf.download("000300.SS", period="1mo", auto_adjust=True, progress=False)
    if not yf_df.empty:
        yf_df.columns = [c[0] for c in yf_df.columns]
        print(f"  yfinance: {yf_df.shape[0]} 日 | 最新价 {yf_df['Close'].iloc[-1]:.0f}")
    else:
        print("  yfinance: 无法获取 000300.SS (A股限制)")
except Exception:
    print("  yfinance: 获取 A股受限")

try:
    ak_df = ak.stock_zh_index_daily(symbol="sh000300")
    ak_df.index = pd.to_datetime(ak_df.index)
    print(f"  AkShare:  {ak_df.shape[0]} 日 | 最新价 {ak_df['close'].iloc[-1]:.0f}")
except Exception as e:
    print(f"  AkShare: {e}")

print()
print("=" * 60)
print("AkShare 可用接口速查")
print("=" * 60)
print("""
  stock_zh_a_hist()     — 个股日线 (支持前复权)
  stock_zh_a_spot_em()   — 个股实时行情
  stock_zh_index_daily() — 指数日线
  fund_etf_spot_em()     — ETF实时行情
  stock_hsgt_north_net_flow_in_em() — 北向资金
  stock_individual_fund_flow() — 个股资金流向
  stock_zh_a_tick_tx()   — 分笔交易 (Tick级)
  stock_zh_a_hist_min_em() — 分钟线
""")
