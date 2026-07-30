#!/usr/bin/env python3
"""
data_archiver.py — 量化数据归档模块 (v1.0)

每次预测/复盘时自动保存完整数据快照，构建可回测的历史数据集。

归档结构:
  ~/.cache/hermes-quant/backtest/
    <market>/  (US / HK)
      predictions/  ← 每次预测的完整明细CSV
      raw_data/     ← 每日原始行情快照 (个股+宏观因子) 去重
      features/     ← 每只股票的特征值快照
      macros/       ← 宏观因子时间序列 (累积)
      review/       ← 复盘结果累积
    ticker_map.csv  ← 股票池快照
    data_stats.json ← 归档统计

无外部依赖 (仅 Python 标准库 + pandas/numpy，这两个已在项目 venv 中)。
设计为幂等：多次运行不会重复写入同一天的数据。
"""

import os, json, sys, time, pickle, hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
BT_DIR = os.path.join(CACHE_DIR, "backtest")

# ─── 路径工厂 ───

def _bt_mkdir(*subdirs):
    d = os.path.join(BT_DIR, *subdirs)
    os.makedirs(d, exist_ok=True)
    return d

def _pred_path(market, date_str):
    return os.path.join(BT_DIR, market, "predictions", f"pred_{market}_{date_str}.csv")

def _raw_path(market, ticker, date_str):
    return os.path.join(BT_DIR, market, "raw_data", ticker, f"{date_str}.csv")

def _macro_path(market, date_str):
    return os.path.join(BT_DIR, market, "macros", f"macro_{date_str}.csv")

def _feature_path(market, ticker, date_str):
    return os.path.join(BT_DIR, market, "features", ticker, f"{date_str}.csv")

def _review_path(market):
    return os.path.join(BT_DIR, market, "review", "review_history.csv")

def _ticker_map_path():
    return os.path.join(BT_DIR, "ticker_map.csv")

def _stats_path():
    return os.path.join(BT_DIR, "data_stats.json")

# ─── 归档函数 ───

def archive_raw_data(market, ticker, df):
    """
    归档单只股票的原始行情快照。
    df: 带 'Close','Open','High','Low','Volume' 列的 DataFrame，index 为 DatetimeIndex
    每个交易日保存一次，去重（同一个<market,ticker,date>不会重复写）。
    """
    if df is None or (hasattr(df, 'empty') and df.empty):
        return 0
    # 取最新一天
    last_date = df.index[-1]
    date_str = last_date.strftime("%Y-%m-%d")
    fpath = _raw_path(market, _safe_ticker(ticker), date_str)
    if os.path.exists(fpath):
        return 0  # 已归档，跳过
    _bt_mkdir(market, "raw_data", _safe_ticker(ticker))
    # 写出最近60个交易日（足够训练窗口）
    recent = df.tail(60)
    recent.to_csv(fpath, index=True)
    return 1


def archive_batch_raw_data(market, ticker_dfs):
    """
    批量归档多只股票的原始行情。
    ticker_dfs: dict {ticker: pd.DataFrame} �� list of (ticker, df) 
    """
    count = 0
    if isinstance(ticker_dfs, dict):
        items = ticker_dfs.items()
    else:
        items = ticker_dfs
    for item in items:
        if isinstance(item, (list, tuple)):
            ticker, df = item
        else:
            continue
        count += archive_raw_data(market, ticker, df)
    return count


def archive_macro_snapshot(market, macro_data):
    """
    归档宏观因子当日快照。
    macro_data: dict {name: pd.Series}，由 get_macro_data() 返回。
    只保存各因子最新值。
    """
    if not macro_data:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    fpath = _macro_path(market, today)
    if os.path.exists(fpath):
        return 0
    _bt_mkdir(market, "macros")
    rows = {"date": today}
    for name, series in macro_data.items():
        if series is not None and not series.empty:
            # 取最新2个值 (today 和 yesterday)
            vals = series.tail(2)
            for dt, val in vals.items():
                if isinstance(dt, (pd.Timestamp, datetime)):
                    pass  # 直接用
                else:
                    dt = pd.Timestamp(dt)
                col = f"{name}_{dt.strftime('%Y-%m-%d')}"
                rows[col] = float(val) if pd.notna(val) else None
    pd.DataFrame([rows]).to_csv(fpath, index=False)
    return 1


def archive_features(market, ticker, features_data):
    """
    归档特征值快照。
    features_data: dict 或 pd.DataFrame，评分时的特征值。
    保存最新一行的特征向量。
    """
    if not features_data:
        return 0
    today = datetime.now().strftime("%Y-%m-%d")
    fpath = _feature_path(market, _safe_ticker(ticker), today)
    if os.path.exists(fpath):
        return 0
    _bt_mkdir(market, "features", _safe_ticker(ticker))
    if isinstance(features_data, pd.DataFrame):
        if features_data.empty:
            return 0
        # 只保存最新一行
        latest = features_data.iloc[-1:].copy()
        latest.insert(0, "ticker", ticker)
        latest.insert(0, "date", today)
        latest.to_csv(fpath, index=True)
    elif isinstance(features_data, dict):
        row = {"date": today, "ticker": ticker}
        row.update(features_data)
        pd.DataFrame([row]).to_csv(fpath, index=False)
    return 1


def archive_predictions(market, predictions, date_str=None):
    """
    归档预测结果明细。
    predictions: score_stock_v5() 返回的结果列表（每个 dict 含 ticker, score, direction ...）
    """
    if not predictions:
        return 0
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    fpath = _pred_path(market, date_str)
    if os.path.exists(fpath):
        return 0
    _bt_mkdir(market, "predictions")
    rows = []
    for p in predictions:
        rows.append({
            "date": date_str,
            "signal_date": p.get("signal_date", date_str),
            "data_asof": p.get("data_asof", ""),
            "ticker": p["ticker"],
            "market": p.get("market", market),
            "benchmark": p.get("benchmark", ""),
            "benchmark_status": p.get("benchmark_status", ""),
            "cache_age_hours": p.get("cache_age_hours", ""),
            "data_stale": p.get("data_stale", ""),
            "sector": p.get("sector", ""),
            "score": p.get("score", 0),
            "rank_pctl": p.get("rank_pctl", 0),
            "confidence": p.get("confidence", 0),
            "walk_forward_r2": p.get("walk_forward_r2", 0),
            "direction": p.get("direction", ""),
            "direction_source": p.get("direction_source", ""),
            "target_horizon_days": p.get("target_horizon_days", ""),
            "direction_horizon_days": p.get("direction_horizon_days", ""),
            "price": p.get("price", 0),
            "mom_1m": p.get("mom_1m", 0),
            "mom_3m": p.get("mom_3m", 0),
            "trailing_return_5d": p.get("trailing_return_5d", ""),
            "trailing_return_21d": p.get("trailing_return_21d", ""),
            "regime_state": p.get("_regime_info", {}).get("state", ""),
            "regime_adx": p.get("_regime_info", {}).get("adx", 0),
            "models_used": "+".join(p.get("models_used", ["rf"])),
            "sentiment_adj": p.get("sentiment_adj", 0),
            "event_adj": p.get("event_adj", 0),
        })
    pd.DataFrame(rows).to_csv(fpath, index=False, encoding="utf-8-sig")
    return len(rows)


def archive_review(market, opt_record):
    """
    归档复盘结果。
    opt_record: cron_market_job.py 的 run_post_market() 返回的复盘 dict。
    """
    if not opt_record:
        return 0
    fpath = _review_path(market)
    _bt_mkdir(market, "review")
    row = {
        "date": opt_record.get("date", ""),
        "accuracy": opt_record.get("accuracy", 0),
        "total": opt_record.get("total", 0),
        "correct": opt_record.get("correct", 0),
        "spearman_corr": opt_record.get("spearman_corr", 0),
        "bias": opt_record.get("bias", ""),
        "actions_taken": "; ".join(opt_record.get("actions_taken", [])),
        "suggestion_types": "; ".join(s.get("type", "") for s in opt_record.get("suggestions", [])),
    }
    if os.path.exists(fpath):
        existing = pd.read_csv(fpath)
        # 去重
        dup = (existing["date"] == row["date"])
        if dup.any():
            return 0
        pd.concat([existing, pd.DataFrame([row])], ignore_index=True).to_csv(fpath, index=False)
    else:
        pd.DataFrame([row]).to_csv(fpath, index=False)
    return 1


def archive_ticker_map(tickers_us=None, tickers_hk=None):
    """
    归档股票池快照。
    """


def archive_daily_detail(market, compare, date_str=None):
    """
    归档逐只股票对比明细。
    compare: cron_market_job.py 中 run_post_market() 构建的 compare 列表，
            每个元素含 ticker, score, pred_dir, actual_chg, actual_dir, correct, price, sector, name 等。
    每个交易日存一个 CSV，用于回测时按评分分组算PnL。
    """
    if not compare:
        return 0
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    fpath = os.path.join(BT_DIR, market, "daily_detail", f"detail_{date_str}.csv")
    if os.path.exists(fpath):
        # 同一天多次复盘，追加
        existing = pd.read_csv(fpath)
        existing_tickers = set(existing["ticker"])
        new_items = [c for c in compare if c["ticker"] not in existing_tickers]
        if not new_items:
            return 0
        compare = new_items
    _bt_mkdir(market, "daily_detail")
    rows = []
    for c in compare:
        rows.append({
            "date": date_str,
            "ticker": c["ticker"],
            "sector": c.get("sector", ""),
            "score": c.get("score", 0),
            "pred_dir": c.get("pred_dir", ""),
            "actual_chg": c.get("actual_chg", 0),
            "actual_dir": c.get("actual_dir", ""),
            "correct": c.get("correct", False),
            "price": c.get("price", 0),
        })
    df = pd.DataFrame(rows)
    if os.path.exists(fpath):
        existing = pd.read_csv(fpath)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(fpath, index=False, encoding="utf-8-sig")
    return len(rows)


def archive_ticker_map(tickers_us=None, tickers_hk=None):
    """
    归档股票池快照。
    """
    fpath = _ticker_map_path()
    _bt_mkdir()
    rows = []
    if tickers_us:
        for t in tickers_us:
            rows.append({"ticker": t, "market": "US", "date_added": datetime.now().strftime("%Y-%m-%d")})
    if tickers_hk:
        for t in tickers_hk:
            rows.append({"ticker": t, "market": "HK", "date_added": datetime.now().strftime("%Y-%m-%d")})
    if rows:
        pd.DataFrame(rows).to_csv(fpath, index=False)
    return len(rows)


def update_stats(market, pred_count=0, raw_count=0, macro_count=0, feature_count=0, detail_count=0):
    """更新归档统计"""
    stats = {}
    if os.path.exists(_stats_path()):
        with open(_stats_path()) as f:
            stats = json.load(f)
    today = datetime.now().strftime("%Y-%m-%d")
    ms = stats.setdefault(market, {"last_archive": "", "total_days": 0, "total_predictions": 0, "total_raw_snapshots": 0, "total_macro_snapshots": 0, "total_features": 0, "daily_records": {}})
    ms["last_archive"] = today
    ms["total_predictions"] += pred_count
    ms["total_raw_snapshots"] += raw_count
    ms["total_macro_snapshots"] += macro_count
    ms["total_features"] += feature_count
    ms["total_details"] = ms.get("total_details", 0) + detail_count
    dr = ms["daily_records"].get(today, {"pred_count": 0, "raw_count": 0, "feature_count": 0, "detail_count": 0})
    dr["pred_count"] += pred_count
    dr["raw_count"] += raw_count
    dr["feature_count"] += feature_count
    dr["detail_count"] = dr.get("detail_count", 0) + detail_count
    ms["daily_records"][today] = dr
    ms["total_days"] = len(ms["daily_records"])
    with open(_stats_path(), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def print_stats():
    """打印归档统计"""
    if not os.path.exists(_stats_path()):
        print("  归档尚未启动或为空")
        return
    with open(_stats_path()) as f:
        stats = json.load(f)
    for market, ms in stats.items():
        print(f"\n  📊 {market} 归档统计:")
        print(f"     最后归档: {ms.get('last_archive', 'N/A')}")
        print(f"     归档天数: {ms.get('total_days', 0)}")
        print(f"     预测记录: {ms.get('total_predictions', 0)} 条")
        print(f"     行情快照: {ms.get('total_raw_snapshots', 0)} 条")
        print(f"     宏观快照: {ms.get('total_macro_snapshots', 0)} 条")
        print(f"     特征快照: {ms.get('total_features', 0)} 条")
        print(f"     对比明细: {ms.get('total_details', 0)} 条")


def _safe_ticker(ticker):
    """将 ticker 中的点/^号替换为下划线，确保文件系统兼容"""
    return ticker.replace(".", "_").replace("^", "_")


# ═══════════════════════════════════════
# 一次归档：全量归档当前 state 中的所有预测
# ═══════════════════════════════════════
def full_archive_from_state(market, state, macro_data=None):
    """
    从 state json 中取出 last_predictions / last_pre_full 做归档。
    适合 cron_market_job 调用，一次归档所有数据。
    """
    predictions = state.get("last_predictions", [])
    if not predictions:
        print(f"  ⚠️ {market} state 中无预测记录，跳过归档")
        return

    pred_count = archive_predictions(market, predictions)
    raw_count = 0
    feature_count = 0
    macro_count = 0

    # 归档行情: 尝试用 last_pre_full 中的 data
    full_data = state.get("last_pre_full", [])
    for p in (full_data or predictions):
        ticker = p["ticker"]
        # data_archiver 自身无法获取 df（数据在 ml_optimized_picker_v5 内部）
        # 这里仅归档已有的特征值
        for k in ["walk_forward_r2", "direction", "actual_5d", "mom_1m", "mom_3m", "price",
                   "rank_pctl", "confidence", "models_consensus", "adaptive_window"]:
            pass  # 这些已在 predictions CSV 中

    # 归档宏观因子 (如果提供了 macro_data)
    if macro_data:
        macro_count = archive_macro_snapshot(market, macro_data)
        print(f"  📡 宏观因子快照: {macro_count}")

    print(f"  📦 归档: {market} | 预测 {pred_count} 条 | 行情 {raw_count} | 特征 {feature_count} | 宏观 {macro_count}")
    update_stats(market, pred_count=pred_count, raw_count=raw_count, macro_count=macro_count, feature_count=feature_count)


def full_archive_from_run(market, predictions, macro_data=None):
    """
    直接从 run_ml_picking_v5 返回的结果归档（无需 state 文件）。
    在 cron_market_job 的 pre/post 中调用，数据最新。
    """
    if not predictions:
        print(f"  ⚠️ {market} 无预测结果，跳过归档")
        return

    pred_count = archive_predictions(market, predictions)
    feature_count = 0
    macro_count = 0

    # 归档特征值（每个预测结果含 _latest_features 字段）
    for p in predictions:
        lf = p.get("_latest_features")
        if lf:
            archive_features(market, p["ticker"], lf)
            feature_count += 1

    if macro_data:
        macro_count = archive_macro_snapshot(market, macro_data)

    # 归档行情（从 state 已有缓存中批量归档）
    # get_cached_data 保存的 .pkl 已经由系统维护，data_archiver 不再重复归档
    # 有需要时可通过 state 文件的 last_pre_full 中的 price/mom 等做衍生

    print(f"  📦 归档: {market} | 预测 {pred_count} 条 | 特征 {feature_count} | 宏观 {macro_count}")
    update_stats(market, pred_count=pred_count, raw_count=0, macro_count=macro_count, feature_count=feature_count)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="量化数据归档工具")
    parser.add_argument("--stats", action="store_true", help="打印归档统计")
    parser.add_argument("--full-archive", action="store_true", help="从 state 文件执行一次全量归档")
    parser.add_argument("--market", choices=["US", "HK"], help="市场 (与 --full-archive 搭配使用)")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        sys.exit(0)

    if args.full_archive and args.market:
        state_path = os.path.join(CACHE_DIR, "market_jobs", f"{args.market.lower()}_state.json")
        if not os.path.exists(state_path):
            print(f"❌ 未找到 {args.market} state 文件: {state_path}")
            sys.exit(1)
        with open(state_path) as f:
            state = json.load(f)
        full_archive_from_state(args.market, state)
        print_stats()
        sys.exit(0)

    print("用法: python data_archiver.py --stats 或 --full-archive --market US")
