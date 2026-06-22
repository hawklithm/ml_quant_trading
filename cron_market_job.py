#!/usr/bin/env python3
"""
cron_market_job.py — ML量化选股定时任务脚本

功能:
  - pre:  开市前选股预测 (运行v5评分系统)
  - post: 收市后复盘对比 + 差异分析 + 自动优化建议 + 代码修改

依赖:
  ml_optimized_picker_v5.py (v5评分系统)

用法:
  python cron_market_job.py --market HK --mode pre
  python cron_market_job.py --market US --mode post
"""

import sys, os, json, time, re, numpy as np
from datetime import datetime, timedelta

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ─── 路径配置 ───
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
STATE_DIR = os.path.join(CACHE_DIR, "market_jobs")
os.makedirs(STATE_DIR, exist_ok=True)

# ─── 冬令时EDST UTC-5, 差13小时, 但我们的crontab也是季节性的

MARKET_CONFIG = {
    "HK": {
        "state_file": os.path.join(STATE_DIR, "hk_state.json"),
        "label": "港股",
    },
    "US": {
        "state_file": os.path.join(STATE_DIR, "us_state.json"),
        "label": "美股",
    },
}

V5_CODE_PATH = os.path.join(os.path.dirname(__file__), "ml_optimized_picker_v5.py")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "v5_config.json")

# ─── 配置调节预设 ───
# review系统自动调优时, 根据accuracy调节confidence_weight的策略
TUNE_PRESETS = {
    "auto_reduce": {
        "desc": "准确性中等, 适度降低confidence比重",
        "steps": [0.15, 0.12, 0.10, 0.08],
    },
    "auto_reduce_low": {
        "desc": "准确率低, 大幅降低confidence比重",
        "steps": [0.15, 0.10, 0.07, 0.05],
    },
}

NAMES_HK = {
    "0700.HK":"腾讯","9988.HK":"阿里","9999.HK":"网易","1810.HK":"小米",
    "3690.HK":"美团","0941.HK":"中移动","0883.HK":"中海油",
    "0388.HK":"港交所","0005.HK":"汇丰","1299.HK":"友邦",
    "2269.HK":"药明","2382.HK":"舜宇","9618.HK":"京东","1024.HK":"快手",
    "0939.HK":"建行","3988.HK":"中行","0857.HK":"中石油",
    "0027.HK":"银河","1928.HK":"金沙","1177.HK":"中生",
}


def load_v5_module():
    """懒加载 v5 模块"""
    from ml_optimized_picker_v5 import US_WATCHLIST, HK_WATCHLIST
    from ml_optimized_picker_v5 import score_stock_v5, get_macro_data
    from ml_optimized_picker_v5 import run_ml_picking_v5, print_report_v5, save_results_v5
    from ml_optimized_picker_v5 import get_cached_data  # 复盘复用缓存
    return {
        "US_WATCHLIST": US_WATCHLIST,
        "HK_WATCHLIST": HK_WATCHLIST,
        "score_stock_v5": score_stock_v5,
        "get_macro_data": get_macro_data,
        "get_cached_data": get_cached_data,
        "run_ml_picking_v5": run_ml_picking_v5,
        "print_report_v5": print_report_v5,
        "save_results_v5": save_results_v5,
    }


# ═══════════════════════════════════════
# 状态管理
# ═══════════════════════════════════════
def load_state(market):
    f = MARKET_CONFIG[market]["state_file"]
    if os.path.exists(f):
        with open(f, "r") as fh:
            return json.load(fh)
    return {"market": market, "optimizations": [], "prediction_history": []}


def save_state(market, state):
    f = MARKET_CONFIG[market]["state_file"]
    with open(f, "w") as fh:
        json.dump(state, fh, indent=2, default=str)
    print(f"  💾 状态已保存 ({os.path.basename(f)})")


def _get_market_date(market, now=None):
    """
    获取当前市场的"日期"
    - HK: CST日���
    - US: 美股ET日期 (CST 04:30复盘对应前一天的ET日期)
    """
    if now is None:
        now = datetime.now()
    
    if market == "HK":
        return now.strftime("%Y-%m-%d")
    else:
        # 美股: CST → EDT (夏令时差12小时)
        # CST 20:00 = EDT 08:00, CST 04:30 = EDT 16:30(前一天)
        et_date = now - timedelta(hours=12)
        return et_date.strftime("%Y-%m-%d")


def _is_hk_trading_day(d=None):
    """粗略判断港股交易日 (周一到周五, 排除简单假期)"""
    if d is None:
        d = datetime.now()
    # 简单: 周一到周五
    return d.weekday() < 5


def _is_us_trading_day(d=None):
    """粗略判断美股交易日 (周一到周五)"""
    if d is None:
        d = datetime.now()
    return d.weekday() < 5


# ═══════════════════════════════════════
# Pre-Market: 开市前选股预测
# ══════════════════════════════��════════
def run_pre_market(market, sentiment=False):
    """开市前运行选股预测，保存结果到状态文件"""
    v5 = load_v5_module()
    state = load_state(market)
    today = _get_market_date(market)
    label = MARKET_CONFIG[market]["label"]

    if state.get("last_pre_date") == today:
        print(f"  {label}今天已在 {state.get('last_pre_time','?')} 运行过开市前预测，跳过。")
        return

    print(f"\n{'='*60}")
    print(f"  📊 {label}开市前选股预测 — {today}")
    print(f"{'='*60}")

    # 判断是否交易日
    if market == "HK" and not _is_hk_trading_day():
        print(f"  今天不是{label}交易日，跳过。")
        return
    if market == "US" and not _is_us_trading_day():
        print(f"  今天不是{label}交易日，跳过。")
        return

    tickers = v5["US_WATCHLIST"] if market == "US" else v5["HK_WATCHLIST"]
    print(f"  股票池: {len(tickers)} 只")

    # 运行v5评分
    results, errors = v5["run_ml_picking_v5"](
        tickers=tickers, market=market,
        force_refresh=False, verbose=True
    )

    if not results:
        print(f"  ❌ 没有成功评分的股票")
        return

    # 排序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 保存预测结果到状态文件
    predictions = []
    predictions_full = []
    for r in results:
        pred = {
            "ticker": r["ticker"],
            "name": NAMES_HK.get(r["ticker"], r["ticker"]),
            "score": r["score"],
            "direction": r["direction"],
            "confidence": round(r["models_consensus"], 4),
            "price": r["price"],
            "rank_pctl": r["rank_pctl"],
            "sector": r.get("sector", "other"),
            "mom_1m": r["mom_1m"],
            "mom_3m": r["mom_3m"],
            "walk_forward_r2": r["walk_forward_r2"],
            "direction_source": r.get("direction_source", "unknown"),
        }
        predictions.append(pred)
        predictions_full.append(r)

    state["last_pre_date"] = today
    state["last_pre_time"] = datetime.now().strftime("%H:%M")
    state["last_predictions"] = predictions
    state["last_pre_full"] = predictions_full  # 完整数据(用于复盘)
    save_state(market, state)

    # 输出Top N (自动)
    print(f"\n  📋 {label} Top {min(10, len(predictions))}")
    print(f"  {'#':>3} {'代码':>8} {'评分':>7} {'方向':>6} {'置信':>5} {'价格':>8} {'R²':>7}")
    print(f"  {'-'*50}")
    for i, p in enumerate(predictions[:10]):
        print(f"  {i+1:>3} {p['ticker']:>8} {p['score']:>7.3f} {p['direction']:>6}"
              f" {p['confidence']:>5.2f} {p['price']:>8.2f} {p['walk_forward_r2']:>7.3f}")

    # 按方向统计
    directions = {}
    for p in predictions:
        directions.setdefault(p["direction"], 0)
        directions[p["direction"]] += 1
    dir_str = " | ".join(f"{k}: {v}只" for k, v in sorted(directions.items()))
    print(f"\n  方向分布: {dir_str}")

    # ─── 异常事件检测 (--sentiment) ───
    if sentiment and predictions:
        print(f"\n  📰 加载新闻情绪 + 异常事件检测...")
        try:
            from finbert_sentiment import build_sentiment_factors, sentiment_boost

            all_tickers = [p["ticker"] for p in predictions]
            sentiment_factors = build_sentiment_factors(all_tickers)

            event_stocks = []
            for p in predictions:
                t = p["ticker"]
                sf = sentiment_factors.get(t, {})
                if sf and sf.get("news_count", 0) > 0:
                    original = p["score"]
                    fused, adj, evt_adj = sentiment_boost(original, sf)
                    p["score"] = fused
                    p["sentiment_adj"] = adj
                    p["event_adj"] = evt_adj
                    if sf.get("events"):
                        event_stocks.append((t, sf["event_labels"], sf.get("event_discount", 1.0)))

            predictions.sort(key=lambda x: x["score"], reverse=True)

            if event_stocks:
                print(f"\n  ⚠️  异常事件预警 ({len(event_stocks)} 只):")
                for t, labels, discount in event_stocks:
                    print(f"    🔴 {t}: {' + '.join(labels)} (折扣 {discount:.3f})")

            # 重新保存 state（包含情绪修正后的评分）
            save_state(market, state)
            print(f"  ✅ 情绪融合完成 ({len(all_tickers)} 只)")
        except ImportError:
            print(f"  ⚠️  finbert_sentiment.py 未找到，跳过情绪融合")
        except Exception as e:
            print(f"  ⚠️  情绪融合失败: {e}")

    # ─── 数据质量报告 ───
    r2s = [p["walk_forward_r2"] for p in predictions]
    avg_r2 = np.mean(r2s)
    neg_r2 = sum(1 for r in r2s if r < 0)
    price_missing = [p["ticker"] for p in predictions if not p.get("price") or p["price"] <= 0]
    print(f"\n  📊 数据质量: 平均R²={avg_r2:.2f} | 负R²={neg_r2}/{len(predictions)}")
    if price_missing:
        print(f"     ⚠️  收盘价缺失(回退到前一日): {', '.join(price_missing[:5])}")
    if avg_r2 < -0.1:
        print(f"     ⚠️  R²偏负（市场结构变化），已启用动量兜底保持排序稳定性")
    if len(price_missing) == 0:
        print(f"     ✅ 全部30只价格正常")

    print(f"  ✅ {label}开市前预测完成!")


# ═══════════════════════════════════════
# Post-Market: 收市后复盘对比
# ═══════════════════════════════════════
def run_post_market(market):
    """收市后复盘：对比预测 vs 实际，分析差异，自动优化"""
    import pandas as pd
    from scipy.stats import spearmanr
    import numpy as np

    v5 = load_v5_module()
    state = load_state(market)
    today = _get_market_date(market)
    label = MARKET_CONFIG[market]["label"]

    if state.get("last_pre_date") != today:
        print(f"  没有 {label} 今天开市前的预测记录。可能是非交易日或之前任务未执行。跳过复盘。")
        return

    print(f"\n{'='*60}")
    print(f"  📋 {label}收市复盘 — {today}")
    print(f"{'='*60}")

    predictions = state.get("last_predictions", [])
    print(f"  开市前预测: {len(predictions)} 只股票")

    if not predictions:
        print("  ⚠️ 预测记录为空，跳过复盘")
        return

    # 复用缓存: 用 get_cached_data 替代直接 yf.download, 减少YF调用
    v5 = load_v5_module()
    get_cached = v5["get_cached_data"]

    # 下载今日收盘数据对比
    compare = []
    errors = []
    for i, p in enumerate(predictions):
        ticker = p["ticker"]
        try:
            df = get_cached(ticker, period="2y", force_refresh=False)
            if df.empty or len(df) < 2:
                errors.append(ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            # 获取今日收盘涨跌幅
            closes = df["Close"].values.astype(float)
            if len(closes) < 2:
                errors.append(ticker)
                continue
            today_chg = (closes[-1] / closes[-2] - 1) * 100

            # 判断方向是否正确
            pred_dir = p["direction"]
            if today_chg > 0.5:
                actual_dir = "看涨"
            elif today_chg < -0.5:
                actual_dir = "看跌"
            else:
                actual_dir = "震荡"

            correct = (
                (pred_dir == actual_dir) or
                (pred_dir == "看涨" and today_chg > 0) or
                (pred_dir == "看跌" and today_chg < 0)
            )

            compare.append({
                "ticker": ticker,
                "name": p.get("name", ticker),
                "score": p["score"],
                "pred_dir": pred_dir,
                "actual_chg": round(today_chg, 2),
                "actual_dir": actual_dir,
                "correct": correct,
                "price": p.get("price", 0),
                "sector": p.get("sector", "other"),
            })
        except Exception as e:
            errors.append(f"{ticker}: {e}")

    if not compare:
        print("  ⚠️ 无法获取任何股票收盘数据")
        return

    # ─── 统计 ───
    total = len(compare)
    correct_count = sum(1 for c in compare if c["correct"])
    accuracy = correct_count / total if total > 0 else 0

    # Spearman: 评分 vs 实际涨跌幅
    scores = [c["score"] for c in compare]
    chgs = [c["actual_chg"] for c in compare if np.isfinite(c["actual_chg"])]
    scores_clean = [c["score"] for c in compare if np.isfinite(c["actual_chg"])]
    score_chg_corr, _ = spearmanr(scores_clean, chgs) if len(set(scores_clean)) > 1 and len(chgs) > 5 else (0, 1)
    score_chg_corr = score_chg_corr if not (isinstance(score_chg_corr, float) and np.isnan(score_chg_corr)) else 0

    # 方向偏差分析
    pred_up = sum(1 for c in compare if c["pred_dir"] == "看涨")
    pred_down = sum(1 for c in compare if c["pred_dir"] == "看跌")
    actual_up = sum(1 for c in compare if c["actual_chg"] > 0)
    actual_down = sum(1 for c in compare if c["actual_chg"] < 0)
    if pred_down > actual_down + 5:
        bias = "偏看跌(实际涨多于跌)"
    elif pred_up > actual_up + 5:
        bias = "偏看涨(实际跌多于涨)"
    else:
        bias = ""

    # ─── 自动生成优化建议 ───
    suggestions = []
    actions_taken = []
    config_changes = []

    # 1. 准确率判断
    if accuracy < 0.35:
        suggestions.append({
            "type": "low_accuracy",
            "severity": "critical",
            "detail": f"准确率仅{accuracy:.0%}，模型可能偏离市场。建议：①检查是否代码/参数有误 ②回滚到v4基准对比 ③检查宏观因子数据是否过期",
        })
    elif accuracy < 0.5:
        suggestions.append({
            "type": "moderate_accuracy",
            "severity": "high",
            "detail": f"准确率{accuracy:.0%}，仍有改进空间。建议检查模型是否过度依赖少数特征",
        })

    # 2. 方向偏差
    if bias:
        suggestions.append({
            "type": "direction_bias",
            "severity": "high",
            "detail": f"系统存在{bias}。建议：①检查21d分类器的看跌阈值 ②确认是否有政策/消息面利好",
        })

    # 3. 板块准确率
    sectors = {}
    for c in compare:
        sectors.setdefault(c["sector"], {"total": 0, "correct": 0})
        sectors[c["sector"]]["total"] += 1
        if c["correct"]:
            sectors[c["sector"]]["correct"] += 1
    for sector, stats in sectors.items():
        if stats["total"] >= 3:  # 至少3只才统计
            sec_acc = stats["correct"] / stats["total"]
            if sec_acc < 0.3:
                suggestions.append({
                    "type": "sector_mismatch",
                    "severity": "medium",
                    "detail": f"板块 {sector} 准确率仅 {sec_acc:.0%}。建议：检查该板块的宏观因子匹配是否正确(板块ETF vs 个股关联度)",
                })

    # 4. 高共识错误
    high_conf_wrong = [c for c in compare if c["score"] > 0.5 and not c["correct"]]
    if high_conf_wrong:
        tickers_str = ", ".join(f"{c['ticker']}({c['name']})" for c in high_conf_wrong[:5])
        suggestions.append({
            "type": "high_conf_failures",
            "severity": "high",
            "detail": f"以下高共识度股票预测错误: {tickers_str}。建议：检查这些股票是否有异常事件驱动(财报/并购/监管)",
        })

    # ─── 5. 方向信号来源分析 ───
    dir_sources = {}
    for p in predictions:
        src = p.get("direction_source", "unknown")
        dir_sources.setdefault(src, 0)
        dir_sources[src] += 1
    if dir_sources:
        src_str = ", ".join(f"{k}: {v}只" for k, v in sorted(dir_sources.items()))
        suggestions.append({
            "type": "direction_source",
            "severity": "info",
            "detail": f"方向信号来源: {src_str}",
        })

    # 6. 排名质量: Top 30% vs Bottom 30%
    if len(compare) >= 6:
        compare_sorted = sorted(compare, key=lambda x: x["score"], reverse=True)
        n_top = max(3, len(compare_sorted) // 3)
        valid_top = [c for c in compare_sorted[:n_top] if np.isfinite(c.get("actual_chg", np.nan))]
        valid_bot = [c for c in compare_sorted[-n_top:] if np.isfinite(c.get("actual_chg", np.nan))]
        if valid_top and valid_bot:
            top_avg = np.mean([c["actual_chg"] for c in valid_top])
            bot_avg = np.mean([c["actual_chg"] for c in valid_bot])
            spread = top_avg - bot_avg
            if spread > 0:
                suggestions.append({
                    "type": "rank_spread",
                    "severity": "info",
                    "detail": f"Top{n_top}平均涨跌{top_avg:+.2f}% vs Bottom{n_top}平均涨跌{bot_avg:+.2f}%, 差异{spread:+.2f}个百分点" + 
                              (" — 评分排序有效" if spread > 1 else " — 评分排序区分度偏弱"),
                })
            else:
                suggestions.append({
                    "type": "rank_spread_reverse",
                    "severity": "high",
                    "detail": f"评分排序与涨跌幅呈反向关系! Top{n_top}{top_avg:+.2f}% < Bottom{n_top}{bot_avg:+.2f}%, 差异{spread:+.2f}个百分点",
                })

    # ─── 5. 强烈推荐股票准确率
    strong_picks = [c for c in compare if c["score"] > 0.55]
    if strong_picks:
        strong_correct = sum(1 for c in strong_picks if c["correct"])
        strong_acc = strong_correct / len(strong_picks)
        if strong_acc < 0.5:
            suggestions.append({
                "type": "strong_pick_failures",
                "severity": "critical",
                "detail": f"强烈推荐股票(评分>0.55)准确率仅{strong_acc:.0%}！说明排名百分位失效。建议检查特征重要性是否集中在少数过拟合特征上",
            })

    # ─── 执行自动优化 ───
    if suggestions:
        print(f"\n  生成 {len(suggestions)} 条优化建议:")

        for s in suggestions:
            print(f"\n  [{s['severity'].upper()}] {s['type']}")
            print(f"  {s['detail']}")

        # 自动调整：根据问题类型生成结构化配置修改
        for s in suggestions:
            if s["type"] == "moderate_accuracy" and accuracy < 0.5:
                config_changes.append({
                    "keys": ["ml_scoring", "score_formula", "confidence_weight"],
                    "old": None,
                    "new": "auto_reduce",
                    "reason": "准确率中等, 适度降低模型一致性贡献"
                })
                actions_taken.append("降低confidence评分权重(自动调节)")

            if s["type"] == "low_accuracy":
                config_changes.append({
                    "keys": ["ml_scoring", "score_formula", "confidence_weight"],
                    "old": None,
                    "new": "auto_reduce_low",
                    "reason": "准确率低, 大幅降��模型一致性贡献"
                })
                actions_taken.append("大幅降低confidence权重(自动调节)")

        if not actions_taken:
            print(f"\n  → 本次优化建议已记录，暂无需自动执行的重度修改")

    # 保存优化记录
    opt_record = {
        "date": today,
        "market": market,
        "accuracy": round(accuracy, 4),
        "total": len(compare),
        "correct": sum(1 for c in compare if c["correct"]),
        "spearman_corr": round(score_chg_corr, 4),
        "bias": bias,
        "suggestions": suggestions,
        "actions_taken": actions_taken,
        "config_changes": config_changes,
    }
    state.setdefault("optimizations", []).append(opt_record)
    state.setdefault("prediction_history", []).append({
        "date": today,
        "accuracy": round(accuracy, 4),
        "total": len(compare),
        "correct": sum(1 for c in compare if c["correct"]),
        "optimized": bool(actions_taken),
    })
    save_state(market, state)

    # ─── 实际应用代码修改 ───
    for change in config_changes:
        _apply_v5_change(change, state, today)
        if isinstance(change, dict) and "action" not in change:
            change["action"] = "applied"

    # 保存优化记录 (更新后保存)

    # 输出最终摘要
    print(f"\n  {'='*60}")
    print(f"  📋 今日复盘总结")
    print(f"{'='*60}")
    print(f"  市场: {label}")
    print(f"  日期: {today}")
    print(f"  方向预测准确率: {accuracy:.1%} ({sum(1 for c in compare if c['correct'])}/{len(compare)})")
    print(f"  评分-涨跌幅秩相关: {score_chg_corr:+.3f}")
    if bias:
        print(f"  方向偏差: {bias}")
    if suggestions:
        print(f"  发现问题: {len(suggestions)} 个")
        for i, s in enumerate(suggestions[:3]):
            print(f"    {i+1}. [{s['severity'].upper()}] {s['type']}")
    if actions_taken:
        print(f"\n  🔧 自动优化执行:")
        for a in actions_taken:
            print(f"    ✅ {a}")
    else:
        print(f"\n  🟢 当前设置运行正常，未执行自动修改")

    return opt_record


def _load_config():
    """读取 v5_config.json"""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)

def _save_config(cfg):
    """写回 v5_config.json"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def _resolve_value(keys, cfg):
    """按 key 路径读取当前值"""
    target = cfg
    for k in keys:
        target = target.get(k)
        if target is None:
            return None
    return target

def _get_tune_history(state_or_market):
    """从 state dict 或 market name 读取历史调优记录"""
    if isinstance(state_or_market, dict):
        return state_or_market.get("tune_history", [])
    state = load_state(state_or_market)
    return state.get("tune_history", [])

def _apply_v5_change(change, state=None, today=None):
    """
    通用 JSON patcher: 根据 config_changes 结构化指令修改 v5_config.json
    支持的 change 格式:
      {"keys": ["ml_scoring", "score_formula", "confidence_weight"],
       "new": 0.12}                       # 直接设定新值
      {"keys": [...], "new": "auto_reduce"}  # 按预设阶梯调优
    """
    if not isinstance(change, dict) or "keys" not in change:
        print(f"  跳过非结构化变更: {change}")
        return

    cfg = _load_config()
    keys = change["keys"]
    reason = change.get("reason", "")
    old_val = _resolve_value(keys, cfg)
    new_val = change.get("new")

    if new_val is None:
        print(f"  无新值: {'.'.join(keys)}")
        return

    # 预设调优: 根据历史调节记录自动降级
    if isinstance(new_val, str) and new_val.startswith("auto_"):
        preset = TUNE_PRESETS.get(new_val)
        if preset:
            tune_key = ".".join(keys)
            history = _get_tune_history(state) if state else []
            prev_idx = -1
            for h in history:
                if h.get("key") == tune_key:
                    prev_idx = h.get("step_index", -1)  # 取最后一条匹配记录
            step_idx = prev_idx + 1
            if step_idx >= len(preset["steps"]):
                step_idx = len(preset["steps"]) - 1
            new_val = preset["steps"][step_idx]
            change["step_index"] = step_idx

    # 写入配置
    target = cfg
    for k in keys[:-1]:
        target = target.setdefault(k, {})  # 确保中间路径存在
    old_str = str(target.get(keys[-1], "?"))
    target[keys[-1]] = new_val
    _save_config(cfg)

    print(f"  \u2705 \u914d\u7f6e\u66f4\u65b0: {' > '.join(keys)}: {old_str} -> {new_val}")
    if reason:
        print(f"     \u539f\u56e0: {reason}")

    # 记录调优历史
    if state:
        tune_history = state.setdefault("tune_history", [])
        tune_history.append({
            "date": today or datetime.now().strftime("%Y-%m-%d"),
            "key": ".".join(keys),
            "old": old_val,
            "new": new_val,
            "reason": reason,
            "step_index": change.get("step_index", -1),
        })
        save_state(state.get("market", "US"), state)
    print(f"  \u2705 \u914d\u7f6e\u5df2\u4fdd\u5b58: {CONFIG_PATH}")


# ──────────────────────────────────
# CLI
# ──────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML量化选股 cron job")
    parser.add_argument("--market", required=True, choices=["HK", "US"])
    parser.add_argument("--mode", required=True, choices=["pre", "post"])
    parser.add_argument("--sentiment", action="store_true", help="融合新闻情绪因子和异常事件检测")
    args = parser.parse_args()

    if args.mode == "pre":
        run_pre_market(args.market, sentiment=args.sentiment)
    else:
        result = run_post_market(args.market)
        if result:
            # no_agent模式: 输出JSON到stdout用于cron通知
            summary = {
                "market": args.market,
                "date": result["date"],
                "accuracy": result["accuracy"],
                "total": result["total"],
                "correct": result["correct"],
                "issues": len(result["suggestions"]),
                "actions": result["actions_taken"],
            }
            print(f"\n__CRON_RESULT__:{json.dumps(summary)}")
