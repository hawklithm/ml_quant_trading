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

import sys, os, json, time, re
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
    return {
        "US_WATCHLIST": US_WATCHLIST,
        "HK_WATCHLIST": HK_WATCHLIST,
        "score_stock_v5": score_stock_v5,
        "get_macro_data": get_macro_data,
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
        # 美股: CST 04:30对应前一个ET交易日结束
        # 用美东时间判断
        et_hour = now.hour - 12  # ET = CST - 1 (夏令时)
        if et_hour < 0:
            et_hour += 24
        et_date = now
        if et_hour < 8:  # ET早于08:00 → 前一个交易日
            et_date = now - timedelta(days=1)
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
def run_pre_market(market):
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
    print(f"  ✅ {label}开市前预测完成!")


# ═══════════════════════════════════════
# Post-Market: 收市后复盘对比
# ═══════════════════════════════════════
def run_post_market(market):
    """收市后复盘：对比预测 vs 实际，分析差异，自动优化"""
    import yfinance as yf
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

    # 下载今日收盘数据对比
    compare = []
    errors = []
    for i, p in enumerate(predictions):
        ticker = p["ticker"]
        try:
            df = yf.download(ticker, period="5d", auto_adjust=True, progress=False)
            if df.empty:
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
    chgs = [c["actual_chg"] for c in compare]
    score_chg_corr, _ = spearmanr(scores, chgs) if len(set(scores)) > 1 else (0, 1)

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

    # 5. 强烈推荐股票准确率
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

        # 自动调整：准确率低时降低confidence权重
        for s in suggestions:
            if s["type"] == "moderate_accuracy" and accuracy < 0.5:
                config_changes.append("评分公式: confidence 权重 0.4→0.3 (准确率低, 降低模型一致性贡献)")
                actions_taken.append("降低confidence评分权重到0.3")

            if s["type"] == "low_accuracy":
                config_changes.append("评分公式: confidence 权重 0.4→0.25 (准确率<40%)")
                actions_taken.append("大幅降低confidence权重到0.25")

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
        _apply_v5_change(change)

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


def _apply_v5_change(change_str):
    """根据复盘优化建议，实际修改 ml_optimized_picker_v5.py 代码"""
    if not os.path.exists(V5_CODE_PATH):
        print(f"  ⚠️ v5代码未找到: {V5_CODE_PATH}")
        return

    with open(V5_CODE_PATH, "r") as f:
        code = f.read()

    modified = False

    # 匹配: confidence 权重 0.4→0.25 / 0.4→0.3
    # change_str如: "评分公式: confidence 权重 0.4→0.25 (准确率<40%)"
    m = re.search(r'confidence\s*权重\s*[\d.]+→([\d.]+)', change_str)
    if m:
        new_weight = float(m.group(1))
        # 找代码里的 final_score 行
        m2 = re.search(
            r'final_score\s*=\s*latest_rank\s*\*\s*[\d.]+\s*\+\s*confidence\s*\*\s*[\d.]+',
            code
        )
        if m2:
            old_line = m2.group(0)
            rank_w = round(1.0 - new_weight, 2)
            new_line = f"final_score = latest_rank * {rank_w} + confidence * {new_weight}"
            code = code.replace(old_line, new_line)
            modified = True
            print(f"  ✅ 已应用代码修改: {old_line}")
            print(f"     → {new_line}")

    if modified:
        with open(V5_CODE_PATH, "w") as f:
            f.write(code)
        print(f"  ✅ v5代码已更新: {V5_CODE_PATH}")
    else:
        print(f"  ℹ️  无需代码修改: {change_str}")


# ──────────────────────────────────
# CLI
# ──────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ML量化选股 cron job")
    parser.add_argument("--market", required=True, choices=["HK", "US"])
    parser.add_argument("--mode", required=True, choices=["pre", "post"])
    args = parser.parse_args()

    if args.mode == "pre":
        run_pre_market(args.market)
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
