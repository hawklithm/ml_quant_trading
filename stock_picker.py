#!/usr/bin/env python3
"""
A股多因子选股系统 v1

功能:
  1. 获取全市场 A股实时行情 + 历史数据
  2. 计算多维度因子 (动量/质量/波动/资金)
  3. 多因子打分排名
  4. 输出 Top N 选股结果
  4. 可视化因子分布

用法:
  python stock_picker.py                # 默认输出 Top 30
  python stock_picker.py --top 50       # 输出 Top 50
  python stock_picker.py --sector       # 按行业分组输出
  python stock_picker.py --save         # 保存结果到 CSV
  python stock_picker.py --detail       # 显示完整因子明细

注意:
  - 首次运行需要下载数据, 约 1-2 分钟
  - 仅在工作日交易时段有实时数据
  - 因子权重可自定义 (见 FACTOR_WEIGHTS)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import warnings
warnings.filterwarnings("ignore")

plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["font.size"] = 10

# ──────────── 配置 ────────────
TOP_N = 30      # 默认输出前30只
MIN_TRADING_DAYS = 120  # 最少交易天数
LIQUIDITY_THRESHOLD = 50000000  # 最低日均成交额 5000万

# 因子权重配置
FACTOR_WEIGHTS = {
    # 动量因子 (40%)
    "momentum_1m":      0.08,  # 1月动量
    "momentum_3m":      0.12,  # 3月动量
    "momentum_6m":      0.10,  # 6月动量
    "momentum_12m":     0.10,  # 12月动量

    # 质量因子 (25%)
    "sma20_dev":        0.05,  # 价格/20日均线偏离
    "volatility_1m":   -0.05,  # 月度波动率 (负权重, 低波动好)
    "max_dd_3m":       -0.05,  # 3月最大回撤 (负权重)
    "volume_trend":     0.05,  # 成交量趋势
    "rsi_14":           0.05,  # RSI (正常区间中值)

    # 资金/情绪因子 (20%)
    "fund_flow_5d":     0.10,  # 5日资金净流入
    "fund_flow_10d":    0.05,  # 10日资金净流入
    "turnover_rate":    0.05,  # 换手率 (适度活跃)

    # 基础因子 (15%)
    "price_level":     -0.05,  # 价格位置 (负权重, 低位好)
    "volume_ratio":     0.05,  # 量比
    "amplitude":        0.05,  # 振幅
}


# ═══════════════════════════════════════════════
# 一、数据获取
# ═══════════════════════════════════════════════

def get_all_stocks():
    """获取全市场股票列表及实时行情"""
    import akshare as ak
    print("获取 A股全市场实时行情...")

    df = ak.stock_zh_a_spot_em()
    print(f"  全市场股票数: {len(df)}")

    # 标准化列名
    df = df.rename(columns={
        "代码": "code", "名称": "name", "最新价": "price",
        "涨跌幅": "change_pct", "涨跌额": "change",
        "成交量": "volume", "成交额": "amount",
        "振幅": "amplitude", "最高": "high", "最低": "low",
        "今开": "open", "昨收": "pre_close",
        "量比": "volume_ratio", "换手率": "turnover",
        "市盈率-动态": "pe", "市净率": "pb",
        "总市值": "market_cap", "流通市值": "float_mv",
        "60日涨跌幅": "chg_60d",
    })

    # 修改列名匹配
    rename_map = {}
    for col in df.columns:
        if "60日" in str(col):
            rename_map[col] = "chg_60d"
        elif "涨跌幅" in str(col):
            rename_map[col] = "change_pct"
        elif "成交额" in str(col):
            rename_map[col] = "amount"
        elif "换手率" in str(col):
            rename_map[col] = "turnover"

    df = df.rename(columns=rename_map)

    # 只保留有代码的表
    if "code" not in df.columns:
        # 尝试找代码列
        for col in df.columns:
            if "代码" in str(col) or "code" in str(col).lower():
                df = df.rename(columns={col: "code"})
                break
        else:
            print("  警告: 无法识别股票代码列")
            return None

    return df


def get_hist_data(code, days=365):
    """获取个股历史日线"""
    import akshare as ak
    try:
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days)
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq"
        )
        if df.empty:
            return None
        # 标准列名
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close",
            "最高": "high", "最低": "low", "成交量": "volume",
            "成交额": "amount", "振幅": "amplitude",
            "涨跌幅": "change_pct", "涨跌额": "change",
            "换手率": "turnover"
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        return None


# ═══════════════════════════════════════════════
# 二、因子计算
# ═══════════════════════════════════════════════

def calc_factors(hist_df, spot_row):
    """计算单只股票的因子值"""
    if hist_df is None or len(hist_df) < MIN_TRADING_DAYS:
        return None

    close = hist_df["close"].values.astype(float)
    factors = {}

    # 动量因子
    factors["momentum_1m"] = (close[-1] / close[-21] - 1) if len(close) >= 21 else 0
    factors["momentum_3m"] = (close[-1] / close[-63] - 1) if len(close) >= 63 else 0
    factors["momentum_6m"] = (close[-1] / close[-126] - 1) if len(close) >= 126 else 0
    factors["momentum_12m"] = (close[-1] / close[-252] - 1) if len(close) >= 252 else 0

    # 价格偏离
    sma20 = pd.Series(close).rolling(20).mean().values[-1]
    factors["sma20_dev"] = (close[-1] / sma20 - 1) if sma20 > 0 else 0

    # 波动率 (年化)
    returns = pd.Series(close).pct_change().dropna().values
    factors["volatility_1m"] = np.std(returns[-21:]) * np.sqrt(252) if len(returns) >= 21 else 0

    # 最大回撤
    rolling_max = pd.Series(close[-63:]).cummax().values if len(close) >= 63 else close
    factors["max_dd_3m"] = (np.min(rolling_max - close[-len(rolling_max):]) / np.max(rolling_max)) if len(rolling_max) > 0 else 0

    # RSI
    if len(returns) >= 14:
        gains = np.maximum(returns[-14:], 0)
        losses = -np.minimum(returns[-14:], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            factors["rsi_14"] = 100 - (100 / (1 + rs))
        else:
            factors["rsi_14"] = 100 if avg_gain > 0 else 50
    else:
        factors["rsi_14"] = 50

    # 成交量趋势
    volumes = hist_df["volume"].values.astype(float)
    vol_sma20 = pd.Series(volumes).rolling(20).mean().values[-1] if len(volumes) >= 20 else np.mean(volumes)
    factors["volume_trend"] = (volumes[-1] / vol_sma20 - 1) if vol_sma20 > 0 else 0

    # 价格位置 (在60日高低区间的位置)
    if len(close) >= 60:
        high_60 = np.max(close[-60:])
        low_60 = np.min(close[-60:])
        factors["price_level"] = (close[-1] - low_60) / (high_60 - low_60) if (high_60 - low_60) > 0 else 0.5
    else:
        factors["price_level"] = 0.5

    # 换手率
    factors["turnover_rate"] = spot_row.get("turnover", 50)
    # 标准化
    factors["turnover_rate"] = min(factors["turnover_rate"] / 10, 10)

    # 量比
    factors["volume_ratio"] = spot_row.get("volume_ratio", 1)

    # 振幅
    factors["amplitude"] = spot_row.get("amplitude", 0)

    return factors


# ═══════════════════════════════════════════════
# 三、评分系统
# ═══════════════════════════════════════════════

def score_stock(factors):
    """根据因子加权计算综合评分"""
    if factors is None:
        return -999

    score = 0
    details = {}

    for factor_name, weight in FACTOR_WEIGHTS.items():
        if factor_name not in factors:
            continue

        raw_value = factors[factor_name]

        # 对原始值做简单的标准化/截断处理
        if factor_name.startswith("momentum"):
            # 涨幅 >30% 或 <-30% 截断
            clipped = np.clip(raw_value, -0.30, 0.30)
            normalized = clipped / 0.30
        elif factor_name == "sma20_dev":
            normalized = np.clip(raw_value / 0.15, -1, 1)
        elif factor_name == "volatility_1m":
            normalized = -np.clip(raw_value / 0.50, -1, 1)
        elif factor_name == "max_dd_3m":
            normalized = -np.clip(np.abs(raw_value) / 0.30, 0, 1)
        elif factor_name == "volume_trend":
            normalized = np.clip(raw_value / 2, -1, 1)
        elif factor_name == "rsi_14":
            # RSI 偏离50越远, 极端值扣分
            normalized = (50 - abs(raw_value - 50)) / 50
        elif factor_name == "price_level":
            # 价格在低位 (<0.3) 加分, 高位 (>0.7) 扣分
            normalized = -np.clip((raw_value - 0.3) / 0.4, -1, 1)
        elif factor_name == "turnover_rate":
            normalized = np.clip(raw_value / 10, 0, 1)
        elif factor_name == "volume_ratio":
            normalized = np.clip(raw_value / 3, -1, 1)
        elif factor_name == "amplitude":
            normalized = np.clip(raw_value / 10, 0, 1)
        elif factor_name == "fund_flow_5d" or factor_name == "fund_flow_10d":
            normalized = np.clip(raw_value, -1, 1)
        else:
            normalized = np.clip(raw_value, -1, 1)

        contribution = normalized * weight
        score += contribution
        details[factor_name] = {
            "raw": raw_value,
            "normalized": round(normalized, 3),
            "weight": weight,
            "contribution": round(contribution, 3)
        }

    return score, details


# ═══════════════════════════════════════════════
# 四、资金流向 (AkShare)
# ═══════════════════════════════════════════════

def get_fund_flow(code):
    """获取个股资金流向"""
    import akshare as ak
    try:
        df = ak.stock_individual_fund_flow(stock=code, market="sh" if code.startswith("6") else "sz")
        if df.empty:
            return {}
        # 取最近5日和10日
        # 列名可能是中文, 取最近5行
        if len(df) >= 5:
            flow_5d = df.head(5)["主力净流入-净额"].sum() if "主力净流入-净额" in df.columns else 0
        else:
            flow_5d = 0
        if len(df) >= 10:
            flow_10d = df.head(10)["主力净流入-净额"].sum()
        else:
            flow_10d = flow_5d

        # 标准化: 用成交额归一化
        return {"fund_flow_5d": flow_5d / 1e8, "fund_flow_10d": flow_10d / 1e8}
    except Exception:
        return {}


# ═══════════════════════════════════════════════
# 五、主流程
# ═══════════════════════════════════════════════

def stock_picking(filter_sector=None, top_n=TOP_N):
    """主选股流程"""
    t0 = time.time()

    # 1. 获取全市场股票列表
    spot_df = get_all_stocks()
    if spot_df is None:
        print("无法获取市场数据, 请检查网络")
        return None

    # 筛选条件: 成交量>0 且 有价格
    spot_df = spot_df[spot_df["price"] > 0].copy()

    # 如果指定了板块过滤
    if filter_sector:
        spot_df = spot_df[spot_df["name"].str.contains(filter_sector, na=False)]

    print(f"  有效交易股票: {len(spot_df)}")
    print(f"  正在逐个分析...")

    # 2. 逐个股票计算因子和评分
    results = []
    codes = spot_df["code"].tolist()
    total = len(codes)

    for idx, code in enumerate(codes):
        if (idx + 1) % 50 == 0 or idx == 0:
            print(f"  进度: {idx+1}/{total}", end="\r")

        row = spot_df.iloc[idx]
        name = row.get("name", "")

        # 数据检查: 价格合理
        price = float(row["price"])
        if price <= 0:
            continue

        # 流动性过滤
        amount = float(row.get("amount", 0))
        if amount < LIQUIDITY_THRESHOLD:
            continue

        # 获取历史数据
        hist = get_hist_data(code)
        if hist is None:
            continue

        # 计算因子
        factors = calc_factors(hist, row)
        if factors is None:
            continue

        # 补充资金流向
        # fund_flow = get_fund_flow(code)
        # factors.update(fund_flow)

        # 评分
        total_score, details = score_stock(factors)

        results.append({
            "code": code,
            "name": name,
            "price": price,
            "score": total_score,
            "factors": factors,
            "details": details,
            "market_cap": row.get("market_cap", 0),
            "industry": row.get("name", "")[0:0],
            "change_pct": row.get("change_pct", 0),
            "pe": row.get("pe", 0),
        })

    # 3. 排名
    results.sort(key=lambda x: x["score"], reverse=True)
    top = results[:top_n]

    elapsed = time.time() - t0
    print(f"\n  总耗时: {elapsed:.0f}秒")
    print(f"  分析了 {len(results)}/{total} 只股票(���滤后)")

    return top, results


# ═══════════════════════════════════════════════
# 六、输出
# ═══════════════════════════════════════════════

def print_results(top, all_results, detail=False):
    """打印选股结果"""
    print("\n" + "=" * 80)
    print(f"A股多因子选股结果 Top {len(top)}")
    print(f"数据日期: {datetime.date.today()}")
    print("=" * 80)

    # 表头
    print(f"{'排名':>4} {'代码':>8} {'名称':<10} {'评分':>6} {'价格':>8} "
          f"{'1月动':>6} {'3月动':>6} {'6月动':>6} {'偏离':>6} "
          f"{'波动':>6} {'RSI':>5} {'换手':>5} {'量比':>5}")
    print("-" * 80)

    for i, s in enumerate(top):
        f = s["factors"]
        print(f"{i+1:>4} {s['code']:>8} {s['name']:<10} "
              f"{s['score']:>6.2f} {s['price']:>8.2f} "
              f"{f.get('momentum_1m',0)*100:>5.1f}% "
              f"{f.get('momentum_3m',0)*100:>5.1f}% "
              f"{f.get('momentum_6m',0)*100:>5.1f}% "
              f"{f.get('sma20_dev',0)*100:>5.1f}% "
              f"{f.get('volatility_1m',0)*100:>5.1f}% "
              f"{f.get('rsi_14',50):>5.0f} "
              f"{f.get('turnover_rate',0):>5.1f} "
              f"{f.get('volume_ratio',1):>5.2f}")

    print("-" * 80)

    # 因子贡献
    if detail:
        print(f"\n{'='*80}")
        print(f"Top 10 因子明细")
        print(f"{'='*80}")
        for i, s in enumerate(top[:10]):
            print(f"\n  #{i+1} {s['name']} ({s['code']})  评分: {s['score']:.2f}")
            for fn, fd in sorted(s["details"].items(), key=lambda x: abs(x[1]["contribution"]), reverse=True)[:8]:
                arrow = "+" if fd["contribution"] > 0 else ""
                print(f"    {fn:<18} raw={fd['raw']:<10.4f} → norm={fd['normalized']:<8.3f} × w={fd['weight']:<6.2f} = {arrow}{fd['contribution']:.3f}")

    # 统计
    print(f"\n{'='*80}")
    print("统计概览")
    print(f"{'='*80}")
    avg_score = np.mean([s["score"] for s in top])
    avg_mom_3m = np.mean([s["factors"].get("momentum_3m", 0) for s in top])
    avg_rsi = np.mean([s["factors"].get("rsi_14", 50) for s in top])
    print(f"  Top{len(top)} 平均评分: {avg_score:.2f}")
    print(f"  Top{len(top)} 平均3月动量: {avg_mom_3m*100:.1f}%")
    print(f"  Top{len(top)} 平均RSI: {avg_rsi:.0f}")


def save_results(top, filename="stock_picks"):
    """保存到 CSV"""
    if not top:
        return

    rows = []
    for s in top:
        row = {
            "code": s["code"], "name": s["name"],
            "price": s["price"], "score": round(s["score"], 3),
        }
        row.update(s["factors"])
        rows.append(row)

    df = pd.DataFrame(rows)
    # 把因子展开
    for fn in FACTOR_WEIGHTS:
        if fn not in df.columns:
            df[fn] = 0

    csv_path = f"{filename}_{datetime.date.today()}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n已保存: {csv_path}")


def plot_results(top, all_results):
    """画因子分布图"""
    if not top:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. 评分分布
    scores = [s["score"] for s in all_results]
    ax = axes[0, 0]
    ax.hist(scores, bins=40, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(np.percentile(scores, 90), color="green", ls="--", lw=1.5, label="Top 10%")
    ax.axvline(np.median(scores), color="orange", ls="--", lw=1, label="中位数")
    ax.set_title("综合评分分布", fontweight="bold")
    ax.set_xlabel("评分")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # 2. 3月动量分布
    mom3 = [s["factors"].get("momentum_3m", 0) * 100 for s in all_results]
    ax = axes[0, 1]
    ax.hist(mom3, bins=40, color="#FF9800", alpha=0.7, edgecolor="white")
    ax.axvline(0, color="red", ls="-", lw=0.5)
    ax.set_title("3月动量分布", fontweight="bold")
    ax.set_xlabel("3月动量 (%)")
    ax.grid(True, alpha=0.25)

    # 3. Top N 评分对比
    ax = axes[0, 2]
    names = [s["name"][:4] for s in top[:15]]
    scores_top = [s["score"] for s in top[:15]]
    colors = plt.cm.RdYlGn(np.array(scores_top) / max(abs(s) for s in scores_top + [1]))
    ax.barh(names[::-1], scores_top[::-1], color=colors[::-1])
    ax.set_title(f"Top {min(15, len(top))} 评分排名", fontweight="bold")
    ax.set_xlabel("综合评分")
    ax.axvline(0, color="#666", ls="-", lw=0.5)

    # 4. 动量与波动散点
    ax = axes[1, 0]
    mom1 = [s["factors"].get("momentum_1m", 0) * 100 for s in all_results]
    vols = [s["factors"].get("volatility_1m", 0) * 100 for s in all_results]
    ax.scatter(vols, mom1, alpha=0.3, s=5, c="#9C27B0")
    ax.set_xlabel("月波动率 (%)")
    ax.set_ylabel("月动量 (%)")
    ax.set_title("动量 vs 波动率")
    ax.grid(True, alpha=0.25)

    # 5. RSI vs 评分
    ax = axes[1, 1]
    rsis = [s["factors"].get("rsi_14", 50) for s in all_results]
    ax.scatter(rsis, scores, alpha=0.3, s=5, c="#4CAF50")
    ax.axvline(30, color="green", ls="--", lw=0.5, alpha=0.5)
    ax.axvline(70, color="red", ls="--", lw=0.5, alpha=0.5)
    ax.set_xlabel("RSI (14)")
    ax.set_ylabel("综合评分")
    ax.set_title("RSI vs 评分")
    ax.grid(True, alpha=0.25)

    # 6. 价格位置分布
    ax = axes[1, 2]
    prices = [s["factors"].get("price_level", 0.5) for s in all_results]
    ax.hist(prices, bins=30, color="#F44336", alpha=0.7, edgecolor="white")
    ax.axvline(0.3, color="green", ls="--", lw=1, label="低位")
    ax.axvline(0.7, color="red", ls="--", lw=1, label="高位")
    ax.set_title("价格位置分布 (60日区间)", fontweight="bold")
    ax.set_xlabel("位置 (0=最低, 1=最高)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig("stock_picks_distribution.png", dpi=150, bbox_inches="tight")
    print(f"\n图表: stock_picks_distribution.png")


# ═══════════════════════════════════════════════
# 七、CLI
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    top_n = TOP_N
    detail = False
    save = False

    for arg in sys.argv[1:]:
        if arg.startswith("--top"):
            try:
                top_n = int(arg.split("=")[1])
            except:
                if sys.argv[sys.argv.index(arg) + 1]:
                    top_n = int(sys.argv[sys.argv.index(arg) + 1])
        elif arg == "--detail":
            detail = True
        elif arg == "--save":
            save = True

    sector = None
    if "--sector" in sys.argv:
        idx = sys.argv.index("--sector")
        if idx + 1 < len(sys.argv):
            sector = sys.argv[idx + 1]

    top, all_results = stock_picking(filter_sector=sector, top_n=top_n)
    if top:
        print_results(top, all_results, detail=detail)
        if save:
            save_results(top)
        plot_results(top, all_results)
