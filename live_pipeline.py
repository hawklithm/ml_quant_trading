#!/usr/bin/env python3
"""
实时行情 + 模拟交易管道

架构:
  - 行情源: yfinance (定时轮询)
  - 存储: SQLite (本地持久化)
  - 策略引擎: 内置的简单均线策略
  - 模拟交易: 模拟券商接口

用法:
  python live_pipeline.py             # 运行实时管道
  python live_pipeline.py --backfill   # 回填历史数据
  python live_pipeline.py --report    # 查看持仓和交易记录

依赖:
  pip install apscheduler   (已内置在代码中)
"""

import sys
import json
import sqlite3
import time
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

DB_PATH = "live_trading.db"

# ═══════════════════════════════════════════════
# 数据库初始化
# ═══════════════════════════════════════════════

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 行情数据
    c.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (ticker, timestamp)
        )
    """)

    # 交易记录
    c.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            side TEXT,  -- BUY/SELL
            price REAL,
            quantity INTEGER,
            timestamp TEXT,
            strategy TEXT,
            pnl REAL DEFAULT 0
        )
    """)

    # 持仓
    c.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            quantity INTEGER DEFAULT 0,
            avg_cost REAL DEFAULT 0,
            last_updated TEXT
        )
    """)

    # 策略配置
    c.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════
# 行情获取
# ═══════════════════════════════════════════════

def fetch_prices(tickers):
    """获取最新行情"""
    try:
        data = yf.download(tickers, period="1d", interval="1m",
                           auto_adjust=True, progress=False)
        if data.empty:
            return {}

        if isinstance(tickers, str):
            tickers = [tickers]

        result = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    close_val = float(data["Close"].iloc[-1])
                    open_val = float(data["Open"].iloc[-1])
                    high_val = float(data["High"].iloc[-1])
                    low_val = float(data["Low"].iloc[-1])
                    vol_val = int(data["Volume"].iloc[-1]) if not np.isnan(data["Volume"].iloc[-1]) else 0
                else:
                    close_val = float(data["Close"][t].iloc[-1])
                    open_val = float(data["Open"][t].iloc[-1])
                    high_val = float(data["High"][t].iloc[-1])
                    low_val = float(data["Low"][t].iloc[-1])
                    vol_val = int(data["Volume"][t].iloc[-1])

                result[t] = {
                    "open": open_val, "high": high_val,
                    "low": low_val, "close": close_val,
                    "volume": vol_val,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            except:
                continue

        return result
    except Exception as e:
        return {"error": str(e)}


def backfill_history(tickers, years=2):
    """回填历史数据到数据库"""
    conn = sqlite3.connect(DB_PATH)
    count = 0

    for ticker in tickers:
        print(f"回填 {ticker}...")
        df = yf.download(ticker, period=f"{years}y", auto_adjust=True, progress=False)
        if df.empty:
            continue
        df.columns = [c[0] for c in df.columns]

        for idx, row in df.iterrows():
            ts = idx.strftime("%Y-%m-%d")
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO prices (ticker, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (ticker, ts, float(row["Open"]), float(row["High"]),
                      float(row["Low"]), float(row["Close"]), int(row["Volume"])))
                count += 1
            except:
                continue

    conn.commit()
    conn.close()
    print(f"回填完成: {count} 条记录")


def save_prices(data):
    """保存行情到 DB"""
    conn = sqlite3.connect(DB_PATH)
    saved = 0
    for ticker, info in data.items():
        if ticker == "error":
            continue
        try:
            conn.execute("""
                INSERT OR REPLACE INTO prices (ticker, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (ticker, info["timestamp"], info["open"], info["high"],
                  info["low"], info["close"], info["volume"]))
            saved += 1
        except Exception as e:
            print(f"  保存 {ticker} 失败: {e}")
    conn.commit()
    conn.close()
    return saved


# ═══════════════════════════════════════════════
# 策略引擎
# ════════════════════════���══════════════════════

class StrategyEngine:
    """可插拔策略引擎"""

    def __init__(self, strategy="sma"):
        self.strategy = strategy

    def evaluate(self, ticker, historical_data):
        """
        评估当前是否应该交易
        返回: {"action": "BUY"/"SELL"/"HOLD", "reason": "xxx"}
        """
        if self.strategy == "sma":
            return self._sma_cross(ticker, historical_data)
        elif self.strategy == "rsi":
            return self._rsi_strategy(ticker, historical_data)
        return {"action": "HOLD", "reason": "未知策略"}

    def _sma_cross(self, ticker, df):
        """双均线交叉"""
        if df is None or len(df) < 60:
            return {"action": "HOLD", "reason": "数据不足"}

        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA60"] = df["Close"].rolling(60).mean()

        prev = df.iloc[-2]
        curr = df.iloc[-1]

        if pd.isna(prev["SMA20"]) or pd.isna(prev["SMA60"]):
            return {"action": "HOLD", "reason": "均线数据不足"}

        prev_cross = prev["SMA20"] > prev["SMA60"]
        curr_cross = curr["SMA20"] > curr["SMA60"]

        if not prev_cross and curr_cross:
            return {"action": "BUY", "reason": f"SMA20({curr['SMA20']:.1f})金叉SMA60({curr['SMA60']:.1f})"}
        elif prev_cross and not curr_cross:
            return {"action": "SELL", "reason": f"SMA20({curr['SMA20']:.1f})死叉SMA60({curr['SMA60']:.1f})"}

        return {"action": "HOLD", "reason": "持仓不变"}

    def _rsi_strategy(self, ticker, df):
        """RSI 超买超卖"""
        if df is None or len(df) < 20:
            return {"action": "HOLD", "reason": "数据不足"}

        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        curr_rsi = rsi.iloc[-1]

        if curr_rsi < 30:
            return {"action": "BUY", "reason": f"RSI={curr_rsi:.1f} 超卖"}
        elif curr_rsi > 70:
            return {"action": "SELL", "reason": f"RSI={curr_rsi:.1f} 超买"}

        return {"action": "HOLD", "reason": f"RSI={curr_rsi:.1f} 正常区间"}


# ═══════════════════════════════════════════════
# 模拟交易执行
# ═══════════════════════════════════════════════

class PaperBroker:
    """模拟券商"""

    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash

    def get_cash(self):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT value FROM config WHERE key='cash'")
        row = c.fetchone()
        conn.close()
        if row:
            return float(row[0])
        return self.initial_cash

    def set_cash(self, amount):
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES ('cash', ?)", (str(amount),))
        conn.commit()
        conn.close()

    def get_position(self, ticker):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT quantity, avg_cost FROM positions WHERE ticker=?", (ticker,))
        row = c.fetchone()
        conn.close()
        if row:
            return {"quantity": row[0], "avg_cost": row[1]}
        return {"quantity": 0, "avg_cost": 0}

    def execute(self, ticker, action, price, quantity=100):
        """执行交易"""
        conn = sqlite3.connect(DB_PATH)
        now = datetime.datetime.now().isoformat()

        if action == "BUY":
            cost = price * quantity
            cash = self.get_cash()

            if cost > cash:
                conn.close()
                return {"status": "REJECTED", "reason": "现金不足"}

            self.set_cash(cash - cost)

            # 更新持仓
            pos = self.get_position(ticker)
            new_qty = pos["quantity"] + quantity
            new_cost = (pos["avg_cost"] * pos["quantity"] + cost) / new_qty if new_qty > 0 else price
            conn.execute("INSERT OR REPLACE INTO positions (ticker, quantity, avg_cost, last_updated) VALUES (?, ?, ?, ?)",
                        (ticker, new_qty, new_cost, now))

            conn.execute("INSERT INTO trades (ticker, side, price, quantity, timestamp, strategy) VALUES (?, 'BUY', ?, ?, ?, 'live')",
                        (ticker, price, quantity, now))

        elif action == "SELL":
            pos = self.get_position(ticker)
            if pos["quantity"] < quantity:
                quantity = pos["quantity"]
            if quantity <= 0:
                conn.close()
                return {"status": "REJECTED", "reason": "持仓不足"}

            revenue = price * quantity
            cost_basis = pos["avg_cost"] * quantity
            pnl = revenue - cost_basis
            self.set_cash(self.get_cash() + revenue)

            new_qty = pos["quantity"] - quantity
            conn.execute("INSERT OR REPLACE INTO positions (ticker, quantity, avg_cost, last_updated) VALUES (?, ?, ?, ?)",
                        (ticker, new_qty, pos["avg_cost"] if new_qty > 0 else 0, now))

            conn.execute("INSERT INTO trades (ticker, side, price, quantity, timestamp, strategy, pnl) VALUES (?, 'SELL', ?, ?, ?, 'live', ?)",
                        (ticker, price, quantity, now, pnl))

        conn.commit()
        conn.close()
        return {"status": "EXECUTED", "action": action, "ticker": ticker, "price": price, "quantity": quantity}


# ══════════��════════════════════════════════════
# 主循环
# ═══════════════════════════════════════════════

def run_pipeline(tickers, strategy="sma", interval_minutes=5):
    """主循环: 获取行情 → 评估 → 执行"""
    print(f"\n{'='*60}")
    print(f"实时交易管道启动")
    print(f"{'='*60}")
    print(f"标的: {', '.join(tickers)}")
    print(f"策略: {strategy}")
    print(f"频率: 每 {interval_minutes} 分钟")
    print(f"初始资金: $100,000")
    print(f"{'='*60}\n")

    engine = StrategyEngine(strategy)
    broker = PaperBroker()

    # 初始化资金
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM config WHERE key='cash'")
    if not c.fetchone():
        conn.execute("INSERT INTO config (key, value) VALUES ('cash', '100000.0')")
    conn.commit()
    conn.close()

    iteration = 0
    try:
        while True:
            iteration += 1
            now = datetime.datetime.now()
            print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S')}] 第 {iteration} 轮")

            # 获取行情
            prices = fetch_prices(tickers)
            if "error" in prices:
                print(f"  行情获取失败: {prices['error']}")
                time.sleep(interval_minutes * 60)
                continue

            # 保存行情
            saved = save_prices(prices)
            print(f"  行情: 保存 {saved} 条")

            # 对每个标的评估
            for ticker in tickers:
                if ticker not in prices:
                    continue

                price = prices[ticker]["close"]
                print(f"\n  {ticker}: ${price:.2f}")

                # 取历史数据
                conn2 = sqlite3.connect(DB_PATH)
                hist_df = pd.read_sql(
                    "SELECT timestamp, open, high, low, close, volume FROM prices WHERE ticker=? ORDER BY timestamp",
                    conn2, params=(ticker,)
                )
                conn2.close()

                if len(hist_df) < 60:
                    print(f"    历史数据不足 ({len(hist_df)}行), 跳过")
                    continue

                hist_df["Close"] = hist_df["close"].astype(float)
                hist_df["Open"] = hist_df["open"].astype(float)

                # 策略评估
                signal = engine.evaluate(ticker, hist_df)
                print(f"    信号: {signal['action']} ({signal['reason']})")

                # 执行交易
                pos = broker.get_position(ticker)
                print(f"    持仓: {pos['quantity']} 股 @ ${pos['avg_cost']:.2f}")
                print(f"    现金: ${broker.get_cash():.2f}")

                if signal["action"] == "BUY" and pos["quantity"] == 0:
                    qty = int(broker.get_cash() * 0.5 / price / 100) * 100  # 50%仓位
                    if qty > 0:
                        result = broker.execute(ticker, "BUY", price, qty)
                        print(f"    执行: {result}")
                elif signal["action"] == "SELL" and pos["quantity"] > 0:
                    result = broker.execute(ticker, "SELL", price, pos["quantity"])
                    print(f"    执行: {result}")

            # 投资组合估值
            total_value = broker.get_cash()
            conn3 = sqlite3.connect(DB_PATH)
            pos_df = pd.read_sql("SELECT ticker, quantity, avg_cost FROM positions WHERE quantity > 0", conn3)
            conn3.close()

            if not pos_df.empty:
                for _, row in pos_df.iterrows():
                    market_price = prices.get(row["ticker"], {}).get("close", row["avg_cost"])
                    total_value += row["quantity"] * market_price

            print(f"\n  投资组合价值: ${total_value:.2f}")
            print(f"  收益率: {(total_value/100000 - 1)*100:.2f}%")

            time.sleep(interval_minutes * 60)

    except KeyboardInterrupt:
        print(f"\n\n管道已停止")
        generate_report()


# ═══════════════════════════════════════════════
# 报告生成
# ═══════════════════════════════════════════════

def generate_report():
    """生成交易报告"""
    conn = sqlite3.connect(DB_PATH)

    print("\n" + "=" * 60)
    print("交易报告")
    print("=" * 60)

    # 持仓
    print("\n当前持仓:")
    pos_df = pd.read_sql("SELECT * FROM positions WHERE quantity > 0", conn)
    if pos_df.empty:
        print("  (空仓)")
    else:
        print(f"  {pos_df.to_string(index=False)}")

    # 交易记录
    print("\n最近交易记录:")
    trades_df = pd.read_sql("SELECT * FROM trades ORDER BY timestamp DESC LIMIT 20", conn)
    if trades_df.empty:
        print("  (无交易)")
    else:
        print(f"  {trades_df.to_string(index=False)}")

    # 统计
    print("\n交易统计:")
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_trades,
            SUM(CASE WHEN side='BUY' THEN 1 ELSE 0 END) as buys,
            SUM(CASE WHEN side='SELL' THEN 1 ELSE 0 END) as sells,
            SUM(pnl) as total_pnl
        FROM trades
    """).fetchone()
    print(f"  总交易: {stats[0]} (买入: {stats[1]}, 卖出: {stats[2]})")
    print(f"  已实现盈亏: ${stats[3]:.2f}" if stats[3] else "  已实现盈亏: $0.00")

    cash = conn.execute("SELECT value FROM config WHERE key='cash'").fetchone()
    print(f"  可用现金: ${float(cash[0]):.2f}" if cash else "")

    conn.close()


# ═══════════════════════════════════════════════
# CLI 入口
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    init_db()

    if "--backfill" in sys.argv:
        tickers = sys.argv[sys.argv.index("--backfill") + 1:]
        if not tickers:
            tickers = ["AAPL", "MSFT", "SPY", "QQQ"]
        backfill_history(tickers)
        sys.exit(0)

    if "--report" in sys.argv:
        generate_report()
        sys.exit(0)

    tickers = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not tickers:
        tickers = ["AAPL", "SPY"]

    run_pipeline(tickers, strategy="sma", interval_minutes=5)
