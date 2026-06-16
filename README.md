# 量化交易工具箱

## 环境
`source ~/projects/quant-trading/.venv/bin/activate`

## 项目文件

```
~/projects/quant-trading/
├── .venv/                   # Python 3.12 虚拟环境
│
├── hello_quant.py           # [入门] 拉数据+统计+简单回测
├── engine_compare.py        # [1] 回测引擎对比 pandas vs Backtrader
├── strategies.py            # [2] 4种策略: 均值回归/MACD/配对/ADX
├── akshare_demo.py          # [3] A股数据源 AkShare 演示
├── ml_factor_model.py       # [4] ML随机森林多因子预测模型
├── live_pipeline.py         # [5] 实时行情+模拟交易管道
│
├── double_ma.py             # 双均线参数扫描+热力图
├── explore.py               # 多标的统计概览
├── bt_demo.py               # Backtrader 框架演示
│
├── live_trading.db          # 管道数据库
├── *.png                    # 策略图表输出
└── README.md
```

## 已安装的库
numpy, pandas, matplotlib, scipy, statsmodels, scikit-learn, plotly,
yfinance, backtrader, TA-Lib, vectorbt, akshare

## 快速使用

```bash
source ~/projects/quant-trading/.venv/bin/activate

# 入门
python hello_quant.py AAPL

# [1] 回测引擎对比
python engine_compare.py

# [2] 4种策略一起看
python strategies.py SPY

# [3] A股数据
python akshare_demo.py

# [4] ML多因子模型
python ml_factor_model.py

# [5] 实时管道（先回填历史数据）
python live_pipeline.py --backfill AAPL SPY
python live_pipeline.py                              # 启动循环
python live_pipeline.py --report                     # 查看交易记录
```

## 5个方向总览

### 1. 回测引擎对比
| 维度 | pandas向量化 | Backtrader |
|------|-------------|------------|
| 速度 | ~3ms/回测 | ~250ms/回测 |
| 代码量 | ~4行 | ~40行 |
| 灵活度 | 简单策略 | 复杂策略 |
| 实盘接入 | ✗ | ✓ (IB API) |
| 推荐 | 策略探索/参数扫描 | 最终验证/实盘 |

**推荐工作流**: pandas 做策略探索 → Backtrader 做最终验证

### 2. 4种策略
- **均值回归(布林带)**: 赚价格回归的钱, 适合震荡市
- **趋势跟踪(MACD)**: 赚趋势延续的钱, 适合趋势市
- **配对交易**: 市场中性, 赚价差回归的钱
- **ADX趋势强度**: 判断是否有趋势, 配合其他策略做过滤

### 3. A股数据 (AkShare)
- 免费, 无需注册, 覆盖全市场
- 沪深300有5917日历史数据 vs yfinance仅20日
- 支持: 个股/指数/ETF/资金流向/分笔交易

### 4. ML多因子模型
- 18个特征: 动量+波动率+均价偏离+RSI+成交量
- RandomForest (200棵树, 测试R²=0.14)
- 策略: SPY 5年夏普1.96 vs 买入持有0.96
- 关键: 特征工程 > 模型选择

### 5. 实时行情管道
- SQLite 存储行情+交易+持仓
- 可插拔策略引擎 (SMA/RSI)
- 模拟执行 (资金管理+风控)
- 报告生成 (交易统计+持仓)
- 运行: 命令行启动, Ctrl+C 停止

## 重要提醒
- 夏普>2基本是过拟合
- 回测≠实盘, 样本外验证必不可少
- A股推荐用 AkShare 替代 yfinance
- 实时管道是模拟的, 不要用于实盘
