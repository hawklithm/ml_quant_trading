# 量化系统改造实施计划

## 0. 目标、范围与执行规则

本计划针对当前仓库的 ML 选股、因子评分、情绪融合、组合优化、回测和纸面交易链路。目标是先修复结果可信度，再建立 point-in-time 回测，最后完成工程化重构。未经本计划明确授权，不新增模型、不改变交易逻辑、不把纸面交易接入真实券商。

所有任务必须遵守：

1. 每个阶段单独提交，提交前运行语法检查和测试。
2. 所有收益、准确率和排名指标必须标明预测周期、数据截止日期、交易成本假设和基准。
3. 外部数据失败时必须返回可识别的失败状态，不得静默使用未知日期的缓存。
4. 新旧结果必须并行保存，完成对照后才能删除旧逻辑。
5. “脚本能运行”不是完成标准，必须满足本文件的验收条件。

## 1. 目标架构

    数据源 → 原始数据快照 → point-in-time 特征 → 标签
           → Purged Walk-forward 训练/预测
           → 信号与概率校准 → 组合构建
           → 含成本回测 → 纸面执行 → 归档与监控

建议最终目录：

    quant/
      data/          # 数据源、缓存、快照、质量检查
      features/      # 特征和标签
      models/        # 训练、预测、校准
      signals/       # ML、因子、情绪、市场状态融合
      portfolio/     # 权重、约束、再平衡
      backtest/      # 事件驱动回测和指标
      execution/     # 纸面 broker 与订单状态
    tests/
    configs/

迁移完成前，ml_optimized_picker_v5.py 仍是唯一主模型入口；其他版本只作为对照，不再继续增加功能。

---

## 2. 阶段一：修复正确性与数据口径

### T1.1 建立可复现环境与基线

涉及：仓库根目录、README.md。

任务：

1. 从所有 Python import 中整理直接依赖，创建 requirements.txt，固定 Python 3.12 兼容版本。
2. 创建 tests/，加入最小测试运行说明。
3. 修复 README.md 和 Python 文档中的 UTF-8 乱码，统一使用 UTF-8。
4. 记录当前基线：代码版本、配置哈希、数据截止日期、可用 ticker 数、每只股票样本数。

验收：

- 新环境执行 pip install -r requirements.txt 成功。
- python -m compileall -q . 成功。
- pytest -q 可运行，即使暂无完整模型测试也必须有基础测试。
- 基线报告可由命令重复生成，不能依赖人工复制终端输出。

### T1.2 修复分类模型样本内预测

涉及：ml_optimized_picker_v5.py，重点是 train_model_walk_forward_v5() 和 score_stock_v5()。

任务：

1. 将分类结果结构改为保存 model、scaler、oof_pred、oof_proba、accuracy、fold_metrics。
2. 全量训练模型只用于最新样本预测，不得把全量样本内预测作为方向信号。
3. score_stock_v5() 必须对 X.iloc[-1:] 调用最终分类模型，生成当前方向和概率。
4. 方向准确率只能使用 OOF 预测计算；未产生 OOF 预测的行不得进入分母。
5. 如果分类模型不可用，明确返回 direction_source="momentum_fallback"，并记录原因。
6. 删除使用 cls_model[-3:] 推断方向的逻辑。

验收：

- 单元测试证明最新方向来自 model.predict(X_latest)，不是训练预测数组。
- 随机打乱训练标签后，分类准确率应接近基准，而不是稳定保持高准确率。
- 输出中包含 classification_oos_accuracy 和 direction_probability。

### T1.3 修复 Purged Walk-forward

涉及：ml_optimized_picker_v5.py，新增可测试的时间切分器函数。

任务：

1. 将 forecast_horizon_short、forecast_horizon_long 明确传入切分器。
2. 训练集与测试集之间至少保留 max_horizon 个交易日 gap。
3. 保证每一折训练标签的结束日期早于测试特征开始日期。
4. 每折保存 train_start、train_end、test_start、test_end。
5. 回归和分类统一使用相同的时间切分规则。
6. 禁止随机 train_test_split 用于时间序列模型。

验收：

- 测试检查任意 fold 都满足 max(train_label_end) < test_feature_start。
- 输出每折样本量和日期范围。
- OOS 指标与当前版本并行输出，不能只保留新指标。

### T1.4 统一目标、标签与评估周期

涉及：v5_config.json、ml_optimized_picker_v5.py、cron_market_job.py、data_archiver.py、cross_validate_picker.py。

任务：

1. 将配置改为明确的 prediction_horizons: [5, 21]，禁止使用含义不清的 actual_5d。
2. 每个 horizon 独立生成 target、prediction、realized return 和 direction。
3. 若主模型预测 21 日，则盘后复核必须按 21 个交易日复核；5 日指标只能作为辅助指标。
4. 归档字段统一为 signal_date、data_asof、horizon_days、target_type、realized_return、benchmark_return、excess_return。
5. 删除或重命名 actual_5d，避免 21 日模型继续写入该字段。

验收：

- 任意一条预测记录可以根据字段明确判断预测日、数据截止日和兑现日。
- 5 日和 21 日结果不会混入同一准确率统计。
- 盘后复核、CSV 归档和报告使用同一套字段定义。

### T1.5 修复市场基准选择

涉及：ml_optimized_picker_v5.py、v5_config.json。

任务：

1. 新增 benchmark_by_market 配置：US=SPY、HK=^HSI。
2. 特征、目标和复盘统一调用 get_benchmark_series(market)。
3. score_stock_v5() 显式接收 market，不能只通过 ticker 后缀推断。
4. HK 目标使用 HSI 超额收益，US 目标使用 SPY 超额收益。
5. benchmark 缺失时记录 benchmark_status="missing"，不得静默切换到其他市场基准。

验收：

- HK 测试样本的 target 计算使用 HSI 日期序列。
- US/HK 各有一个固定的基准测试用例。
- 报告中展示 benchmark 名称和数据最后日期。

### T1.6 建立 point-in-time 数据和缓存规则

涉及：ml_optimized_picker_v5.py、tencent_data.py、data_archiver.py。

任务：

1. 每份缓存增加 source、downloaded_at、data_asof、symbol、adjustment 元数据。
2. 预测时只允许使用 data_asof <= signal_time 的数据。
3. 盘前和盘后任务分别定义数据截止时间；盘后不得直接复用盘前旧缓存作为收盘数据。
4. 缓存刷新失败时返回 stale=True 和缓存日期，报告必须明确提示。
5. 归档原始快照时按 market/ticker/signal_date 保存，不能只保存最后 60 行而丢失版本信息。
6. 对 OHLCV 做字段、时区、重复日期、空值、异常价格检查。

验收：

- 测试证明未来日期数据不能进入历史 signal。
- 盘后使用的最后一根 K 线日期正确。
- stale 缓存会导致任务告警，不能被当作正常数据处理。

---

## 3. 阶段二：重建验证和回测体系

### T2.1 建立历史股票池和样本快照

涉及：新增 quant/data/universe.py、data_archiver.py、configs/。

任务：

1. 每个交易日保存当日可选股票池及纳入原因。
2. 记录加入、剔除、停牌、退市和代码变更日期。
3. 回测必须使用当时的股票池，禁止直接使用今天的固定 watchlist。
4. 明确 US/HK 股票代码映射和复权规则。

验收：

- 任意历史日期可以还原当日股票池。
- 回测不会出现使用未来才加入股票池的标的。

### T2.2 实现事件驱动回测

涉及：新增 quant/backtest/engine.py、quant/backtest/costs.py、quant/backtest/metrics.py。

任务：

1. 每个 signal date 重新生成当时可见的特征和模型预测。
2. 按 signal_time 之后的第一个可交易价格成交。
3. 支持持有期、调仓频率、最大持仓数和信号阈值配置。
4. 实现手续费、滑点、印花税/交易费、汇率和最小交易单位。
5. 记录订单、成交、持仓、现金、净值和组合暴露。
6. 支持等权、分数加权、风险平价三种基线。
7. 加入 SPY、HSI 和 buy-and-hold 基准。

验收：

- 生成完整净值曲线和交易明细。
- 能复现固定日期区间的结果。
- 交易成本设为 0 和非 0 时结果明显不同。
- 使用未来价格或未来股票池时，测试必须失败。

### T2.3 建立标准指标与统计显著性报告

任务：

1. 计算 CAGR、年化波动、Sharpe、Sortino、最大回撤、Calmar、换手率和胜率。
2. 计算 Rank IC、ICIR、Top-bottom spread 和分位数组合收益。
3. 按年份、市场、行业、波动率状态和牛熊状态分组。
4. 报告样本量、置信区间和 bootstrap 结果。
5. 单独报告交易成本前后指标。

验收：

- 每份报告带有配置哈希、数据区间、模型版本和生成时间。
- 不再使用单一准确率判断策略有效性。
- 与基准、等权和随机排序策略同时比较。

### T2.4 对情绪因子进行独立回测

涉及：finbert_sentiment.py、news_sentiment_v2.py、新增新闻归档模块。

任务：

1. 归档新闻的发布时间、抓取时间、来源、标题和去重 ID。
2. 只允许使用 signal time 之前已发布且已抓取的新闻。
3. 先单独测试情绪因子，再与 ML 因子做增量贡献分析。
4. 比较无情绪、固定权重情绪、动态权重情绪三组结果。
5. 统计情绪缺失、LLM 失败和关键词 fallback 比例。

验收：

- 可以重放任意历史日期的情绪输入。
- 情绪模块必须证明对 Rank IC 或组合收益有稳定增量，否则默认关闭。

### T2.5 校准 ML 分数和组合优化输入

涉及：portfolio_optimizer.py、新增 quant/models/calibration.py。

任务：

1. 禁止直接使用 score * 0.25 - 0.05 作为预期收益。
2. 使用 OOS 分数分桶，统计每个分数区间的未来收益和置信区间。
3. 使用训练集拟合、验证集选择、测试集最终评估的校准流程。
4. Kelly 默认改为 fractional Kelly，初始系数 0.25。
5. 增加单票权重、行业权重、总杠杆、现金比例和最大回撤约束。

验收：

- 分数越高的桶必须有可验证的收益单调性，否则禁止用于 Kelly。
- Kelly、风险平价、等权三种组合均能输出权重和约束检查结果。
- 所有权重非 NaN，权重和、杠杆和空头约束符合配置。

---

## 4. 阶段三：工程化重构与执行闭环

### T3.1 合并重复版本，建立唯一主链路

任务：

1. 以 ml_optimized_picker_v5.py 为基准拆分公共模块。
2. v1/v2/v4 只保留为 Git tag 或迁移记录，不再作为运行入口。
3. 统一 ticker、DataFrame 列名、时间时区和错误对象。
4. 所有 CLI 使用 argparse，不再手工解析 sys.argv。
5. 删除 _debug_*、*_tmp.py 和已验证无用的临时入口，删除前先确认没有 cron 引用。

验收：

- 主流程只需要一条命令即可完成数据、预测、组合和回测。
- 没有两个模块实现同名但行为不同的核心函数。
- 旧版本行为有迁移说明或对照测试。

### T3.2 实现统一数据质量和错误处理

任务：

1. 删除核心流程中的裸 except: 和静默 pass。
2. 定义 DataUnavailable、StaleData、ModelTrainingError、ExecutionRejected 等异常。
3. 每个 ticker 返回 status、stage、reason 和 data_asof。
4. 将 warnings.filterwarnings("ignore") 改为按模块、按警告类型处理。
5. 使用结构化日志记录任务 ID、市场、ticker、阶段、耗时和错误堆栈。

验收：

- 任意失败 ticker 可以在报告中定位具体阶段。
- 数据错误不会生成看似正常的 score。
- 日志既包含成功数量，也包含失败原因分布。

### T3.3 统一 ML 信号与纸面执行

涉及：live_pipeline.py、新增 quant/execution/paper_broker.py。

任务：

1. 明确主策略：ML 负责选股，组合模块负责权重，执行模块负责订单；SMA/RSI 只作为独立基线。
2. 纸面 broker 支持 NEW、REJECTED、PARTIAL、FILLED、CANCELLED 订单状态。
3. 一个交易操作使用同一 SQLite 事务完成现金、持仓和 trade 写入。
4. 增加手续费、滑点、成交延迟、最小手数和现金检查。
5. 为每个订单保存 signal_id、model_version、signal_time、order_time、fill_time。
6. 纸面执行默认只允许模拟，不提供真实券商凭据读取逻辑。

验收：

- 交易失败不会修改现金或持仓。
- 重复执行同一个 signal 不会重复下单。
- 纸面执行的净值可以与回测执行结果按相同成本模型对账。

### T3.4 增加自动化测试和 CI 检查

任务：

1. 为数据标准化、基准选择、目标生成、purged split、指标计算和权重约束编写单元测试。
2. 添加合成行情测试，禁止测试依赖实时网络。
3. 添加数据泄漏测试：未来日期、未来标签、未来股票池必须被拒绝。
4. CI 至少执行：JSON 校验、compileall、pytest、ruff/格式检查。
5. 为每日 cron 增加 dry-run 模式，dry-run 不修改配置、不写交易、不更新惩罚记录。

验收：

- CI 在干净环境通过。
- 网络不可用时单元测试仍可完成。
- 每个阶段的回归测试能阻止已修复的泄漏再次出现。

### T3.5 限制自动调参和配置漂移

涉及：cron_market_job.py、v5_config.json。

任务：

1. 自动调参只生成建议，不直接覆盖正式配置。
2. 配置修改写入版本化变更记录，包含旧值、新值、原因、指标窗口和操作者。
3. 配置必须通过范围校验，例如权重非负、权重和为 1、阈值有序。
4. 增加 --dry-run 和 --apply-config 两种模式。
5. 正式配置变更必须经过回测对照，确认没有使测试集指标恶化。

验收：

- cron 默认不会修改 v5_config.json。
- 无效配置会在启动前拒绝。
- 任意历史运行可以还原对应配置版本。

---

## 5. 推荐执行顺序与交付物

执行顺序：

    T1.1 → T1.2/T1.3/T1.4/T1.5 → T1.6
         → T2.1 → T2.2 → T2.3
         → T2.4/T2.5
         → T3.1 → T3.2/T3.3 → T3.4 → T3.5

每个阶段必须交付：

- 代码变更和测试。
- 配置变更说明。
- 可复现命令。
- 指标前后对比。
- 已知限制和未完成任务。

阶段完成门槛：

- 阶段一：无已知目标周期错配、基准错配和分类样本内预测。
- 阶段二：存在可重放的含成本 point-in-time 回测，并能与基准比较。
- 阶段三：主链路统一、失败可观测、纸面执行可对账，CI 和回归测试通过。

在阶段二完成前，不得依据模型分数、情绪信号或 Kelly 权重进行真实资金交易。

## 6. 当前实施进度

本轮已完成：

- T1.1：增加 requirements.txt、基础 pytest 测试，并修复配置 UTF-8 读取和网络 smoke test 的收集问题。
- T1.2：分类模型改为保存最终模型和 scaler，最新方向使用最新特征预测；分类准确率使用 OOS 预测。
- T1.3：回归和分类切换到带 21 日 purge gap 的 expanding time split。
- T1.4：将回归目标设为 5 日、分类方向设为 21 日，并增加目标周期、信号日期和数据截止日期字段。
- T1.5：增加 US/SPY、HK/HSI 基准配置和测试，港股不再静默使用 SPY。
- T1.6：增加缓存年龄、数据截止日期、基准状态和 stale 标记，并写入预测归档。
- T2.2 基础设施：新增 quant/backtest 下的成本模型、绩效指标和长仓事件驱动回测原型，已使用合成行情测试。
- T2.1 基础设施：新增 quant/data/universe.py，支持按交易日保存和还原历史股票池；cron 盘前任务会归档当前 watchlist 快照。
- T2.3 基础设施：新增 Rank IC、分位数 Top-bottom spread、换手率和扩展绩效指标。
- T2.4 基础设施：新增新闻归档与 signal-time 过滤模块，未来接入新闻 provider 时必须先经过该过滤层。
- T2.5 基础设施：新增 OOS 分数分桶和单调性校准模块；cross_validate_picker.py 通过 `--calibration-csv` 显式接入校准数据，缺少校准数据时强制使用 risk parity，不直接把 ML 分数映射成收益。
- T2.1/T2.2 接入：新增 run_point_in_time_backtest() 和 v5 replay signal generator；模型回调只能接收 as_of 之前的历史价格和当日股票池。
- T2.5 接入：portfolio_optimizer.py 支持 OOS 分数校准输入、单票权重上限和默认禁空约束。
- T2.4 接入：finbert_sentiment.py 在传入 signal_time 时先归档新闻，再过滤未来发布时间；异常时间戳保留实时过滤 fallback。
- T3.3：PaperBroker 增加订单 ID、signal ID、model version、状态字段、幂等检查和单事务现金/持仓/交易写入。
- T3.4/T3.5：新增 GitHub Actions 基础 CI 和 cron --dry-run；自动配置修改仍需显式 --apply-config。
- T3.5 安全默认值：cron 默认只记录自动调参建议，必须显式传入 --apply-config 才能修改配置。

仍需后续处理的外部事项：在稳定数据源可用时补充真实历史数据重放报告。真实券商接入仍明确禁止。

### 本轮收口记录

- 已新增 `quant/execution/reconcile.py`，可按日期对比 PaperBroker 与回测净值，并按成交日、标的、方向、数量、价格对比成交明细；漂移会返回 `passed=false` 和差异量。
- 已新增 `scripts/run_historical_report.py`，支持 `--prices` 离线 CSV 和 `--download` yfinance 两种输入，输出 JSON、净值 CSV、成交 CSV；报告只能使用 point-in-time signal generator。
- 已新增 `quant/core/errors.py` 统一定义数据不可用、数据过期、模型训练失败和执行拒绝异常；后续核心脚本迁移时必须保留 ticker、stage、reason、data_asof。
- 已新增 `quant/ENTRYPOINTS.md`，明确 v5、cron、回测报告为维护入口，并标注重复旧脚本不得继续扩展。本轮未直接删除旧文件，以避免未知外部 cron/Notebook 引用失效。
- 已验证：`pytest -q`、`python -m compileall -q .`、`python -m json.tool v5_config.json`。真实 Yahoo 下载尝试因服务限流未生成报告；应使用 CSV 输入重跑，不能把限流结果视为有效回测结果。
- 本轮继续完成：历史报告增加等权、固定随机、SPY/HSI 买入持有基准；新增 `scripts/reconcile_paper.py` 接通 PaperBroker SQLite；核心维护链路的裸 `except:` 已替换为限定异常并输出原因；删除 5 个仓库内无引用的调试/临时入口。
- 当前唯一外部依赖项：需要稳定行情源或用户提供历史 CSV 后，运行报告命令并保存真实数据结果，不能用合成数据替代生产结论。

### 第二轮质量增强

- 回测入口新增行情和信号 fail-fast 校验：拒绝重复 date/ticker、非有限价格/分数、空标的和非正收盘价。
- 修复报告命令交易成本参数名错误，统一使用 commission/slippage/sell tax bps；回测目标仓位预留买入成本，现金不应因手续费变为负数。
- 历史报告增加 UTC 生成时间、配置 SHA-256、信号日期数量和成本模型，增强结果可追溯性。
- 新增校验测试，并更新 README 的维护入口、历史报告和 PaperBroker 对账命令。
- point-in-time 回测现在强制 signal generator 返回 `signal_date == as_of`，并拒绝不属于当日股票池的 ticker，避免回调内部产生隐性前视偏差。
- 已删除旧 v1/v4 picker、旧 news sentiment、旧 ml_deep_scan 和 debug 运行入口，迁移说明保留在 `legacy/README.md`；维护入口统一到 v5、FinBERT、quant 和 cron。
- `cross_validate_picker.py` 已改为 argparse，支持 `--calibration-csv`；配置读取统一使用 UTF-8，并通过 `--help` 验证。
- PaperBroker 现在复用统一交易成本模型，手续费会进入现金、持仓平均成本和卖出 PnL；校准映射发现非单调 OOS 桶收益时会拒绝进入 Kelly。
- cron_market_job.py 已修复方向阈值恢复漂移、港股 SPY 基准错配、pre/post 情绪重复融合、复盘基准归因字段缺失和重复加载 v5 模块问题，并增加回归测试。
