#!/usr/bin/env python3
"""FinBERT 新闻情绪分析模块 — 替代关键词/LLM 方法

使用 HuggingFace 'ProsusAI/finbert' 模型进行金融领域情感分析，
提供比关键词匹配更准确的 -1~+1 情感评分。

功能:
  1. 从 Yahoo RSS 获取每只股票的最新新闻
  2. 用 FinBERT transformer 模型做新闻情绪打分 (-1 ~ +1)
  3. 检测关键主题（财报、并购、监管、回购、分析师评级等）
  4. 自动下载并缓存模型（首次运行自动下载）
  5. 兼容 fallback：transformers 不可用时回退到关键词方法
  6. 输出结构化的情绪因子，供 ml_stock_picker.py 使用

用法:
  python finbert_sentiment.py --ticker AAPL      # 单只分析
  python finbert_sentiment.py --batch             # 批量所有 watchlist
  python finbert_sentiment.py --update-score      # 生成情绪因子 → 写入 CSV

依赖 (可选):
  pip install transformers torch           # 使用 FinBERT（推荐）
  不安装则自动降级为关键词方法
"""

import importlib
import json
import os
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from xml.etree import ElementTree
from openai import OpenAI

# ═══════════════════════════════════════
# 配置
# ═══════════════════════════════════════
CACHE_DIR = os.path.expanduser("~/.cache/hermes-quant")
MODEL_CACHE_DIR = os.path.join(CACHE_DIR, "finbert_model")
NEWS_TTL_HOURS = 6
MAX_NEWS_PER_TICKER = 10
FINBERT_MODEL_NAME = "ProsusAI/finbert"

# LLM 分析配置
LLM_MODEL = "DeepSeek-V4-Flash"
LLM_TIMEOUT = 60  # 单批次超时
LLM_RETRY = 2     # 失败重试次数

# 情绪关键词词典 — 用于 fallback 和二次验证
BULLISH_KEYWORDS = [
    "beat estimate", "upgrade", "buy rating", "outperform", "strong buy",
    "positive outlook", "raise guidance", "record revenue", "dividend increase",
    "buyback", "partnership", "FDA approval", "breakthrough", "bullish",
    "growth opportunity", "market share gain", "expansion", "acquisition",
    "exceed expectation", "ahead of estimate", "accelerate growth",
    "margin expansion", "cost cutting", "restructuring", "synergy",
    "new contract", "massive order", "backlog", "AI tailwind",
]

BEARISH_KEYWORDS = [
    "downgrade", "sell rating", "underperform", "reduce", "negative outlook",
    "miss estimate", "below estimate", "guidance cut", "revenue decline",
    "lawsuit", "investigation", "regulatory", "fine", "penalty",
    "CEO resignation", "executive departure", "layoff", "restructuring charge",
    "competition", "market share loss", "price war", "inventory glut",
    "supply chain disruption", "tariff", "trade war", "slowdown",
    "profit warning", "loss", "debt", "bankruptcy", "default",
]

TOPIC_PATTERNS = {
    "earnings": r"(earnings|quarterly result|Q[1-4]|fiscal|EPS|revenue|profit)",
    "dividend_buyback": r"(dividend|buyback|share repurchase|shareholder return)",
    "analyst": r"(upgrade|downgrade|rating|analyst|price target|overweight|equal[._ ]weight)",
    "m_and_a": r"(merger|acquisition|takeover|buy|acquire|merge|deal)",
    "regulatory": r"(SEC|FDA|regulatory|approval|investigation|lawsuit|fine)",
    "product": r"(launch|product|new feature|update|release|beta)",
    "partnership": r"(partner|collaboration|alliance|joint venture|teams? with)",
    "AI": r"(AI|artificial intelligence|machine learning|deep learning|LLM|GPT)",
    "macro": r"(interest rate|inflation|Fed|recession|GDP|employment|tariff|trade)",
}

# ═══════════════════════════════════════
# 异常事件检测模式 — 用于事件驱动做空/风险预警
# ═══════════════════════════════════════
EVENT_PATTERNS = {
    "layoff": (
        r"(layoff|lay off|firing|downsize|severance|workforce\s*reduction|"
        r"job\s*cut|大规模裁员|优化人员|人员优化|缩减人员|人事调整|减员)"
    ),
    "profit_warning": (
        r"(profit\s*warning|guidance\s*cut|revenue\s*warning|earnings\s*warning|"
        r"pre.announce|预警|业绩预警|利润预警|盈警|盈利预警)"
    ),
    "executive_change": (
        r"(CEO\s+(resign|step\s*down|depart|fir(e|ed)|quit|outsted|remov(e|ed))|"
        r"CFO\s+(resign|step\s*down|depart|fir(e|ed)|quit)|"
        r"executive\s*departure|高管变动|创始人\s*(离职|辞职)|管理层\s*(地震|变动)|"
        r"财务总监\s*(离职|辞职)|董秘\s*(离职|辞职))"
    ),
    "debt_crisis": (
        r"(debt\s*crisis|default|bankruptcy|insolvent|covenant\s*breach|"
        r"债务违约|债券违约|资不抵债|破产|债务危机|无力偿还|兑付困难)"
    ),
    "short_report": (
        r"(short\s*seller|short\s*report|做空报告|做空机构|"
        r"Citron|Muddy\s*Water|Blue\s*Orca|Hindenburg|浑水|香橼|哥谭)"
    ),
    "regulatory_crackdown": (
        r"(regulatory\s*crackdown|antitrust|反垄断|约谈|立案调查|"
        r"处罚|罚款|suspended\s*trading|stricter\s*regulation|"
        r"监管\s*(约谈|调查|处罚|罚款|立案)|立案\s*调查)"
    ),
    "product_recall": (
        r"(recall|product\s*defect|召回|质量事故|安全事故|"
        r"产品\s*(召回|缺陷|问题)|安全\s*(事故|隐患))"
    ),
    "delisting_risk": (
        r"(delist|退市|ST|\*ST|特别处理|suspended|摘牌|"
        r"暂停\s*上市|终止\s*上市)"
    ),
}

# 事件严重性映射 (0~1, 越小越严重, 用于 event_penalty 做乘数)
EVENT_SEVERITY = {
    "layoff": 0.80,
    "profit_warning": 0.70,
    "executive_change": 0.85,
    "debt_crisis": 0.50,
    "short_report": 0.75,
    "regulatory_crackdown": 0.75,
    "product_recall": 0.85,
    "delisting_risk": 0.30,
}


# ═══════════════════════════════════════
# 1. FinBERT 模型加载 (带 fallback)
# ═══════════════════════════════════════
class FinBERTAnalyzer:
    """FinBERT 情感分析器，带自动下载和 fallback"""

    def __init__(self):
        self._pipeline = None
        self._tokenizer = None
        self._model = None
        self._available = False
        self._load_model()

    def _load_model(self):
        """加载 FinBERT 模型，失败时设置 _available=False"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

            # 尝试从缓存加载，否则自动下载
            self._tokenizer = AutoTokenizer.from_pretrained(
                FINBERT_MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                local_files_only=False,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                FINBERT_MODEL_NAME,
                cache_dir=MODEL_CACHE_DIR,
                local_files_only=False,
            )
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                return_all_scores=True,
                top_k=None,
            )
            self._available = True

            model_path = os.path.join(MODEL_CACHE_DIR, FINBERT_MODEL_NAME.replace("/", "_"))
            print(f"  ✅ FinBERT 模型加载成功 (缓存: {MODEL_CACHE_DIR})", file=sys.stderr)
        except ImportError:
            self._available = False
            print(
                "  ⚠️  transformers 未安装 (pip install transformers torch)，"
                "回退到关键词分析",
                file=sys.stderr,
            )
        except Exception as e:
            self._available = False
            print(
                f"  ⚠️  FinBERT 加载失败 ({e})，回退到关键词分析",
                file=sys.stderr,
            )

    @property
    def is_available(self):
        return self._available

    def analyze(self, text):
        """对单段文本进行 FinBERT 情感分析

        返回: {
            "positive": float,   # 正向概率
            "negative": float,   # 负向概率
            "neutral": float,    # 中性概率
            "score": float,      # 综合得分 -1~+1
        }
        """
        if not self._available or not self._pipeline:
            return None

        try:
            # 截断过长的文本
            max_len = 512
            if len(text) > max_len:
                text = text[:max_len]

            result = self._pipeline(text)
            if result and isinstance(result, list) and len(result) > 0:
                scores = result[0]
                label_map = {}
                for item in scores:
                    label_map[item["label"].lower()] = item["score"]

                pos = label_map.get("positive", 0.0)
                neg = label_map.get("negative", 0.0)
                neu = label_map.get("neutral", 0.0)

                # FinBERT 输出三个概率 (positive, negative, neutral)
                # 综合得分 = positive - negative，范围 -1 ~ +1
                score = pos - neg

                return {
                    "positive": pos,
                    "negative": neg,
                    "neutral": neu,
                    "score": np.clip(score, -1.0, 1.0),
                }
        except Exception as e:
            print(f"  ⚠️  FinBERT 推理失败: {e}", file=sys.stderr)

        return None


# 全局单例
_finbert = None


def get_finbert():
    """获取 FinBERT 分析器单例"""
    global _finbert
    if _finbert is None:
        _finbert = FinBERTAnalyzer()
    return _finbert


# ═══════════════════════════════════════
# 2. 新闻获取 (复用 news_sentiment 逻辑)
# ═══════════════════════════════════════
def fetch_news(ticker, max_items=MAX_NEWS_PER_TICKER):
    """从 Yahoo RSS 获取股票新闻"""
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r.status_code != 200:
            return []

        root = ElementTree.fromstring(r.content)
        items = root.findall(".//item")

        news = []
        for item in items[:max_items]:
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            pubdate = item.findtext("pubDate", "")
            link = item.findtext("link", "")
            news.append({
                "title": title.strip(),
                "summary": re.sub(r"<[^>]+>", "", desc).strip()[:500],
                "date": pubdate,
                "link": link,
            })
        return news
    except Exception as e:
        return []


def fetch_batch_news(tickers):
    """批量获取新闻（有本地缓存）"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, "news_cache.json")

    # 读取缓存
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            # 清理过期缓存
            now = datetime.now()
            for k in list(cache.keys()):
                cached_time = datetime.fromisoformat(cache[k].get("_cached_at", "2000-01-01"))
                if (now - cached_time).total_seconds() > NEWS_TTL_HOURS * 3600:
                    del cache[k]
        except Exception:
            cache = {}

    results = {}
    for t in tickers:
        if t in cache:
            results[t] = cache[t]["news"]
            continue

        news = fetch_news(t)
        results[t] = news

        # 写入缓存
        cache[t] = {
            "news": news,
            "_cached_at": datetime.now().isoformat(),
        }

    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2, default=str)

    return results


# ═══════════════════════════════════════
# 3a. LLM 情感分析（DeepSeek-V4-Flash，每只股票一批调用）
# ═══════════════════════════════════════
def _get_llm_client():
    """从 Hermes 配置获取 API 密钥，初始化 OpenAI 客户端

    多策略读取（按优先级）：
    1. auth.json (Hermes 凭据池)
    2. config.yaml 直接解析
    3. 环境变量
    """
    import json as _json

    base_url = ""
    api_key = ""

    try:
        # 策略1: auth.json — 找 custom:liantong 或 custom:aigw 的 access_token
        auth_path = os.path.expanduser("~/.hermes/auth.json")
        if os.path.exists(auth_path):
            with open(auth_path) as f:
                auth_data = _json.load(f)
            cp = auth_data.get("credential_pool", {})
            for provider_key in ["custom:liantong", "custom:aigw-gzgy2.cucloud.cn:8443"]:
                for cred in cp.get(provider_key, []):
                    ak = cred.get("access_token", "")
                    bu = cred.get("base_url", "")
                    if ak and ak != "***" and bu:
                        api_key = ak
                        base_url = bu
                        break
                if api_key:
                    break
    except Exception:
        pass

    # 策略2: config.yaml 手动解析
    if not api_key or not base_url:
        try:
            config_path = os.path.expanduser("~/.hermes/config.yaml")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    raw = f.read()
                for line in raw.split("\n"):
                    # model 层的 api_key/base_url
                    stripped = line.strip()
                    if stripped.startswith("api_key:") and "auxiliary" not in line and "custom_providers" not in line:
                        v = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                        if v and v != "***":
                            api_key = api_key or v
                    if stripped.startswith("base_url:") and "auxiliary" not in line and "custom_providers" not in line:
                        v = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                        if v:
                            base_url = base_url or v
        except Exception:
            pass

    if not api_key or not base_url:
        return None

    try:
        # OpenAI/httpx 受 ALL_PROXY 环境变量影响，需要临时清除
        for p in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                  "ALL_PROXY", "all_proxy"):
            os.environ.pop(p, None)
        return OpenAI(base_url=base_url, api_key=api_key, timeout=LLM_TIMEOUT)
    except Exception:
        return None


EVENT_LABELS = {
    "layoff": "裁员",
    "profit_warning": "业绩预警",
    "executive_change": "高管变动",
    "debt_crisis": "债务危机",
    "short_report": "做空报告",
    "regulatory_crackdown": "监管整顿",
    "product_recall": "产品召回",
    "delisting_risk": "退市风险",
}


LLM_SYSTEM_PROMPT = """你是一位金融新闻分析师。分析给定股票的多条新闻，返回JSON格式分析结果。

你必须严格返回以下JSON格式（不要加markdown标记）：
{
  "sentiment_score": 浮点数 -1~+1,
  "sentiment_reason": "简短判断原因",
  "events": [
    {
      "event_type": "layoff|profit_warning|executive_change|debt_crisis|short_report|regulatory_crackdown|product_recall|delisting_risk",
      "confidence": 浮点数 0~1,
      "source_news_indices": [新闻序号列表]
    }
  ]
}

判断标准：
- sentiment_score: 综合所有新闻的整体情绪。-1=极度负面, +1=极度正面, 0=中性。
  如果有多条偏多新闻+中性新闻，整体偏多但不会极端。
  如果有多条负面事件（裁员、预警），打分要相应降低。

事件检测（event_type）：
- layoff: 裁员、人员优化、减员
- profit_warning: 业绩预警、利润预警、盈利预警、guidance cut
- executive_change: CEO/CFO/高管离职、管理层变动
- debt_crisis: 债务危机、违约、破产、资不抵债
- short_report: 做空报告、做空机构
- regulatory_crackdown: 监管约谈、反垄断、立案调查、罚款
- product_recall: 产品召回、质量事故
- delisting_risk: 退市风险、暂停上市

注意：
- 只有确实发生了异常事件才标events，日常新闻不要误报
- 同一事件类型在多条新闻中出现，只report一次，confidence取最高那个
- confidence<0.6的事件不要report（避免假阳性）
- 没有异常事件时events返回空列表[]
"""


def llm_sentiment_by_ticker(ticker, news_list):
    """用 DeepSeek-V4-Flash 对单只股票的多条新闻进行批量语义分析

    每只股票一次 API 调用，返回结构化情绪因子。
    失败时返回 None，由调用方 fallback 到关键词方法。
    """
    if not news_list:
        return None

    client = _get_llm_client()
    if client is None:
        return None

    # 构造新闻文本
    news_lines = []
    for i, n in enumerate(news_list):
        news_lines.append(
            f"[{i+1}] 标题: {n['title']}\n    摘要: {n['summary']}"
        )
    news_text = "\n".join(news_lines)

    user_prompt = f"""股票代码: {ticker}
新闻数量: {len(news_list)}

请分析以下所有新闻，返回JSON：

{news_text}"""

    for attempt in range(LLM_RETRY + 1):
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=1024,
            )
            msg = resp.choices[0].message
            # 某些 API 网关把输出放在 reasoning 而不是 content
            content = (msg.content or msg.reasoning or "").strip()

            # 解析 JSON（处理可能的 markdown 包裹）
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            result = json.loads(content)

            # 验证基本结构
            assert isinstance(result.get("sentiment_score"), (int, float))
            assert -1.0 <= result["sentiment_score"] <= 1.0

            # 规范化事件列表
            events = result.get("events", [])
            valid_events = []
            for evt in events:
                et = evt.get("event_type", "")
                conf = float(evt.get("confidence", 0))
                if et in EVENT_LABELS and conf >= 0.6:
                    valid_events.append({
                        "event_type": et,
                        "confidence": conf,
                    })

            return {
                "sentiment_score": round(float(result["sentiment_score"]), 4),
                "sentiment_reason": result.get("sentiment_reason", ""),
                "events": valid_events,
            }

        except Exception as e:
            if attempt < LLM_RETRY:
                time.sleep(1)
                continue
            # 最后一次失败，记录但返回 None
            return None

    return None


# ═══════════════════════════════════════
# 3. 情感分析核心
# ═══════════════════════════════════════
def detect_topics(title, summary):
    """检测新闻涉及的话题"""
    text = (title + " " + summary).lower()
    topics = []
    for topic, pattern in TOPIC_PATTERNS.items():
        if re.search(pattern, text):
            topics.append(topic)
    return topics


def detect_events(title, summary):
    """检测异常事件类型（裁员/业绩预警/高管变动等）

    返回: list[dict], 每项 {"event_type": str, "severity": float}
    """
    text = (title + " " + summary).lower()
    events = []
    for evt, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, text):
            events.append({
                "event_type": evt,
                "severity": EVENT_SEVERITY.get(evt, 1.0),
            })
    return events


def compute_event_discount(events, discount_base=0.85, min_keep=0.10):
    """根据检测到的异常事件计算评分折扣系数

    多条严重事件叠加: 取最严重的那个 × (discount_base)^(后续事件数)
    例如: delisting_risk(0.30) + profit_warning(0.70) = 0.30 × 0.85 = 0.255
    """
    if not events:
        return 1.0
    
    sorted_events = sorted(events, key=lambda e: e["severity"])
    multiplier = 1.0
    for i, evt in enumerate(sorted_events):
        if i == 0:
            multiplier = evt["severity"]
        else:
            multiplier *= discount_base
    return max(multiplier, min_keep)  # 最低保留min_keep%


def keyword_sentiment(title, summary):
    """快速关键词情绪评分（fallback 方法）

    返回: {
        "score": float,       # -1 ~ +1
        "topics": list[str],
        "method": "keyword"
    }
    """
    text = (title + " " + summary).lower()

    bullish_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text)
    bearish_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text)
    topics = detect_topics(title, summary)

    total = bullish_count + bearish_count
    if total > 0:
        score = (bullish_count - bearish_count) / total
    else:
        score = 0.0

    # 纯粹财报分析类新闻轻微偏中性
    if "earnings" in topics and total <= 1:
        score *= 0.5

    return {
        "score": np.clip(score, -1.0, 1.0),
        "topics": topics,
        "method": "keyword",
    }


def finbert_sentiment(title, summary):
    """使用 FinBERT 进行情感分析

    返回: {
        "score": float,       # -1 ~ +1
        "topics": list[str],
        "method": "finbert",
        "prob_pos": float,
        "prob_neg": float,
        "prob_neu": float,
    }
    """
    analyzer = get_finbert()

    if analyzer.is_available:
        text = title + ". " + summary
        result = analyzer.analyze(text)
        if result is not None:
            topics = detect_topics(title, summary)
            return {
                "score": result["score"],
                "topics": topics,
                "method": "finbert",
                "prob_pos": result["positive"],
                "prob_neg": result["negative"],
                "prob_neu": result["neutral"],
            }

    # fallback
    kb = keyword_sentiment(title, summary)
    return kb


def deep_sentiment(ticker, news_list):
    """完整情绪分析（DeepSeek LLM 语义分析 + 关键词兜底）

    优先用 LLM 做每只股票一批的语义分析，失败时回退到关键词方法。

    返回: dict (与 news_sentiment.py 格式一致)
    """
    if not news_list:
        return {
            "sentiment_score": 0.0,
            "sentiment_consistency": 0.5,
            "sentiment_urgency": 0.0,
            "news_count": 0,
            "topics": [],
            "hot_topics": [],
            "recent_headlines": [],
            "recent_direction": 0.0,
            "method": "none",
            "events": [],
            "event_discount": 1.0,
            "event_labels": [],
        }

    # ─── 优先用 LLM 做语义分析 ───
    llm_result = llm_sentiment_by_ticker(ticker, news_list)

    if llm_result is not None:
        # LLM 成功 — 用 LLM 的情绪分 + 事件检测
        llm_score = llm_result["sentiment_score"]
        events = llm_result["events"]

        # 同时用关键词方法获取话题（LLM 不分析话题，关键词快速补充）
        all_topics = []
        for n in news_list:
            topics = detect_topics(n["title"], n["summary"])
            all_topics.extend(topics)

        topic_counts = {}
        for t in all_topics:
            topic_counts[t] = topic_counts.get(t, 0) + 1
        hot_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]

        # 事件折扣
        event_severity = {
            "layoff": 0.80, "profit_warning": 0.70,
            "executive_change": 0.85, "debt_crisis": 0.50,
            "short_report": 0.75, "regulatory_crackdown": 0.75,
            "product_recall": 0.85, "delisting_risk": 0.30,
        }
        unique_events = []
        for evt in events:
            unique_events.append({
                "event_type": evt["event_type"],
                "count": 1,
                "severity": event_severity.get(evt["event_type"], 1.0),
            })
        event_discount = compute_event_discount(unique_events) if unique_events else 1.0

        # 头几条新闻用于展示
        headlines = [n["title"] for n in news_list[:5]]

        return {
            "sentiment_score": llm_score,
            "sentiment_consistency": 0.5,  # LLM 不输出一致性，给默认值
            "sentiment_urgency": min(len(news_list) / 10, 1.0) * (1.0 + abs(llm_score)),
            "news_count": len(news_list),
            "topics": list(set(all_topics)),
            "hot_topics": hot_topics,
            "recent_headlines": headlines,
            "recent_direction": llm_score,  # LLM 已综合判断，方向等同总分
            "method": "llm",
            "events": unique_events,
            "event_discount": round(event_discount, 4),
            "event_labels": [EVENT_LABELS.get(e["event_type"], e["event_type"]) for e in unique_events],
        }

    # ─── LLM 失败，fallback 到 FinBERT/关键词 ───
    scores = []
    all_topics = []
    headlines = []
    all_events = []
    methods_used = set()

    for n in news_list:
        result = finbert_sentiment(n["title"], n["summary"])
        s = result["score"]
        topics = result["topics"]
        methods_used.add(result.get("method", "keyword"))

        events = detect_events(n["title"], n["summary"])
        if events:
            all_events.extend(events)

        scores.append(s)
        all_topics.extend(topics)
        headlines.append(n["title"])

        try:
            pub = datetime.strptime(n["date"][:25], "%a, %d %b %Y %H:%M:%S")
            hours_ago = (datetime.now() - pub).total_seconds() / 3600
            if hours_ago < 48:
                scores[-1] *= 1.5
        except Exception:
            pass

    avg_sentiment = np.clip(np.mean(scores), -1.0, 1.0)

    if len(scores) >= 2:
        consistency = 1.0 - np.std(scores)
    else:
        consistency = 0.5

    recent = scores[:3]
    recent_direction = np.mean(recent)

    topic_counts = {}
    for t in all_topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    hot_topics = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]

    urgency = min(len(news_list) / 10, 1.0) * (1.0 + abs(avg_sentiment))

    event_counts = {}
    for evt in all_events:
        et = evt["event_type"]
        if et not in event_counts:
            event_counts[et] = {"count": 0, "min_severity": 1.0}
        event_counts[et]["count"] += 1
        event_counts[et]["min_severity"] = min(event_counts[et]["min_severity"], evt["severity"])

    unique_events = [
        {"event_type": et, "count": info["count"], "severity": info["min_severity"]}
        for et, info in event_counts.items()
    ]

    event_discount = compute_event_discount(unique_events)

    method = "finbert" if "finbert" in methods_used else "keyword"

    return {
        "sentiment_score": round(avg_sentiment, 4),
        "sentiment_consistency": round(consistency, 4),
        "sentiment_urgency": round(urgency, 4),
        "news_count": len(news_list),
        "topics": list(set(all_topics)),
        "hot_topics": hot_topics,
        "recent_headlines": headlines[:5],
        "recent_direction": round(recent_direction, 4),
        "method": method,
        "events": unique_events,
        "event_discount": round(event_discount, 4),
        "event_labels": [EVENT_LABELS.get(e["event_type"], e["event_type"]) for e in unique_events],
    }


# ═══════════════════════════════════════
# 4. 生成情绪因子
# ═══════════════════════════════════════
def build_sentiment_factors(tickers):
    """生成情绪因子字典 {ticker: {factor1: val, ...}}"""
    news_data = fetch_batch_news(tickers)

    factors = {}
    for t in tickers:
        news = news_data.get(t, [])
        sa = deep_sentiment(t, news)

        factors[t] = {
            "sentiment_score": sa["sentiment_score"],
            "sentiment_urgency": sa["sentiment_urgency"],
            "sentiment_consistency": sa["sentiment_consistency"],
            "news_count": sa["news_count"],
            "hot_topics": ",".join([f"{k}({v})" for k, v in sa["hot_topics"]]),
            "recent_direction": sa["recent_direction"],
            "method": sa["method"],
            "events": sa["events"],
            "event_discount": sa["event_discount"],
            "event_labels": sa["event_labels"],
        }
    return factors


# ═══════════════════════════════════════
# 5. 整合到 ML 评分
# ═══════════════════════════════════════
def sentiment_boost(ml_score_original, sentiment_factors, weight=0.15):
    """将情绪因子融合进 ML 评分（含异常事件折扣）

    Args:
        ml_score_original: ML 模型评分 (0~1)
        sentiment_factors: 情绪因子 dict
        weight: 情绪权重 (默认 15%)

    Returns:
        fused_score: 融合后评分 (0~1)
        adjustment: 调整幅度
        event_adjustment: 事件折扣调整幅度
    """
    s = sentiment_factors

    # 情绪信号强度 (-1 ~ +1)
    sentiment_signal = s.get("sentiment_score", 0) * 0.4
    sentiment_signal += s.get("recent_direction", 0) * 0.3
    sentiment_signal += (s.get("sentiment_urgency", 0) - 0.5) * 0.3

    # 无新闻则中立
    if s.get("news_count", 0) == 0:
        sentiment_signal = 0

    # LLM / FinBERT 方法有更高的置信度权重
    if s.get("method") in ("llm", "finbert") and s.get("news_count", 0) > 0:
        sentiment_signal *= 1.3 if s.get("method") == "llm" else 1.2
        sentiment_signal = np.clip(sentiment_signal, -1.0, 1.0)

    # 有重大负面话题则打折
    topics = s.get("hot_topics", "")
    has_negative_topic = any(
        kw in topics.lower()
        for kw in ["regulatory", "lawsuit", "downgrade"]
    )
    if has_negative_topic and sentiment_signal > 0:
        sentiment_signal *= 0.5

    # 情绪信号从 [-1, 1] 映射到 [0, 1]
    sentiment_norm = (sentiment_signal + 1) / 2

    # 融合: ML (默认 85%) + 情绪 (15%)
    fused = ml_score_original * (1 - weight) + sentiment_norm * weight
    adjustment = fused - ml_score_original

    # ─── 异常事件折扣 (额外惩罚, 不受weight控制) ───
    event_discount = s.get("event_discount", 1.0)
    events = s.get("events", [])
    if events and event_discount < 1.0:
        # 事件折扣: 直接乘以融合后评分
        fused_with_event = fused * event_discount
        event_adjustment = fused_with_event - fused
        return np.clip(fused_with_event, 0, 1), adjustment, event_adjustment

    return np.clip(fused, 0, 1), adjustment, 0.0


# ═══════════════════════════════════════
# 6. CLI
# ═══════════════════════════════════════
def print_sentiment_report(ticker, sentiment):
    """打印情绪报告"""
    s = sentiment
    print(f"\n{'='*70}")
    print(f"  📰 {ticker} 新闻情绪分析 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # 情绪评分
    score = s["sentiment_score"]
    method_label = {"finbert": "FinBERT", "keyword": "关键词", "none": "N/A"}
    method_str = method_label.get(s.get("method", ""), s.get("method", ""))

    if score > 0.3:
        emoji = "🟢"
        label = "偏多"
    elif score < -0.3:
        emoji = "🔴"
        label = "偏空"
    else:
        emoji = "🟡"
        label = "中性"
    print(f"\n  情绪评分: {emoji} {score:+.3f} ({label})")
    print(f"  分析方法: {method_str}")
    print(f"  一致性:   {s['sentiment_consistency']:.2f}")
    print(f"  紧迫度:   {s['sentiment_urgency']:.2f}")

    # FinBERT 详细概率（如果有）
    if s.get("method") == "finbert" and s.get("news_count", 0) > 0:
        # 展示第一条新闻的详细概率（但不在顶层 dict 中，仅显示在报告中）
        pass

    # 新闻数量
    print(f"\n  新闻条数: {s['news_count']}")

    # 热门话题
    if s["hot_topics"]:
        topics_str = ", ".join([f"{t}({c})" for t, c in s["hot_topics"]])
        print(f"  热门话题: {topics_str}")

    # 最近新闻
    if s["recent_headlines"]:
        print(f"\n  最近头条:")
        for h in s["recent_headlines"][:5]:
            print(f"    \u2022 {h[:90]}")


if __name__ == "__main__":
    from ml_stock_picker import US_WATCHLIST

    args = sys.argv[1:]

    if "--ticker" in args:
        idx = args.index("--ticker")
        if idx + 1 < len(args):
            t = args[idx + 1].upper()
            news = fetch_news(t, max_items=15)
            sa = deep_sentiment(t, news)
            print_sentiment_report(t, sa)

    elif "--batch" in args:
        print(f"\n  批量获取 {len(US_WATCHLIST)} 只股票新闻情绪...")
        factors = build_sentiment_factors(US_WATCHLIST)

        # 情绪评分排名
        ranked = sorted(factors.items(), key=lambda x: -x[1]["sentiment_score"])
        # 确定方法标签
        if any(f["method"] == "llm" for _, f in ranked):
            method_label = "DeepSeek-V4-Flash"
        elif any(f["method"] == "finbert" for _, f in ranked):
            method_label = "FinBERT"
        else:
            method_label = "关键词"
        print(f"\n{'='*80}")
        print(f"  情绪评分排名 ({method_label})")
        print(f"{'='*80}")
        print(f" {'#':>3} {'代码':>6} {'情绪分':>8} {'新闻数':>6} {'方向':>7} {'紧迫度':>8} {'方法':>8} {'热门话题'}")
        for i, (t, f) in enumerate(ranked[:15]):
            dir_s = "🟢" if f["recent_direction"] > 0.2 else ("🔴" if f["recent_direction"] < -0.2 else "🟡")
            method_tag = f["method"][:6]
            print(f" {i+1:>3} {t:>6} {f['sentiment_score']:>+7.3f} {f['news_count']:>5} "
                  f"{dir_s}{f['recent_direction']:>+5.2f} {f['sentiment_urgency']:>7.3f} "
                  f"{method_tag:>8} {f['hot_topics'][:40]}")

        # 保存情绪因子到缓存
        os.makedirs(CACHE_DIR, exist_ok=True)
        cache_file = os.path.join(CACHE_DIR, "sentiment_factors.json")
        with open(cache_file, "w") as f:
            json.dump(factors, f, indent=2)
        print(f"\n  已保存情绪因子到 {cache_file}")

    elif "--update-score" in args:
        # 整合进 ML 选股系统
        from ml_stock_picker import run_picking, print_results, US_WATCHLIST

        print(f"\n  步骤1/3: 运行 ML 选股系统...")
        results = run_picking(period="2y")
        if not results:
            print("  ❌ ML 选股失败")
            sys.exit(1)

        print(f"\n  步骤2/3: 获取新闻情绪 (FinBERT)...")
        factors = build_sentiment_factors(US_WATCHLIST)

        # 统计使用方法
        finbert_count = sum(1 for f in factors.values() if f.get("method") == "finbert")
        keyword_count = sum(1 for f in factors.values() if f.get("method") == "keyword")
        print(f"  FinBERT: {finbert_count} 只 | 关键词 fallback: {keyword_count} 只")

        print(f"\n  步骤3/3: 融合情绪因子...")
        for r in results:
            t = r["ticker"]
            sf = factors.get(t, {})
            if sf:
                original = r["score"]
                fused, adj, evt_adj = sentiment_boost(original, sf)
                r["score"] = fused
                r["sentiment_adj"] = adj
                r["event_adj"] = evt_adj
                r["sentiment_factors"] = sf
            else:
                r["sentiment_adj"] = 0
                r["event_adj"] = 0
                r["sentiment_factors"] = {"sentiment_score": 0}

        # 重排序
        results.sort(key=lambda x: x["score"], reverse=True)
        top = print_results(results, top_n=20)

        # 打印情绪调整详情
        print(f"\n{'='*120}")
        print(f"  情绪因子调整详情")
        print(f"{'='*120}")
        print(f" {'代码':>6} {'原始ML':>8} {'情绪分':>8} {'调整':>8} {'融合后':>8} {'新闻':>5} {'方向':>6} {'方法':>7} {'话题'}")
        print(f" {'-'*80}")
        for r in results[:10]:
            sf = r.get("sentiment_factors", {})
            adj = r.get("sentiment_adj", 0)
            orig = r["score"] - adj
            dir_s = "🟢" if sf.get("recent_direction", 0) > 0.2 else ("🔴" if sf.get("recent_direction", 0) < -0.2 else "🟡")
            method_tag = sf.get("method", "")[:6]
            print(f" {r['ticker']:>6} {orig:>8.4f} {sf.get('sentiment_score',0):>+7.3f} "
                  f"{adj:>+7.4f} {r['score']:>8.4f} "
                  f"{sf.get('news_count',0):>4} {dir_s} {sf.get('recent_direction',0):>+5.2f} "
                  f"{method_tag:>7} {sf.get('hot_topics','')[:25]}")

        # 保存为 CSV (merged sentiment CSV)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        csv_path = f"ml_fused_{timestamp}.csv"
        rows = []
        for r in results[:20]:
            sf = r.get("sentiment_factors", {})
            rows.append({
                "ticker": r["ticker"],
                "ml_raw": round(r["score"] - r.get("sentiment_adj", 0), 4),
                "sentiment_score": sf.get("sentiment_score", 0),
                "sentiment_adj": round(r.get("sentiment_adj", 0), 4),
                "fused_score": round(r["score"], 4),
                "news_count": sf.get("news_count", 0),
                "method": sf.get("method", ""),
                "hot_topics": sf.get("hot_topics", ""),
            })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n  已保存: {csv_path}")

        # 同时保存一份单独的 FinBERT 情绪因子 CSV
        factors_csv = f"finbert_sentiment_{timestamp}.csv"
        factor_rows = []
        for t in US_WATCHLIST:
            sf = factors.get(t, {"sentiment_score": 0, "sentiment_urgency": 0,
                                  "sentiment_consistency": 0, "news_count": 0,
                                  "recent_direction": 0, "method": "none"})
            factor_rows.append({
                "ticker": t,
                "sentiment_score": sf.get("sentiment_score", 0),
                "sentiment_urgency": sf.get("sentiment_urgency", 0),
                "sentiment_consistency": sf.get("sentiment_consistency", 0),
                "news_count": sf.get("news_count", 0),
                "recent_direction": sf.get("recent_direction", 0),
                "method": sf.get("method", "none"),
                "hot_topics": sf.get("hot_topics", ""),
            })
        df_factors = pd.DataFrame(factor_rows)
        df_factors.to_csv(factors_csv, index=False, encoding="utf-8-sig")
        print(f"  已保存: {factors_csv}")

        print(f"\n  ✅ 情绪融合完成! (方法: FinBERT + 关键词混合)")

    else:
        print(__doc__)
