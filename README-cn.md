# lethes

**约束驱动的 LLM 上下文管理库** — 一个把对话历史当作优化问题来处理的消息编排引擎。

> *lethes* 取自希腊神话中的遗忘之河。它负责决定什么该被记住、什么该被压缩、什么该被遗忘。

---

## 核心理念

传统的上下文管理方案只是简单的滑动窗口——超出限制就从头截断。`lethes` 的思路完全不同：

| 传统方案 | lethes |
|---|---|
| 能放多少放多少 | 在约束下求最优解 |
| 消息地位平等 | 每条消息都有动态权重 |
| 截断 = 删除 | 截断 = 选择性压缩或删除 |
| 固定策略 | 算法可插拔 |
| 无用户控制 | 对话内 flag 精确控制 |

---

## 特性

- **动态权重** — 消息权重不是固定数字，而是根据当前输入实时计算的相关性评分
- **约束驱动** — 以词元预算、费用预算为约束，最大化上下文信息量
- **算法可插拔** — 贪心、近期偏置、依赖感知、前缀缓存优化，或自定义
- **用户可控** — 在对话中直接用 `!flag` 语法精确控制编排行为
- **多级摘要** — Turn 级、段落级、对话级三层压缩，而非简单删除
- **KV 缓存优化** — 追踪历史发送序列，最大化前缀命中，降低 API 成本
- **工具调用感知** — 完整支持 OpenAI tool calls 格式；自动绑定 assistant/tool 消息对，确保截断后序列仍合法
- **多模态就绪** — 原生支持 `image_url` 内容块（URL 及 base64），编排时自动提取文本，图像块透明透传
- **无厂商绑定** — 摘要和嵌入通过标准 OpenAI 兼容接口调用，支持 Ollama、vLLM、OpenRouter 等
- **结构化可观测** — 基于 structlog 的 JSON 事件；配合 [lethes-observer](../lethes_observer) 可获得实时 Web 仪表盘

---

## 安装

```bash
pip install lethes                  # 基础安装
pip install lethes[redis]           # 启用 Redis 缓存
pip install lethes[bm25]            # 启用 BM25 关键词相关性
pip install lethes[langchain]       # LangChain 适配器
pip install lethes[all]             # 安装所有可选依赖
```

**最低要求：** Python 3.11+，tiktoken，pydantic ≥ 2，httpx ≥ 0.27，structlog ≥ 24

---

## 快速上手

### 最简用法

```python
from lethes import ContextOrchestrator, Conversation, TokenBudget

orchestrator = ContextOrchestrator(
    budget=TokenBudget(max_tokens=8000),
)

result = await orchestrator.process(
    Conversation.from_openai_messages(raw_messages)
)

ready_messages = result.conversation.to_openai_messages()
# → 标准 OpenAI 格式，可直接传给任何 LLM
```

`raw_messages` 可以包含任意 OpenAI 消息格式：

```python
messages = [
    # 普通文本
    {"role": "user", "content": "今天天气怎样？"},

    # 工具调用（assistant → content 为 null）
    {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_abc", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}
    }]},

    # 工具返回结果
    {"role": "tool", "tool_call_id": "call_abc",
     "name": "get_weather", "content": "22°C，多云"},

    # 多模态（图片 + 文字）
    {"role": "user", "content": [
        {"type": "text", "text": "这张图里有什么？"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64,iVBOR...",
            "detail": "high"
        }},
    ]},
]
```

### 完整配置

```python
from lethes import ContextOrchestrator, Conversation, TokenBudget
from lethes.algorithms import RecencyBiasedAlgorithm, DependencyAwareAlgorithm
from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
from lethes.weighting.embedding import EmbeddingSimilarityStrategy
from lethes.summarizers import LLMSummarizer, TurnSummarizer
from lethes.cache import RedisCache
from lethes.engine import ConstraintSet

# 动态权重：70% 嵌入相似度 + 30% 智能关键词/相干
weighting = CompositeWeightStrategy([
    (EmbeddingSimilarityStrategy(
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="text-embedding-3-small",
        cache=RedisCache.from_url("redis://localhost:6379/0"),
    ), 0.7),
    (SmartWeightingStrategy(), 0.3),
])

# 摘要后端：任意 OpenAI 兼容接口
summarizer = TurnSummarizer(
    backend=LLMSummarizer(
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
    ),
    cache=RedisCache.from_url("redis://localhost:6379/0"),
)

orchestrator = ContextOrchestrator(
    budget=TokenBudget(max_tokens=12000),
    algorithm=DependencyAwareAlgorithm(
        inner=RecencyBiasedAlgorithm(recency_factor=2.0)
    ),
    weighting=weighting,
    turn_summarizer=summarizer,
    constraints=ConstraintSet(require_last_user=True),
)
```

---

## 工具调用与多模态

### 工具调用（Tool Calls）

lethes 完整支持 OpenAI tool calls 格式，无需任何额外配置：

```
用户消息
  ↓
assistant (content=null, tool_calls=[{id, type, function}])  ← 自动设为对方的依赖
  ↓
tool (tool_call_id=..., content="结果")                       ← 自动设为对方的依赖
  ↓
assistant ("根据工具结果，答案是…")
```

**关键保证：**

| 场景 | lethes 的处理 |
|---|---|
| tool result 被保留，但 assistant tool_calls 被截断 | `ConstraintChecker` 自动将 assistant 提升到 keep |
| assistant tool_calls 被保留，但 tool result 被截断 | `ConstraintChecker` 自动将 tool result 提升到 keep |
| 一次调用多个并行工具 | 全部双向绑定，整组保持一致 |
| tool 消息有预计算摘要，算法想摘要它 | 拒绝摘要，改为 drop（工具对不可拆分） |

依赖关系在 `Conversation.from_openai_messages` 解析时**自动注入**，无需手工设置。

### 多模态消息（图片）

```python
# URL 格式
{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}

# Base64 格式（data URI）
{"type": "image_url", "image_url": {
    "url": "data:image/png;base64,iVBOR...",
    "detail": "high"   # "low" | "high" | "auto"
}}
```

- 编排时只提取 `type=text` 块做权重计算和摘要
- 图片块原样透传，不被修改或丢弃
- 词元计数只统计文字部分（图片 token 由 API 计算）

---

## 编排逻辑详解

### 消息重要性评分公式

每条消息的最终权重由三个因子相乘得到：

```
weight = base_weight × relevance_score × recency_multiplier
```

其中 `relevance_score` 由 `SmartWeightingStrategy` 计算：

```
relevance_score(msg_i) = keyword_score(msg_i, query)
                       × pair_coherence_boost(msg_i)
                       × role_factor(msg_i)
```

| 因子 | 公式 | 说明 |
|---|---|---|
| `keyword_score` | BM25 归一化到 [floor, 1.0] | 消息文本与当前 query 的关键词重叠程度 |
| `pair_coherence_boost` | `max(kw_score, coherence × prev_user_score)` | assistant 回复继承上一条 user 消息的分数（Q&A 对不被拆散） |
| `role_factor` | `tool_penalty` if role=tool or tool_calls else 1.0 | 工具调用中间结果降权 |

`recency_multiplier` 由 `RecencyBiasedAlgorithm` 在选择阶段施加：

```
recency_multiplier = 1 + factor × (position_from_oldest / (n - 1))
```

最旧消息乘以 1.0，最新消息乘以 `1 + factor`（默认 factor=2.0）。

### 选择流程

```
所有消息 (n 条)
    │
    ▼ WeightingStrategy.score()
    │  关键词打分、pair coherence 提升、工具消息降权
    │
    ▼ algorithm.select()
    │  按 weight 排序 → 贪心填满预算
    │  （RecencyBiasedAlgorithm 在此施加近期乘数）
    │
    ▼ ConstraintChecker.repair()
       强制保留最后一条 user 消息
       解析 dependencies → tool 对永不分离
       满足 min_chat_messages
```

### 子智能体分析（LLMContextAnalyzer）

除关键词相关性外，还可启用 LLM 子智能体对消息重要性做语义分析。

**关键设计：五档分类而非浮点打分。**

LLM 不擅长输出校准的浮点数，但非常擅长分类决策。每条消息被分入五档之一：

| 标签 | 含义 | 映射权重 |
|---|---|---|
| **K** Keep | 必须保留：直接回答当前问题 | 1.00 |
| **H** Helpful | 应该保留：有用的背景信息 | 0.75 |
| **M** Maybe | 中性：可能有用，预算允许再保留 | 0.50 |
| **S** Skip | 可跳过：可能不需要，旧内容或偏题 | 0.25 |
| **D** Drop | 丢弃：与当前问题明显无关 | 0.05 |

LLM 输出格式（极简，节省 token）：
```json
{"labels": ["K", "H", "M", "S", "D", ...]}
```

解析器支持三种容错格式：JSON object → JSON array → 正则提取字母。

```python
from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
from lethes.weighting.llm_analyzer import LLMContextAnalyzer
from lethes.cache import RedisCache

weighting = CompositeWeightStrategy([
    (SmartWeightingStrategy(), 0.4),          # 快速关键词信号（无 API 调用）
    (LLMContextAnalyzer(                       # 语义分类（五档，结果缓存 30 分钟）
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
        cache=RedisCache.from_url("redis://..."),
    ), 0.6),
])
```

`LLMContextAnalyzer` 还支持可选的**两阶段入口逻辑**（`use_entry_logic=True`）：先发送紧凑的话题聚类概览，再让 LLM 通过 `expand_topic` 工具按需展开感兴趣的聚类，最后再打标签——适合对话极长、全量消息会撑爆子智能体上下文的场景。

---

## Flag 控制语法

用户可以在消息开头用 `!` 前缀直接控制编排行为：

```
!key=value,+persistKey=value2,-closeFeature 消息正文
```

| 前缀 | 含义 | 示例 |
|---|---|---|
| `!key` / `!key=val` | 临时 flag，仅当前轮有效 | `!nosum` |
| `!+key` / `!+key=val` | 持久 flag，写入会话状态 | `!+pin` |
| `!-key` | 移除持久 flag | `!-pin` |

### 内置 Flag

#### 截断 / 预算控制

| Flag | 作用 |
|---|---|
| `!full` | 关闭所有截断，传递完整上下文 |
| `!target=N` | 设置本轮 token 目标（尽量接近 N，而非上限） |
| `!context=N` | 本轮只保留最近 N 轮对话 |
| `!nosum` | 禁止摘要，超限消息直接丢弃 |

#### 强制保留（锚定）

| Flag | 作用 |
|---|---|
| `!pin` | 固定**当前**消息，永不截断或压缩 |
| `!recent=N` | 强制保留最近 N 条非系统消息（无论权重） |
| `!keep_tag=标签` | 保留所有带该标签的消息（配合 `!+tag=` 使用） |

#### 权重覆盖

| Flag | 作用 |
|---|---|
| `!weight=N` | 设置当前消息的基础权重（默认 1.0） |
| `!tool_penalty=F` | 本轮工具调用中间消息的权重乘数 |
| `!pair_coherence=F` | 本轮 Q&A 对相干系数（0.0–1.0） |

#### 消息元数据

| Flag | 作用 |
|---|---|
| `!tag=标签名` | 为当前消息添加标签（配合 `!keep_tag=` 使用） |

**示例：**

```
# 重要背景，永久固定
!+pin 这是重要的系统背景，请始终参考。

# 强制保留最近 4 条消息 + 设置紧凑 token 目标
!recent=4,target=6000 请基于最近对话回答。

# 本轮不需要工具调用历史（降低工具权重）
!tool_penalty=0.1 请总结一下对话，忽略工具调用细节。

# 标记重要消息，之后可按标签保留
!+tag=key_decision 我们决定使用 PostgreSQL 作为主数据库。
# ... 若干轮后 ...
!keep_tag=key_decision 请基于我们之前的架构决策继续。
```

---

## 架构详解

### 完整编排流水线（9 步）

```
原始消息列表（list[dict]，OpenAI 格式）
        │
        ▼  Conversation.from_openai_messages()
           解析角色、工具调用依赖、多模态内容块
        │
        ▼  ContextOrchestrator.process()
        │
  ①  Flag 解析       提取 !flag 前缀，写入 SessionFlags，清理消息内容
  ②  预算覆盖        full → 无限制；context=N → 轮数限制；target=N → 软目标
  ③  锚点固定        !recent=N 和 !keep_tag= 在评分前固定消息
  ④  词元计数        填充每条消息的 token_count（tiktoken）
  ⑤  动态权重评分   WeightingStrategy.score() → message.weight = base × relevance
  ⑥  算法选择       algorithm.select() → SelectionResult{keep / summarize / drop}
  ⑦  约束修复       ConstraintChecker 提升消息直到所有规则满足
  ⑧  并发摘要       asyncio.gather() 压缩所有 summarize 消息（先查缓存）
  ⑨  上下文组装 + 追踪  system + 压缩摘要块 + 保留消息（按原始顺序）；
                      PrefixSequenceTracker 记录本次发送序列
        │
        ▼  OrchestratorResult
           .conversation.to_openai_messages()  → 可直接传给任何 LLM
           .run_id          → 相关性 ID（匹配观测日志事件）
           .token_count     → 输出对话的总词元数
           .estimated_cost_usd
```

### 动态权重层（`weighting/`）

| 策略 | 速度 | 精度 | 依赖 |
|---|---|---|---|
| `StaticWeightStrategy` | 极快 | — | 无（score=1.0，默认） |
| `KeywordRelevanceStrategy` | 快 | 中等 | `rank_bm25`（可选）或内置 TF-IDF |
| `SmartWeightingStrategy` | 快 | 良好 | 无（关键词 + Q&A 相干 + 工具降权） |
| `EmbeddingSimilarityStrategy` | 中等 | 高 | 任意嵌入 API，建议配合缓存 |
| `LLMContextAnalyzer` | 较慢 | 最高 | 任意 chat API；结果缓存 30 分钟 |
| `CompositeWeightStrategy` | 取决于内部策略 | 可配置 | 内部策略的并集 |

所有策略实现同一协议：

```python
async def score(
    messages: list[Message],
    query: str,
    conversation: Conversation,
    context: dict | None = None,
) -> dict[str, float]:   # {message_id: relevance_score}
```

### 选择算法（`algorithms/`）

```python
def select(
    conversation: Conversation,
    budget: Budget,
    constraints: ConstraintSet,
    token_counter: TokenCounter,
) -> SelectionResult:
```

| 算法 | 说明 |
|---|---|
| `GreedyByWeightAlgorithm` | 按 weight 降序贪心填充预算；优先摘要而非删除（`prefer_summarize=True`） |
| `RecencyBiasedAlgorithm` | 施加线性近期乘数（默认 `factor=2.0`）后委托给贪心算法 |
| `DependencyAwareAlgorithm` | 装饰器：递归确保被保留消息的依赖链也被保留 |
| `PrefixCacheOptimizedAlgorithm` | 固定上次发送的最长公共前缀以最大化 KV 缓存命中，再贪心填充剩余预算 |

### 多级摘要（`summarizers/`）

```
TurnSummarizer          单个 user+assistant 轮 → 摘要字符串  [有缓存，TTL=24 小时]
      ↓
SegmentSummarizer       N 轮 → 段落摘要  （两遍压缩：先逐轮，再汇总）
      ↓
ConversationSummarizer  整个对话 → 高层概述
```

`LLMSummarizer` 通过 httpx 直接调用 `/v1/chat/completions`，不依赖任何 SDK，支持所有兼容 OpenAI 格式的服务。支持 `extra_body` 传入厂商特定参数。

### 缓存层（`cache/`）

```python
class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
```

| 后端 | 导入 | 说明 |
|---|---|---|
| `InMemoryCache` | `from lethes.cache import InMemoryCache` | 默认，零依赖 |
| `RedisCache` | `from lethes.cache import RedisCache` | `RedisCache.from_url("redis://...")`，需要 `lethes[redis]` |

缓存 key 由消息内容的 SHA-256 哈希派生，内容相同的上下文共享缓存。

### 预算类型（`models/budget.py`）

| 类型 | 关键参数 | 行为 |
|---|---|---|
| `TokenBudget(max_tokens=N)` | `max_tokens`（0 = 无限制） | 硬词元上限 |
| `TokenTargetBudget(target_tokens=N)` | `target_tokens`，`overshoot=150` | 软目标，尽量填满到 N（±150） |
| `CostBudget(max_cost_usd=N)` | `max_cost_usd`（0 = 无限制） | 软费用上限（USD） |
| `CompositeBudget(token_budget, cost_budget)` | 两者都须满足 | `CompositeBudget.unlimited()` 直接透传 |

---

## 可观测性

lethes 通过 structlog 发出结构化 JSON 日志事件，每次 `process()` 调用都有唯一的 `run_id`。在应用启动时配置一次：

```python
from lethes import configure_logging

configure_logging(
    level="DEBUG",   # DEBUG | INFO | WARNING | ERROR
    fmt="json",      # "json"（默认）| "console"
    handlers=[...],  # 可选：传入自定义 logging.Handler 列表
)
```

未设置 formatter 的 handler 会自动接收 structlog JSON formatter。

### lethes-observer

[lethes-observer](../lethes_observer) 是配套的实时仪表盘，通过 HTTP 接收日志事件，在浏览器中展示流水线运行、消息处置、权重及子智能体调用情况。

```python
from lethes import configure_logging
from lethes_observer.handler import LethesObserverHandler  # 来自 lethes_observer/

h = LethesObserverHandler("http://localhost:7456")
configure_logging(level="DEBUG", handlers=[h])
```

启动 observer 服务：
```bash
cd lethes_observer
uvicorn server:app --port 7456
```

---

## Open WebUI 集成

### 安装包版本

```python
from lethes.integrations.open_webui import OpenWebUIFilter as Filter
```

核心 Valves（不含 observer / 实时定价）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_tokens` | `10000` | 词元上限（0 = 无限制） |
| `algorithm` | `recency_biased` | `greedy_by_weight` / `recency_biased` / `dependency_aware` |
| `recency_factor` | `1.5` | 近期偏置系数 |
| `weighting` | `smart` | `static` / `keyword` / `smart` |
| `tool_penalty` | `0.5` | 工具调用消息的权重乘数 |
| `pair_coherence` | `0.8` | assistant 回复继承 user 消息分数的比例 |
| `llm_analysis` | `false` | 启用 LLM 子智能体上下文分析 |
| `llm_analysis_weight` | `0.6` | LLM 分析在复合权重中的占比（1−w 给基础策略） |
| `summary_api_base` | OpenAI | 摘要 / LLM 分析所用的 OpenAI 兼容 API 地址 |
| `summary_api_key` | — | API 密钥 |
| `summary_model` | `gpt-4o-mini` | 摘要 / LLM 分析模型 |
| `summary_target_ratio` | `0.3` | 摘要压缩比 |
| `retry_attempts` | `3` | 摘要调用的重试次数 |
| `nosum_by_default` | `false` | 全局禁用摘要 |
| `cache_backend` | `memory` | `memory` / `redis` |
| `redis_url` | `redis://redis:6379/0` | Redis 连接地址 |
| `pricing_config_path` | — | 自定义定价 JSON 文件路径；留空则使用 OpenRouter 实时定价 |
| `openrouter_pricing` | `true` | 启动时从 OpenRouter 获取实时模型定价；离线/隔离网络环境下可关闭 |

### 独立 filter（`lethes_observer/helper/open_webui_filter.py`）

可直接放入 Open WebUI functions 目录使用，无需安装包。包含所有基础 Valves，另外新增：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `use_openrouter_pricing` | `true` | 从 OpenRouter API 获取实时模型定价 |
| `pricing_cache_ttl_hours` | `24.0` | OpenRouter 定价缓存时长（小时） |
| `model_aliases` | `{}` | JSON 字典，原始模型 ID → 规范名称，在前缀剥离前应用。示例：`{"poe.gemini-3-flash": "gemini-3-flash-preview"}` |
| `observer_url` | — | lethes-observer 服务器地址（如 `http://localhost:7456`） |
| `observer_log_level` | `DEBUG` | 转发给 observer 的日志级别 |

独立 filter 还会在 `outlet()` 中读取 LLM 响应体的实际 token 数量和费用（优先于估算值），并通过 `pipeline.outlet` 事件将真实数据发送给 observer。

---

## 通用中间件

```python
from lethes.integrations import LethesMiddleware

middleware = LethesMiddleware(orchestrator=my_orchestrator)

# 直接调用
processed = await middleware(raw_messages, model_id="gpt-4o")

# 作为装饰器
@middleware.wrap
async def call_llm(messages, **kwargs):
    return await client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )
```

---

## 自定义扩展

所有核心扩展点均为 `Protocol`（结构化子类型），无需继承基类：

```python
# 自定义权重策略
class MyWeightingStrategy:
    async def score(
        self,
        messages: list[Message],
        query: str,
        conversation: Conversation,
        context: dict | None = None,
    ) -> dict[str, float]:   # {message_id: 相关性分数}
        ...
    def name(self) -> str: return "my_strategy"

# 自定义选择算法
class MyAlgorithm:
    def select(
        self,
        conversation: Conversation,
        budget: Budget,
        constraints: ConstraintSet,
        token_counter: TokenCounter,
    ) -> SelectionResult:
        ...
    def name(self) -> str: return "my_algo"

# 自定义摘要后端
class MySummarizer:
    async def summarize(
        self,
        messages: list[Message],
        *,
        target_ratio: float = 0.3,
        context_messages: list[Message] | None = None,
    ) -> str:
        ...
    def name(self) -> str: return "my_summarizer"

# 自定义缓存后端
class MyCache:
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
```

---

## 项目结构

```
src/lethes/
├── models/           消息、对话、预算、定价模型
├── flags/            Flag 语法解析器与会话状态管理
├── weighting/        动态相关性评分层
│   ├── static.py     StaticWeightStrategy
│   ├── keyword.py    KeywordRelevanceStrategy
│   ├── smart.py      SmartWeightingStrategy
│   ├── embedding.py  EmbeddingSimilarityStrategy
│   ├── llm_analyzer.py  LLMContextAnalyzer
│   └── composite.py  CompositeWeightStrategy
├── algorithms/       上下文选择算法
├── cache/            缓存后端与前缀序列追踪
├── summarizers/      LLMSummarizer、TurnSummarizer、SegmentSummarizer、ConversationSummarizer
├── engine/           约束检查、编排计划、主编排器
├── integrations/     OpenWebUIFilter、LethesMiddleware
├── observability.py  configure_logging、get_logger、make_formatter
└── utils/            词元计数、内容提取、ID 生成
```

---

## License

MIT
