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

- **动态权重** — 消息权重不是固定数字，而是根据当前输入和主题实时计算的相关性评分
- **约束驱动** — 以词元预算、费用预算为约束，最大化上下文信息量
- **算法可插拔** — 贪心、近期偏置、依赖感知、前缀缓存优化，或自定义
- **用户可控** — 在对话中直接用 flag 语法精确控制编排行为
- **多级摘要** — Turn 级、段落级、对话级三层压缩，而非简单删除
- **KV 缓存优化** — 追踪历史发送序列，最大化前缀命中，降低 API 成本
- **工具调用感知** — 完整支持 OpenAI tool calls 格式；自动绑定 assistant/tool 消息对的依赖关系，确保上下文截断后序列仍合法
- **多模态就绪** — 原生支持 `image_url` 内容块（URL 及 base64 格式），编排时自动提取文本，图像块透明透传
- **无厂商绑定** — 摘要和嵌入通过标准 OpenAI 兼容接口调用，支持 Ollama、vLLM 等

---

## 安装

```bash
pip install lethes                  # 基础安装
pip install lethes[redis]           # 启用 Redis 缓存
pip install lethes[bm25]            # 启用 BM25 关键词相关性
pip install lethes[all]             # 安装所有可选依赖
```

**最低要求：** Python 3.11+，tiktoken，pydantic，httpx

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
     "name": "get_weather", "content": "22°C, 多云"},

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
from lethes.weighting import CompositeWeightStrategy, KeywordRelevanceStrategy, EmbeddingSimilarityStrategy
from lethes.summarizers import LLMSummarizer, TurnSummarizer
from lethes.cache import RedisCache
from lethes.engine import ConstraintSet

# 动态权重：70% 嵌入相似度 + 30% 关键词匹配
weighting = CompositeWeightStrategy([
    (EmbeddingSimilarityStrategy(
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="text-embedding-3-small",
        cache=RedisCache.from_url("redis://localhost:6379/0"),
    ), 0.7),
    (KeywordRelevanceStrategy(), 0.3),
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

| 问题 | lethes 的处理 |
|---|---|
| tool result 被保留，但 assistant tool_calls 被截断 | ConstraintChecker 自动将 assistant 提升到 keep |
| assistant tool_calls 被保留，但 tool result 被截断 | ConstraintChecker 自动将 tool result 提升到 keep |
| tool 消息有预计算摘要，算法想摘要它 | 拒绝摘要，改为 drop（工具对不可拆分） |
| 一次调用多个并行工具 | 全部双向绑定，整组保持一致 |

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

- 编排时提取 `type=text` 块做权重计算和摘要
- 图片块（`image_url`、`image` 等）**原样透传**，不被修改或丢弃
- 包含图片的消息的词元计数只统计文字部分（图片 token 由 API 自行计算）

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

最旧消息乘以 1.0，最新消息乘以 `1 + factor`（默认 factor=1.5 → 2.5x）。

### 选择流程图解

```
所有消息 (n条)
    │
    ▼ SmartWeightingStrategy
    │  ① BM25 关键词打分
    │  ② pair coherence 提升 assistant 回复
    │  ③ tool 消息降权 × tool_penalty
    │
    ▼ RecencyBiasedAlgorithm
    │  ④ 近期乘数 (线性递增到 1+factor)
    │  ⑤ 按 weight 排序 → 贪心填满预算
    │
    ▼ ConstraintChecker.repair
       ⑥ 强制保留最后一条 user 消息
       ⑦ 解析 dependencies → tool 对永不分离
       ⑧ 满足 min_chat_messages
```

### 子智能体分析（LLMContextAnalyzer）

除了关键词相关性，还可以启用 LLM 子智能体对消息重要性做语义分析。

**关键设计：五档分类而非浮点打分。**

LLM 不擅长输出校准的浮点数（0.73 和 0.75 对它来说没有区别），但它非常擅长做分类决策。每条消息被分入五档之一：

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

解析器支持三种容错格式：JSON object → JSON array → 正则提取字母（应对 LLM 不严格遵守格式的情况）。

```python
from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
from lethes.weighting.llm_analyzer import LLMContextAnalyzer

weighting = CompositeWeightStrategy([
    (SmartWeightingStrategy(), 0.4),          # 快速关键词信号（无 API 调用）
    (LLMContextAnalyzer(                       # 语义分类（五档，结果缓存）
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
        cache=RedisCache.from_url("redis://..."),
    ), 0.6),
])

---

## Flag 控制语法

用户可以在对话消息开头用 `!` 前缀直接控制编排行为：

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
| `!tool_penalty=F` | 本轮工具调用中间消息的权重乘数（默认 0.5） |
| `!pair_coherence=F` | 本轮 Q&A 对相干系数（0.0–1.0，默认 0.8） |

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

### 编排流水线（9 步）

```
原始消息列表 (list[dict], OpenAI 格式)
        │
        ▼
Conversation.from_openai_messages()
        │
        ▼
ContextOrchestrator.process()
        │
  ①  Flag 解析         提取 !flag 前缀，写入 SessionFlags，清理消息内容
  ②  预算覆盖         full → 无限制；context=N → 轮数限制；nosum → 禁摘要
  ③  词元计数         并发填充每条消息的 token_count
  ④  动态权重评分     WeightingStrategy.score(messages, query=当前输入)
                       → message.weight = base_weight × relevance_score
  ⑤  算法选择        algorithm.select() → SelectionResult{keep/summarize/drop}
  ⑥  约束修复        ConstraintChecker 确保最后一条用户消息保留、系统消息保留等
  ⑦  并发摘要        asyncio.gather() 并发压缩所有 summarize 消息（先查缓存）
  ⑧  上下文组装      system + 压缩摘要块 + 保留消息（按原始顺序）
  ⑨  前缀记录        PrefixSequenceTracker 记录本次发送序列（供下次缓存优化）
        │
        ▼
OrchestratorResult.conversation.to_openai_messages()
```

### 动态权重层 (`weighting/`)

**权重不是固定值，而是当前输入与消息内容相关性的实时评分：**

```
message.weight = message.base_weight × relevance_score
```

`base_weight` 由用户通过 API 或 `!weight=N` flag 设置；
`relevance_score` 由以下策略之一动态计算：

| 策略 | 速度 | 精度 | 依赖 |
|---|---|---|---|
| `StaticWeightStrategy` | 极快 | — | 无（默认，score=1.0） |
| `KeywordRelevanceStrategy` | 快 | 中等 | `rank_bm25`（可选）或内置 TF-IDF |
| `EmbeddingSimilarityStrategy` | 中等 | 高 | 任意嵌入 API（带缓存） |
| `CompositeWeightStrategy` | 取决于内部策略 | 可配置 | 内部策略的并集 |

`EmbeddingSimilarityStrategy` 支持**话题增强**：通过 `TopicDetector` 识别话题聚类，同话题的消息获得额外分数加成。

### 选择算法 (`algorithms/`)

所有算法实现 `SelectionAlgorithm` 协议（结构化子类型，无需继承）：

```python
class SelectionAlgorithm(Protocol):
    def select(
        self,
        conversation: Conversation,
        budget: Budget,
        constraints: ConstraintSet,
        token_counter: TokenCounter,
    ) -> SelectionResult: ...
```

| 算法 | 说明 |
|---|---|
| `GreedyByWeightAlgorithm` | 按 weight 降序贪心填充预算，有摘要则优先摘要替代删除 |
| `RecencyBiasedAlgorithm` | 对近期消息施加衰减系数后委托给贪心算法 |
| `DependencyAwareAlgorithm` | 装饰器：确保被保留消息的依赖链也被保留（递归解析） |
| `PrefixCacheOptimizedAlgorithm` | 固定上次发送的最长公共前缀以最大化 KV 缓存命中，再贪心填充剩余预算 |

### 多级摘要 (`summarizers/`)

```
TurnSummarizer      单个 user+assistant 轮 → 摘要字符串（有缓存）
      ↓
SegmentSummarizer   N 轮 → 段落摘要（两遍压缩：先逐轮，再汇总）
      ↓
ConversationSummarizer  整个对话 → 高层概述
```

`LLMSummarizer` 通过 httpx 直接调用 `/v1/chat/completions`，不依赖任何 SDK，支持所有兼容 OpenAI 格式的服务。

### 缓存层 (`cache/`)

实现 `CacheBackend` 协议即可替换：

```python
class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
```

内置两种实现：`InMemoryCache`（默认，无依赖）和 `RedisCache`（需要 `lethes[redis]`）。

缓存 key 由消息内容的 SHA-256 哈希派生，语义相同的上下文共享缓存，不会重复调用摘要 API。

---

## Open WebUI 集成

在 Open WebUI Functions 中直接使用：

```python
from lethes.integrations.open_webui import OpenWebUIFilter as Filter
```

或将 `src/lethes/integrations/open_webui.py` 复制到 Open WebUI functions 目录。

`Valves` 配置项：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `max_tokens` | `10000` | 上下文词元上限（0 = 不限） |
| `max_turns` | `25` | 保留最近 N 轮（暂未启用，用 `!context=N` 替代） |
| `algorithm` | `greedy_by_weight` | 选择算法 |
| `recency_factor` | `2.0` | 近期偏置系数 |
| `weighting` | `keyword` | 权重策略（`static` / `keyword`） |
| `summary_api_base` | OpenAI | 摘要 API 地址 |
| `summary_api_key` | — | 摘要 API 密钥 |
| `summary_model` | `gpt-4o-mini` | 摘要模型 |
| `cache_backend` | `memory` | 缓存后端（`memory` / `redis`） |
| `redis_url` | `redis://redis:6379/0` | Redis 地址 |

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
    ) -> dict[str, float]:
        # 返回 {message_id: 相关性分数} 即可
        ...
    def name(self) -> str: return "my_strategy"

# 自定义选择算法
class MyAlgorithm:
    def select(self, conversation, budget, constraints, token_counter) -> SelectionResult:
        ...
    def name(self) -> str: return "my_algo"

# 自定义摘要后端
class MySummarizer:
    async def summarize(self, messages, *, target_ratio=0.3, context_messages=None) -> str:
        ...
    def name(self) -> str: return "my_summarizer"
```

---

## 项目结构

```
src/lethes/
├── models/           消息、对话、预算、定价模型
├── flags/            Flag 语法解析器与会话状态管理
├── weighting/        动态相关性评分层
├── algorithms/       上下文选择算法
├── cache/            缓存后端与前缀序列追踪
├── summarizers/      多级摘要实现
├── engine/           约束检查、编排计划、主编排器
├── integrations/     Open WebUI Filter & 通用中间件
├── config/pricing/   内置模型定价表（JSON）
└── utils/            词元计数、内容提取、ID 生成
```

---

## 定价表

内置 `default_pricing.json` 涵盖主流模型（USD / 1M tokens），支持 glob 匹配：

| 模型族 | 输入 | 缓存命中 | 输出 |
|---|---|---|---|
| GPT-4o | $2.50 | $1.25 | $10.00 |
| GPT-4o-mini | $0.15 | $0.075 | $0.60 |
| Claude Sonnet 4 / 3.5 | $3.00 | $0.30 | $15.00 |
| Claude Haiku 4 / 3.5 | $0.80 | $0.08 | $4.00 |
| Gemini 2.0 Flash | $0.10 | $0.025 | $0.40 |
| Gemini 2.5 Pro | $1.25 | $0.31 | $10.00 |

可通过 `Valves.pricing_config_path` 指定自定义定价文件。

---

## License

MIT
