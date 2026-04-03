# lethes

**Constraint-driven LLM context management library** — a message orchestration engine that treats conversation history as an optimization problem.

> *lethes* is named after the river of oblivion in Greek mythology. It decides what must be remembered, what can be compressed, and what should be forgotten.

---

## Core Idea

Traditional context management is a sliding window — truncate from the front when you run out of space. lethes takes a fundamentally different approach:

| Traditional | lethes |
|---|---|
| Fit as many messages as possible | Optimize under constraints |
| All messages are equal | Every message has a dynamic weight |
| Truncation = deletion | Truncation = selective compression or drop |
| Fixed strategy | Pluggable algorithms |
| No user control | In-conversation flag syntax |

---

## Features

- **Dynamic weighting** — message weights are real-time relevance scores against the current query, not fixed numbers
- **Constraint-driven** — maximize context information subject to token-budget and cost-budget constraints
- **Pluggable algorithms** — greedy, recency-biased, dependency-aware, prefix-cache-optimized, or bring your own
- **User-controllable** — a `!flag` syntax lets users adjust orchestration behaviour inline within the conversation
- **Multi-level summarization** — turn-level, segment-level, and conversation-level compression instead of blunt deletion
- **KV-cache optimization** — tracks previously sent sequences to maximize prefix hits and lower API cost
- **Tool-call aware** — full OpenAI tool calls support; automatically binds assistant/tool message pairs so truncation never produces an invalid sequence
- **Multimodal-ready** — native `image_url` content blocks (URL and base64); text is extracted for scoring, image blocks are passed through untouched
- **Vendor-agnostic** — summarization and embedding use any OpenAI-compatible endpoint (Ollama, vLLM, OpenRouter, …)
- **Structured observability** — structlog-based JSON events; plug in [lethes-observer](https://github.com/kb5000/lethes-observer) for a real-time web dashboard

---

## Installation

```bash
pip install lethes                  # core
pip install lethes[redis]           # enable Redis cache
pip install lethes[bm25]            # enable BM25 keyword relevance
pip install lethes[langchain]       # LangChain adapter
pip install lethes[all]             # all optional dependencies
```

**Requirements:** Python 3.11+, tiktoken, pydantic ≥ 2, pydantic-settings ≥ 2, httpx ≥ 0.27, structlog ≥ 24

---

## Quick Start

### Minimal usage

```python
from lethes import ContextOrchestrator, Conversation, TokenBudget

orchestrator = ContextOrchestrator(
    budget=TokenBudget(max_tokens=8000),
)

result = await orchestrator.process(
    Conversation.from_openai_messages(raw_messages)
)

ready_messages = result.conversation.to_openai_messages()
# → standard OpenAI format, ready for any LLM
```

`raw_messages` accepts any OpenAI message format:

```python
messages = [
    # Plain text
    {"role": "user", "content": "What's the weather today?"},

    # Tool call (assistant content is null)
    {"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_abc", "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'}
    }]},

    # Tool result
    {"role": "tool", "tool_call_id": "call_abc",
     "name": "get_weather", "content": "22°C, partly cloudy"},

    # Multimodal (image + text)
    {"role": "user", "content": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64,iVBOR...",
            "detail": "high"
        }},
    ]},
]
```

### Full configuration

```python
from lethes import ContextOrchestrator, Conversation, TokenBudget
from lethes.algorithms import RecencyBiasedAlgorithm, DependencyAwareAlgorithm
from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
from lethes.weighting.embedding import EmbeddingSimilarityStrategy
from lethes.summarizers import LLMSummarizer, TurnSummarizer
from lethes.cache import RedisCache
from lethes.engine import ConstraintSet

# Weighting: 70% embedding similarity + 30% smart keyword/coherence
weighting = CompositeWeightStrategy([
    (EmbeddingSimilarityStrategy(
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="text-embedding-3-small",
        cache=RedisCache.from_url("redis://localhost:6379/0"),
    ), 0.7),
    (SmartWeightingStrategy(), 0.3),
])

# Summarizer: any OpenAI-compatible endpoint
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

## Tool Calls & Multimodal

### Tool calls

lethes fully supports the OpenAI tool calls format with no extra configuration:

```
user message
  ↓
assistant  (content=null, tool_calls=[{id, type, function}])  ← auto-bound as dependency
  ↓
tool       (tool_call_id=..., content="result")               ← auto-bound as dependency
  ↓
assistant  ("Based on the tool result, the answer is…")
```

**Guarantees:**

| Scenario | lethes behaviour |
|---|---|
| Tool result kept, assistant tool_calls dropped | `ConstraintChecker` promotes the assistant to `keep` |
| Assistant tool_calls kept, tool result dropped | `ConstraintChecker` promotes the tool result to `keep` |
| Parallel tool calls (multiple in one turn) | All pairs are bound bidirectionally and kept together |
| A tool message has a cached summary and the algorithm wants to summarize it | Summarization refused; treated as `drop` (tool pairs are indivisible) |

Dependencies are injected automatically by `Conversation.from_openai_messages` — no manual wiring needed.

### Multimodal messages

```python
# URL format
{"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}

# Base64 data URI
{"type": "image_url", "image_url": {
    "url": "data:image/png;base64,iVBOR...",
    "detail": "high"   # "low" | "high" | "auto"
}}
```

- Only `type=text` blocks are used for weight scoring and summarization
- Image blocks are passed through unchanged
- Token counting covers text content only (image tokens are billed by the API)

---

## How Orchestration Works

### Message importance score

Each message's final weight is a product of three factors:

```
weight = base_weight × relevance_score × recency_multiplier
```

`relevance_score` is computed by `SmartWeightingStrategy`:

```
relevance_score(msg_i) = keyword_score(msg_i, query)
                       × pair_coherence_boost(msg_i)
                       × role_factor(msg_i)
```

| Factor | Formula | Description |
|---|---|---|
| `keyword_score` | BM25 normalised to [floor, 1.0] | Keyword overlap between message text and the current query |
| `pair_coherence_boost` | `max(kw_score, coherence × prev_user_score)` | Assistant replies inherit part of the preceding user message's score — keeps Q&A pairs together |
| `role_factor` | `tool_penalty` if role=tool or has tool_calls, else 1.0 | Downweights intermediate tool-call messages |

`recency_multiplier` is applied by `RecencyBiasedAlgorithm` during selection:

```
recency_multiplier = 1 + factor × (position_from_oldest / (n - 1))
```

Oldest message → ×1.0; newest message → ×(1 + factor). Default `factor=2.0`.

### Selection pipeline

```
All messages (n)
    │
    ▼ WeightingStrategy.score()
    │  keyword scoring, pair coherence boost, tool downweight
    │
    ▼ algorithm.select()
    │  sort by weight → greedy fill within budget
    │  (recency multiplier applied here by RecencyBiasedAlgorithm)
    │
    ▼ ConstraintChecker.repair()
       force-keep last user message
       resolve dependencies → tool pairs never split
       satisfy min_chat_messages
```

### LLM sub-agent analysis (`LLMContextAnalyzer`)

Beyond keyword relevance, you can enable an LLM sub-agent for semantic importance classification.

**Key design: five-label classification, not floating-point scores.**

LLMs are poor at calibrated floats (0.73 vs 0.75 means nothing to them) but excellent at classification. Each message gets one label:

| Label | Meaning | Weight |
|---|---|---|
| **K** Keep | Must keep — directly answers the current question | 1.00 |
| **H** Helpful | Should keep — useful background context | 0.75 |
| **M** Maybe | Neutral — marginally relevant, keep if budget allows | 0.50 |
| **S** Skip | Likely not needed — old, off-topic, or superseded | 0.25 |
| **D** Drop | Clearly irrelevant to the current question | 0.05 |

Output format (compact, minimal tokens):
```json
{"labels": ["K", "H", "M", "S", "D", ...]}
```

The parser accepts three fallback formats: JSON object → JSON array → regex letter extraction.

```python
from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
from lethes.weighting.llm_analyzer import LLMContextAnalyzer
from lethes.cache import RedisCache

weighting = CompositeWeightStrategy([
    (SmartWeightingStrategy(), 0.4),       # fast keyword signal (no API call)
    (LLMContextAnalyzer(                    # semantic 5-label classification (cached)
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
        cache=RedisCache.from_url("redis://..."),
    ), 0.6),
])
```

`LLMContextAnalyzer` also supports an optional **two-phase entry logic** (`use_entry_logic=True`): it first receives a compact topic-cluster overview, then calls an `expand_topic` tool to examine specific clusters in detail before labelling — useful for very long conversations where the full message list would overflow the sub-agent's context.

---

## Flag Control Syntax

Users control orchestration behaviour inline with a `!` prefix:

```
!key=value,+persistKey=value2,-closeFeature message body
```

| Prefix | Meaning | Example |
|---|---|---|
| `!key` / `!key=val` | Temporary — current turn only | `!nosum` |
| `!+key` / `!+key=val` | Persistent — stored in session state | `!+pin` |
| `!-key` | Remove a persistent flag | `!-pin` |

### Built-in flags

#### Truncation / budget control

| Flag | Effect |
|---|---|
| `!full` | Disable all truncation, pass the full context |
| `!target=N` | Token target for this turn (aim for N, not a hard cap) |
| `!context=N` | Keep only the most recent N conversation turns |
| `!nosum` | Disable summarization; over-budget messages are dropped |

#### Anchoring (force-keep)

| Flag | Effect |
|---|---|
| `!pin` | Pin the **current** message — never truncated or summarized |
| `!recent=N` | Force-keep the N most recent non-system messages, regardless of weight |
| `!keep_tag=label` | Keep all messages with the given tag (use with `!+tag=`) |

#### Weight overrides

| Flag | Effect |
|---|---|
| `!weight=N` | Base weight of the current message (default 1.0) |
| `!tool_penalty=F` | Override tool-call downweight multiplier for this turn |
| `!pair_coherence=F` | Override Q&A coherence factor for this turn (0.0–1.0) |

#### Metadata

| Flag | Effect |
|---|---|
| `!tag=label` | Attach a tag to the current message (use with `!keep_tag=`) |

**Examples:**

```
# Pin important background permanently
!+pin This is critical system context. Always refer to it.

# Force keep last 4 messages + compact token target
!recent=4,target=6000 Answer based on the recent conversation.

# Deprioritize tool call history this turn
!tool_penalty=0.1 Summarize the conversation, ignoring tool-call details.

# Tag a key decision and retrieve it by tag later
!+tag=key_decision We decided to use PostgreSQL as the primary database.
# ... several turns later ...
!keep_tag=key_decision Continue based on our earlier architecture decisions.
```

---

## Architecture

### Full orchestration pipeline (9 steps)

```
Raw message list  (list[dict], OpenAI format)
        │
        ▼  Conversation.from_openai_messages()
           Parses roles, tool_call dependencies, multimodal blocks
        │
        ▼  ContextOrchestrator.process()
        │
  ①  Flag parsing      Extract !flag prefixes → SessionFlags; clean message content
  ②  Budget override   full → unlimited; context=N → turn limit; target=N → soft target
  ③  Anchor pinning    !recent=N and !keep_tag= pin messages before scoring
  ④  Token counting    Fill token_count on each message (tiktoken)
  ⑤  Dynamic weighting WeightingStrategy.score() → message.weight = base × relevance
  ⑥  Algorithm         algorithm.select() → SelectionResult {keep / summarize / drop}
  ⑦  Constraint repair ConstraintChecker promotes messages until all rules satisfied
  ⑧  Summarization     asyncio.gather() compresses all summarize messages (cache-first)
  ⑨  Assembly + tracking  system msgs + summary block + kept msgs (original order);
                           PrefixSequenceTracker records sent sequence for next turn
        │
        ▼  OrchestratorResult
           .conversation.to_openai_messages()  → ready for any LLM
           .run_id          → correlation ID (matches observer log events)
           .token_count     → total tokens in outgoing conversation
           .estimated_cost_usd
```

### Weighting strategies (`weighting/`)

| Strategy | Speed | Accuracy | Requires |
|---|---|---|---|
| `StaticWeightStrategy` | Instant | — | Nothing (score = 1.0 for all) |
| `KeywordRelevanceStrategy` | Fast | Moderate | `rank_bm25` (optional) or built-in TF-IDF |
| `SmartWeightingStrategy` | Fast | Good | Nothing (keyword + Q&A coherence + tool penalty) |
| `EmbeddingSimilarityStrategy` | Medium | High | Any embedding API + cache recommended |
| `LLMContextAnalyzer` | Slow | Highest | Any chat API; results cached (30 min TTL) |
| `CompositeWeightStrategy` | Depends | Configurable | Union of inner strategies |

All strategies implement the same protocol:

```python
async def score(
    messages: list[Message],
    query: str,
    conversation: Conversation,
    context: dict | None = None,
) -> dict[str, float]:   # {message_id: relevance_score}
```

### Selection algorithms (`algorithms/`)

```python
def select(
    conversation: Conversation,
    budget: Budget,
    constraints: ConstraintSet,
    token_counter: TokenCounter,
) -> SelectionResult:
```

| Algorithm | Description |
|---|---|
| `GreedyByWeightAlgorithm` | Sort by weight descending, greedily fill budget; prefer summarize over drop (`prefer_summarize=True`) |
| `RecencyBiasedAlgorithm` | Apply a linear recency multiplier (default `factor=2.0`) then delegate to greedy |
| `DependencyAwareAlgorithm` | Decorator: ensure kept messages' dependency chains are recursively kept |
| `PrefixCacheOptimizedAlgorithm` | Lock the longest common prefix from the previous turn to maximize KV-cache hits, then greedy-fill the remainder |

### Summarizers (`summarizers/`)

```
TurnSummarizer          single turn (user+assistant) → summary string  [cached, TTL=24 h]
      ↓
SegmentSummarizer       N turns → segment summary  (two-pass: per-turn then aggregate)
      ↓
ConversationSummarizer  full conversation → high-level overview
```

`LLMSummarizer` calls `/v1/chat/completions` directly via httpx — no SDK dependency, works with any OpenAI-compatible service. Supports `extra_body` for provider-specific parameters.

### Cache layer (`cache/`)

```python
class CacheBackend(Protocol):
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
```

| Backend | Import | Notes |
|---|---|---|
| `InMemoryCache` | `from lethes.cache import InMemoryCache` | Default; zero dependencies |
| `RedisCache` | `from lethes.cache import RedisCache` | `RedisCache.from_url("redis://...")` requires `lethes[redis]` |

Cache keys are derived from SHA-256 hashes of message content — identical contexts share cache entries.

### Budget types (`models/budget.py`)

| Type | Key parameter | Behaviour |
|---|---|---|
| `TokenBudget(max_tokens=N)` | `max_tokens` (0 = unlimited) | Hard token cap |
| `TokenTargetBudget(target_tokens=N)` | `target_tokens`, `overshoot=150` | Aim to fill to ~N tokens (soft target) |
| `CostBudget(max_cost_usd=N)` | `max_cost_usd` (0 = unlimited) | Soft USD cost cap |
| `CompositeBudget(token_budget, cost_budget)` | Both must be satisfied | `CompositeBudget.unlimited()` for pass-through |

---

## Observability

lethes emits structured JSON log events via structlog, keyed by `run_id` per `process()` call. Configure once at startup:

```python
from lethes import configure_logging
import logging

configure_logging(
    level="DEBUG",   # DEBUG | INFO | WARNING | ERROR
    fmt="json",      # "json" (default) | "console"
    handlers=[...],  # optional: provide your own logging.Handler list
)
```

Each handler with no formatter set receives the structlog JSON formatter automatically.

### lethes-observer

[lethes-observer](https://github.com/kb5000/lethes-observer) is a companion real-time dashboard that receives log events over HTTP and provides an in-browser view of pipeline runs, message dispositions, weights, and sub-agent calls.

```python
from lethes import configure_logging
from lethes_observer.handler import LethesObserverHandler  # from lethes_observer/

h = LethesObserverHandler("http://localhost:7456")
configure_logging(level="DEBUG", handlers=[h])
```

Start the observer server:
```bash
cd lethes_observer
uvicorn server:app --port 7456
```

---

## Open WebUI Integration

### Installed package

```python
from lethes.integrations.open_webui import OpenWebUIFilter as Filter
```

Core Valves (no observer / live pricing):

| Valve | Default | Description |
|---|---|---|
| `max_tokens` | `10000` | Token limit (0 = unlimited) |
| `algorithm` | `recency_biased` | `greedy_by_weight` / `recency_biased` / `dependency_aware` |
| `recency_factor` | `1.5` | Recency bias strength |
| `weighting` | `smart` | `static` / `keyword` / `smart` |
| `tool_penalty` | `0.5` | Weight multiplier for tool-call messages |
| `pair_coherence` | `0.8` | Score fraction inherited by assistant replies |
| `llm_analysis` | `false` | Enable LLM sub-agent context analysis |
| `llm_analysis_weight` | `0.6` | LLM analysis fraction in composite weighting |
| `summary_api_base` | OpenAI | OpenAI-compatible API for summarization / LLM analysis |
| `summary_api_key` | — | API key |
| `summary_model` | `gpt-4o-mini` | Model for summarization / LLM analysis |
| `summary_target_ratio` | `0.3` | Summarization compression ratio |
| `retry_attempts` | `3` | Retry attempts for summarization calls |
| `nosum_by_default` | `false` | Globally disable summarization |
| `cache_backend` | `memory` | `memory` / `redis` |
| `redis_url` | `redis://redis:6379/0` | Redis URL |
| `pricing_config_path` | — | Path to a custom pricing JSON file; leave empty to use OpenRouter live pricing |
| `openrouter_pricing` | `true` | Fetch live model pricing from OpenRouter at startup; disable in offline/air-gapped environments |

### Standalone filter (`lethes_observer/helper/open_webui_filter.py`)

The standalone file can be dropped directly into your Open WebUI functions directory — no package install needed. It includes all base Valves plus:

| Valve | Default | Description |
|---|---|---|
| `use_openrouter_pricing` | `true` | Fetch live model pricing from OpenRouter API |
| `pricing_cache_ttl_hours` | `24.0` | How long to cache OpenRouter pricing data |
| `model_aliases` | `{}` | JSON map of raw model IDs → canonical names, applied before prefix-stripping. Example: `{"poe.gemini-3-flash": "gemini-3-flash-preview"}` |
| `observer_url` | — | lethes-observer server URL (e.g. `http://localhost:7456`) |
| `observer_log_level` | `DEBUG` | Log level forwarded to the observer |

The standalone filter also:
- Reads actual token counts and cost from the LLM response body (`outlet()`) rather than always estimating
- Logs a `pipeline.outlet` event so the observer can display real vs estimated cost

---

## Generic Middleware

```python
from lethes.integrations import LethesMiddleware

middleware = LethesMiddleware(orchestrator=my_orchestrator)

# Direct call
processed = await middleware(raw_messages, model_id="gpt-4o")

# As a decorator
@middleware.wrap
async def call_llm(messages, **kwargs):
    return await client.chat.completions.create(
        model="gpt-4o", messages=messages, **kwargs
    )
```

---

## Custom Extensions

All extension points are `Protocol` types — no base class required:

```python
# Custom weighting strategy
class MyWeightingStrategy:
    async def score(
        self,
        messages: list[Message],
        query: str,
        conversation: Conversation,
        context: dict | None = None,
    ) -> dict[str, float]:   # {message_id: relevance_score}
        ...
    def name(self) -> str: return "my_strategy"

# Custom selection algorithm
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

# Custom summarization backend
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

# Custom cache backend
class MyCache:
    async def get(self, key: str) -> str | None: ...
    async def set(self, key: str, value: str, ttl: int | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def exists(self, key: str) -> bool: ...
```

---

## Project Structure

```
src/lethes/
├── models/           Messages, conversations, budgets, pricing models
├── flags/            Flag syntax parser and session state
├── weighting/        Dynamic relevance scoring strategies
│   ├── static.py     StaticWeightStrategy
│   ├── keyword.py    KeywordRelevanceStrategy
│   ├── smart.py      SmartWeightingStrategy
│   ├── embedding.py  EmbeddingSimilarityStrategy
│   ├── llm_analyzer.py  LLMContextAnalyzer
│   └── composite.py  CompositeWeightStrategy
├── algorithms/       Context selection algorithms
├── cache/            Cache backends + prefix sequence tracker
├── summarizers/      LLMSummarizer, TurnSummarizer, SegmentSummarizer, ConversationSummarizer
├── engine/           ConstraintChecker, ContextPlan, ContextOrchestrator
├── integrations/     OpenWebUIFilter, LethesMiddleware
├── observability.py  configure_logging, get_logger, make_formatter
└── utils/            Token counting, content extraction, ID generation
```

---

## License

MIT
