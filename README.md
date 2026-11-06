# lethes v2 (alpha)

Original-text recall and cache-stable context for long Open WebUI conversations,
including fiction and roleplay. This is a breaking rewrite of v1.

lethes receives the complete current history on every request. It keeps recent
conversation verbatim, retrieves earlier original passages, and inserts them as
historical reminders. It does **not** extract or maintain fictional rules, facts,
character state, or a rolling summary.

## Mechanism

- **Path-local recall:** lexical BM25 with Chinese characters/bigrams, optional
  semantic embeddings and query expansion, multiple recent messages as queries,
  neighboring passages, union of candidates, deduplication and budgeted inclusion.
  There is no small fixed top-k cutoff. Oversized/overflow candidates are reported.
- **Stable overlays:** a reminder's exact payload and insertion point stay fixed
  throughout its cache cycle. Subsequent messages and reminders are appended.
- **Reuse before insertion:** recent originals and existing reminders satisfy the
  current query's recall budget. Already-covered passages are not quoted again,
  and their slots are not refilled with weaker candidates. Near-tied optional
  matches prefer resident evidence; direct-query matches retain priority.
- **Versioned paths:** hashes include content, role and ancestry. Input edits,
  output edits, regeneration, rewinds and branches are ordinary paths, even when
  the host reuses message IDs. Snapshots can only be restored on their own path.
- **Persistent state:** SQLite WAL stores immutable assembly snapshots and separate
  request attempts. Restarting does not lose reminder placement. Concurrent writes
  to identical histories converge on the first stored snapshot.
- **Cycle economics:** with configured prices, compare renewal-cycle average costs
  for an indefinitely continuing chat. No user-departure horizon is assumed.
  Capacity and path validity remain hard constraints.
- **Usage-first budgeting:** matching historical provider input counts take
  precedence over local counting. Appended content uses a measured prefix plus
  an estimated increment. Count provenance, potential reusable prefixes and
  actual provider cache usage remain distinct.

## Install and run

```bash
uv sync --extra dev
uv run pytest
uv run python examples/replay.py --synthetic-turns 300 --output /tmp/lethes-replay.csv
```

```python
from lethes import ContextConfig, ContextManager, SQLiteStore

manager = ContextManager(
    ContextConfig(context_window=32768, reply_reserve=4096),
    store=SQLiteStore("./data/lethes.sqlite3"),
)
result = await manager.prepare(
    complete_current_history,
    user_id="user-id", chat_id="chat-id", model="model-id",
)
# Send result.messages to the model. The next invocation again supplies the
# complete current original history, including any edits or chosen branches.
# When available:
manager.record_usage(result.request_id, provider_usage, model=returned_model_id)
```

The default counter is an offline UTF-8 byte estimate with framing overhead. It is
conservative for typical text BPE models, **not** a provider billing tokenizer.
For model-aware local counting, install `lethes[litellm]` and pass
`LiteLLMTokenCounter("gpt-4o")` as `counter=`. This calls the SDK's local
[`token_counter`](https://docs.litellm.ai/docs/completion/token_usage), including
tools and tool choice; it does not request a completion or a remote count endpoint.
Unknown deployment aliases require an explicit registered tokenizer model.
`response_format` is reserved as serialized schema text because this SDK API has
no schema parameter. Local counts remain estimates, including model fallbacks
and default image costs; the alpha focuses on text chat. For offline operation,
set `LITELLM_LOCAL_MODEL_COST_MAP=True` before importing the SDK and make the
chosen model's tokenizer resources available locally.

Alternatively install `lethes[tokenizers]` and pass
`TiktokenCounter("o200k_base")`, or supply a provider-specific `TokenCounter`.
The byte/tiktoken counters reject image/audio content rather than counting URLs.

Provider usage is persisted and preferred for the same model, request envelope
and assembled messages, even after restarting or switching local counters.
For a matching complete-message prefix, only its appended portion is estimated.
Edited/rebuilt layouts use local estimates with a conservative correction for
observed undercounting; a cheaper old request cannot discount unrelated text.
`result.input_tokens` reports the selected budget count (`estimated_tokens` is
its compatibility alias). `count_source` distinguishes `reported_usage`,
`reported_prefix_plus_estimate`, `usage_calibrated_estimate` and `local_estimate`;
`count_reference` identifies the supporting historical request when applicable.

Configure `CachePricing(uncached_per_million=..., cached_per_million=...)` to enable
economic rebuilds. Without prices, valid prefixes are reused until capacity
requires rebuilding. Prices are explicit; there is no guessed model-price table.

## Open WebUI

Build and install this checkout into the **Open WebUI backend environment**:

```bash
uv build
# Inside the Open WebUI backend environment/container:
pip install /path/to/lethes-2.0.0a1-py3-none-any.whl
# For the LiteLLM counting backend, install its extra instead:
pip install '/path/to/lethes-2.0.0a1-py3-none-any.whl[litellm]'
```

In Admin Panel → Functions, import/paste
[`src/external/open_webui_filter.py`](src/external/open_webui_filter.py), activate
it, attach it to the desired model, and select its toggle in the chat. The loader
is small because the package contains the implementation; this alpha must be
installed from the built wheel until published.

Set Valves:

- `state_path`: persistent SQLite path in the backend volume. Default:
  `/app/backend/data/lethes/state.sqlite3`. Processes sharing this file must run on
  the same host with SQLite locking support; separate hosts need a different store.
- `context_window`, `reply_reserve`, `safety_reserve`, `recent_tokens`,
  `minimum_recent_turns`: actual model capacity and protected recent dialogue.
- `input_price`, `cached_input_price`: USD per million input tokens, plus cache
  minimum, block size and TTL matching your provider. Leave prices unset to disable
  speculative economic rebuilding. Economics use estimates until calibrated.
- `tokenizer`: `bytes`, `litellm` (install the wheel with its `[litellm]` extra),
  or an installed tiktoken encoding. `tokenizer_model` can map a deployment alias
  to its actual registered model; otherwise LiteLLM uses workspace `base_model_id`
  or the requested model ID. Bind dated returned model IDs explicitly when needed.
- Optional embedding/query model, endpoint and key: enables semantic recall/query
  expansion. With no helper model, lexical recall remains fully offline. Helper
  failure falls back to lexical retrieval with a diagnostic. Embeddings and query
  results are bounded in-memory caches; original text remains authoritative.

`inlet()` leaves the full history intact and excludes internal title/query/tag
tasks. `request()` assembles after pipeline enrichment, and is idempotent across
native tool-loop hops, whether the host rebuilds the original history or reuses
an already compiled request. `stream()`/`outlet()` record available usage without
altering output. Per-request provider usage takes precedence over outlet totals;
multi-hop turn aggregates are saved but never used to count one request. Model
mismatches remain in the ledger and invalidate older calibration for that route.
Output tokens are excluded from input counts; reply capacity is reserved separately.
Missing emitters and fresh completed-callback metadata are handled.
Unscoped API calls without a user and chat identity pass through unchanged.

This targets Open WebUI versions with the documented `request()` hook. The official
[Filter guide](https://docs.openwebui.com/features/extensibility/plugin/functions/filter/)
and [reserved arguments](https://docs.openwebui.com/features/extensibility/plugin/development/reserved-args/)
were checked on **2026-11-06**. Provider normalization and workspace-model system
prompts can still be applied after `request()`. Other filters/RAG/system changes
can reduce cache hits. Only upstream usage establishes actual billed cache hits.
Do not also enable another history-trimming filter.

## Replay your own data

```bash
uv run python examples/replay.py --input chat.json --output /tmp/replay.csv
```

Accepted formats:

- An OpenAI-format message array, or `{"messages": [...]}`: replay each user-turn
  prefix in sequence.
- `{"histories": [[...], [...], ...]}`: replay complete history snapshots, allowing
  edits, rewinds and branch switches in the test data.

The replay harness makes **no main-model API calls**. It measures assembly,
capacity, branch reuse and potentially reusable prefixes, not answer quality or
real cache savings. Synthetic fixtures currently verify mechanics. Real roleplay
data and live Open WebUI/provider validation are still needed to evaluate implicit
mechanism recall, narrative continuity and realized costs.

See [design and economics](docs/design.md) for invariants and assumptions.

## Compatibility

v1 scoring algorithms, summary hierarchy, flags, Redis cache adapters and public
APIs were removed. Migrate to `ContextManager.prepare()`; existing v1 plugins and
observer integrations are not compatible. The new SQLite schema is separate from
v1 caches. `delete_scope(digest([user_id, chat_id]))` removes stored chat state and
attempts; chat-deletion event wiring is left to the host deployment.

MIT license.

## Novel text workload

```bash
uv run python examples/novel_replay.py --text /path/to/novel.txt \
  --output /tmp/novel-report
```

The loader preserves every character and groups neighboring lines into bounded
passages. Artificial `continue` turns exercise sequential context assembly; they
are explicitly **not** real user interactions or generated continuations.
The report validates every recalled passage against its original message revision,
checks budgets, and exercises rewind/edit/restore. It contains counts, source hashes
and positions, not the novel text. An optional `--probes probes.json` accepts a list
of `{"query": "...", "expected": "exact source substring"}` checks. These are
human-selected passage-recall checks, not a general narrative-quality benchmark.

For a native Open WebUI JSON export (including its message tree):

```bash
uv run python examples/export_replay.py --input /path/to/chat-export.json \
  --output /tmp/interactive-report
```

The importer follows `parentId`, enumerates retained branches and finishes on
`currentId`. It never concatenates sibling nodes or includes host user metadata
in model messages. Replay checks actual user-turn prefixes, stable overlays,
input/output edits and restoring the original path. Both workload scripts accept
`--probes`; token estimates default to `bytes`. `--tokenizer o200k_base` requires
`lethes[tokenizers]` **and the encoding vocabulary to be available locally** (or a
network that permits its initial download). To exercise the LiteLLM backend, use
`--tokenizer litellm --tokenizer-model gpt-4o` with its optional extra installed.
Offline replay has no real provider usage to calibrate against.
