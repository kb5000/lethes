# v2 design

## Authority and identity

Each invocation supplies the complete **current** Open WebUI path. It is the source
of truth. The manager stores assembly choices, not an inferred fictional world.
Canonical message payloads omit host IDs/timestamps/usage but preserve provider
extensions. The revision hash is H(parent revision, canonical payload). Identical
speech at different ancestry positions remains distinct; edited content with the
same host ID is a new revision.

A snapshot is reusable only when its head occurs on the supplied path. This also
protects fixed prefixes: a checkpoint made at turn 100 may contain a quoted turn
80 even if that quote is physically near the start of the outgoing payload. It
cannot be restored for a rewind to turn 30. On a nearby edit, an earlier valid
ancestor snapshot can still be reused. Prior branches remain independently
restorable; no mutable shared branch pointer exists.

Scopes isolate user/chat. Domains isolate model, request environment, config,
counter identity and retrieval namespace. Tool/schema property order is part of
the request fingerprint because providers may render it into a prompt. Custom
integrations should change the namespace when changing retrieval/serialization
semantics. SQLite snapshots are
first-writer-wins while feasible. Request attempts are separate, including retries
and cancelled attempts. The next supplied history determines which output was
actually adopted; outlet callbacks never advance a narrative path.

## Context cycles and recall

Within a cycle the continuous-tail start and existing reminder bytes/anchors are
immutable. New history extends the tail. New reminders are inserted before the
latest new user message; tool follow-ups defer new recall while preserving old
reminders and intact call/result structure. Initial/rebuilt cycles leave capacity
headroom. The recent-text soft budget also obeys the rebuild fill target so an
oversized setting cannot consume recall space and all append headroom. Protected
minimum recent turns may exceed that target. Capacity rebuilding reselects a
recent continuous tail and original historical passages. Current turns and
instruction messages are protected; an impossible budget raises `ContextOverflow`
rather than silently corrupting them.

Retrieval only sees the current visible path. Multiple recent original messages
supply queries; inserted reminders are not fed back as query text. Optional query
expansion sees recent text only and produces search strings, not stored facts.
Sparse and dense rankings are unioned with reciprocal rank fusion, then neighboring
message ranges are expanded. All positive lexical candidates are eligible; no
small fixed top-k truncation is applied. Budget selection deduplicates original
indices and merges adjacent spans. Reminders serialize exact original message
content, speaker, index and path revision inside a historical-data wrapper.

Each query first has a bounded **reference working set**, independent of free
space in the accumulated prefix. That working set is then reconciled against
the continuous original tail and existing reminders: only missing source indices
can be added. Resident passages consume the reference recall budget even when
their incremental insertion cost is zero. Otherwise the selector would skip
covered candidates as "free" and append the next batch of weaker matches on
every turn, repeatedly exhausting capacity for essentially the same query.

Optional candidates whose relevance is close receive a bounded 15% score bonus
when their focus is already resident. Required direct-query matches remain first;
substantially stronger new evidence can still outrank old optional evidence.
Explicit earliest/original-mention queries retain both the strongest match and
an early match with at least two-thirds of its lexical score. These are retrieval
ordering choices, not fictional-rule extraction. Candidate sets remain intact.

Coverage is checked by source index and content-versioned ancestry. Overlapping
ranges contribute only their missing portion, including when one part is already
present in continuous raw dialogue. Similar statements from distinct events or
revisions are not merged into a guessed fact. Replay validates that no source
index is quoted twice or quoted alongside its already-present continuous original.

Tool follow-up hops without a new user turn extend the saved layout without
calling retrieval again. Compatible existing snapshot/domain formats are reused
across this optimization; there is no forced cache reset. Growth measurements
survive rebuilding. Previously compacted histories still query omitted passages
even when their complete raw input would fit a new cold layout.

Oversized message-level candidates may not fit; `omitted` reports these ranges.
This alpha retrieves whole messages plus neighbors, not character-offset snippets.
No approach guarantees implicit recall without a relevant cue. Real datasets must
measure that separately. Failed optional helper calls retain lexical retrieval
and an explicit warning. Repeated embeddings and query expansions are cached.

Optional embedding windows cover the complete source/query with overlap; query
and document instructions are applied only to encoder inputs. Dense fusion uses
one best-window score per original message/query, preserving original citations
and avoiding extra votes for longer messages. Default windowing is disabled for
compatibility. Similarity floors are configurable for the chosen model. See
[small local models](local-models.md) for configuration and evaluation limits.

Economic rebuild candidates may not lose newly included current-request recall
coverage. Older in-cycle reminders can be reselected at rebuild. Candidate ranking
quality and budget coverage are not converted into invented monetary values.

## Average-cost renewal decision

Assume the conversation continues indefinitely. Let a rebuilt input start at B
estimated tokens, grow by g tokens/request, and remain feasible for T requests.
Cache eligibility includes the configured minimum/block size. Each cycle costs:

    C(T) = first request cost
           + sum(follow-up cached-prefix cost + new-input cost)
           + one-time rebuild overhead

The planner minimizes C(T)/T over capacity-feasible cycle lengths. It uses that
renewal rate r as the opportunity cost and compares each action's minimum excess
cost C_action(T)-r*T, for continuing the old prefix versus rebuilding now. Both
actions return to the same assumed renewal process. A margin avoids marginal
switches. Already paid costs do not enter the action comparison.

This is a **forecast model**, not a proof of global optimality for unknown future
story content. Growth is smoothed from observed input/reminder growth; it does not
predict user departure. Cache reuse follows configured provider pricing and
eligibility assumptions. TTL bounds which prior attempts count as potentially
reusable. Cold/missed input prices should include provider-specific cache-write
fees where applicable. Explicit-cache providers require correct provider-side
cache configuration; this library does not synthesize vendor cache breakpoints.

Candidates currently reuse cached query/embedding results but are assembled
locally. The policy exposes a rebuild-overhead parameter for deployments with
additional one-off charges; automatic helper billing calibration is not yet
implemented. In the default adapter, prices are unset and economic decisions are
disabled until configured. Capacity/validity handling remains active.

## Measurement and integration limits

The independent HTTP service hosts the observer in the same package/image. The
Open WebUI service Filter forwards complete current paths and per-provider usage;
the service owns counting, retrieval, layout persistence and dispatch observations.
The observer reads the same SQLite ledger through authenticated summary/detail
routes; there is no lossy in-memory log-ingestion dependency. Observer responses
exclude original message payloads and raw provider fields. Existing embedded
integration remains an alternative, sharing observation/status serialization.

`potential_cached_tokens` is a conservative message-boundary prefix estimate,
never a claim that a request reached the provider or that a cache is resident.
It can miss partial-message matches and provider-added prefixes. Attempts without
usage may have been cancelled. Only observed usage gives billed cache quantities.
Token counter estimates include framing heuristics; provider normalization can
change counts. Reply and safety reserves bound input capacity independently of
input accounting; an output reservation must not be added to input token usage.
Byte/tiktoken backends reserve external tool schemas separately. The optional
LiteLLM adapter binds the model, tools, tool choice and workspace prefix once per
request; `response_format` receives an extra text estimate. It uses local SDK
counting, with default image costs to avoid fetching image URLs, and rejects
unknown model mappings. It does not establish exact multimodal capacity.

Authoritative per-request provider usage has priority over local estimates:

1. An exact assembled message payload and matching model/envelope use measured
   total input tokens, including cache-read/write input but excluding output.
2. A full-message extension of a measured prefix uses that measured count plus
   the nonnegative local-count increment. This is explicitly a hybrid estimate.
3. Other layouts use local counting plus the largest recent positive undercount
   observed for the same model/envelope. Negative offsets never discount rebuilt
   or edited content merely because an unrelated request was cheaper.

Measurements are scoped to user/chat and model/request environment independently
of the local counter. Switching byte counting to LiteLLM does not erase usable
provider counts. Exact lookup is indexed; calibration uses the latest 16 samples.
New usage does not change the layout domain or reset the reminder cycle. A saved
layout that is now over budget is replaced with a feasible preferred layout using
compare-and-swap; prior layouts remain in an immutable archive and attempt ledger.
The exception is a capacity repair, not an economic rebuild on every observation.

`count_source` and `count_reference` expose measurement provenance. Counts explicitly
marked estimated, invalid counts and multi-hop turn aggregates remain telemetry
and cannot calibrate a single request. Per-request stream usage outranks outlet
totals. A mismatched returned model invalidates older measurements for that route;
request aliases and dated returned IDs require explicit model mapping. If no model
is returned, usage is associated with the recorded request model. Older alpha
records without explicit model metadata can only be backfilled within their original
strict layout domain. Upstream unreported routing or prompt changes still require
host configuration changes to invalidate the environment fingerprint.

Open WebUI's `request()` runs on every model hop, after inlet/RAG processing but
before provider normalization and workspace-model system prompt insertion. The
adapter rehydrates its original source when an already compiled request reappears,
and handles a reconstructed full-source tool hop. Lifecycle metadata stays JSON
serializable. Telemetry uses per-request correlation, including fresh metadata
from `/api/chat/completed`, and never modifies generated prose.

Each dispatch also stores an immutable observation under its request ID: selected
budget count/provenance, effective input limit, epoch/reason, source coverage and
omissions, potential prefix, warnings, economics and preparation latency. Provider
usage remains separate so later calibration cannot rewrite the dispatch decision.
INFO logs expose both records without original text, queries, vectors or secrets.
The native status description is one line and updates from estimated/historical
counts to authoritative current usage; aggregates or explicitly estimated usage
do not receive an actual-input label. Repeated usage and output-only updates avoid
duplicate status events. Missing/disconnected UI emitters do not fail generation.
Disabling status emission leaves the assembly domain unchanged. Status-history
folding remains a host UI preference.

The persistent SQLite path must be on a backend volume with working file locks.
Stored snapshots/attempts contain original chat text. Deployments must connect
chat deletion/retention to `delete_scope`; no automatic deletion event is registered
by the importable Filter. This alpha does not migrate v1 state.

Official integration sources checked 2026-11-06:

- https://docs.openwebui.com/features/extensibility/plugin/functions/filter/
- https://docs.openwebui.com/features/extensibility/plugin/development/reserved-args/
- https://docs.openwebui.com/features/extensibility/plugin/development/events/
- https://github.com/open-webui/open-webui/blob/main/src/lib/components/chat/Messages/ResponseMessage/StatusHistory.svelte
- https://docs.litellm.ai/docs/completion/token_usage
- https://docs.litellm.ai/docs/tokenize

## Coverage under a full prefix

The strongest match from each direct-current-query channel carries a local
coverage requirement, including its neighbor range. If accumulated reminders
leave no room for that range, a fresh assembly is tried. When it can restore that
coverage, rebuilding is a capacity operation (`recall_capacity`), even if the
old payload alone would still fit. Otherwise a full stale prefix could silently
block every new recall. Coverage is a per-request retrieval choice, not a stored
fictional rule. Economic comparisons apply when this capacity requirement is met.

Neighbor-expanded candidates that cannot fit can fall back to their exact hit
message. Omitted neighbors remain explicitly reported. Generic continuation
utterances are neither useful standalone queries nor retrieval centers, although
they can still accompany a matched original passage as neighboring context.
