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
counter identity and retrieval namespace. Custom integrations should change the
namespace when changing retrieval/serialization semantics. SQLite snapshots are
insert-only, first-writer-wins. Request attempts are separate, including retries
and cancelled attempts. The next supplied history determines which output was
actually adopted; outlet callbacks never advance a narrative path.

## Context cycles and recall

Within a cycle the continuous-tail start and existing reminder bytes/anchors are
immutable. New history extends the tail. New reminders are inserted before the
latest new user message; tool follow-ups defer new recall while preserving old
reminders and intact call/result structure. Initial/rebuilt cycles leave capacity
headroom. Capacity rebuilding reselects a recent continuous tail and original
historical passages. Current turns and instruction messages are protected; an
impossible budget raises `ContextOverflow` rather than silently corrupting them.

Retrieval only sees the current visible path. Multiple recent original messages
supply queries; inserted reminders are not fed back as query text. Optional query
expansion sees recent text only and produces search strings, not stored facts.
Sparse and dense rankings are unioned with reciprocal rank fusion, then neighboring
message ranges are expanded. All positive lexical candidates are eligible; no
small fixed top-k truncation is applied. Budget selection deduplicates original
indices and merges adjacent spans. Reminders serialize exact original message
content, speaker, index and path revision inside a historical-data wrapper.

Oversized message-level candidates may not fit; `omitted` reports these ranges.
This alpha retrieves whole messages plus neighbors, not character-offset snippets.
No approach guarantees implicit recall without a relevant cue. Real datasets must
measure that separately. Failed optional helper calls retain lexical retrieval
and an explicit warning. Repeated embeddings and query expansions are cached.

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

`potential_cached_tokens` is a conservative message-boundary prefix estimate,
never a claim that a request reached the provider or that a cache is resident.
It can miss partial-message matches and provider-added prefixes. Attempts without
usage may have been cancelled. Only observed usage gives billed cache quantities.
Token counter estimates include framing heuristics; provider normalization can
change counts. Reply, safety and tool-schema reserves bound local text assembly.
Unsupported multimodal content requires a custom counter.

Open WebUI's `request()` runs on every model hop, after inlet/RAG processing but
before provider normalization and workspace-model system prompt insertion. The
adapter rehydrates its original source when an already compiled request reappears,
and handles a reconstructed full-source tool hop. Lifecycle metadata stays JSON
serializable. Telemetry uses per-request correlation, including fresh metadata
from `/api/chat/completed`, and never modifies generated prose.

The persistent SQLite path must be on a backend volume with working file locks.
Stored snapshots/attempts contain original chat text. Deployments must connect
chat deletion/retention to `delete_scope`; no automatic deletion event is registered
by the importable Filter. This alpha does not migrate v1 state.

Official integration sources checked 2026-11-06:

- https://docs.openwebui.com/features/extensibility/plugin/functions/filter/
- https://docs.openwebui.com/features/extensibility/plugin/development/reserved-args/

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
