# Alpha validation (2026-11-06)

## Automated checks

- 69 tests pass on Python 3.12; core coverage 94.07%.
- Ruff lint/format and mypy pass.
- Source distribution and wheel build; the installed wheel passes a separate
  environment smoke check. CI defines Python 3.11/3.12/3.13 jobs; only 3.12 was
  executed locally.
- Tests cover exact replay, reminder placement, content-versioned input/output
  edits, sibling/future exclusion, restart, concurrent identical requests, protected
  tool pairs, capacity, recall capacity, cycle cost enumeration, helper failures,
  API metadata/usage shapes, Open WebUI export branches and text-preserving loaders.

## Supplied workloads

The supplied data remains outside the repository. Scripts read the original local
files and write only metadata/positions to reports. No source passages, account
metadata or raw chat exports are committed.

| Workload | Size | Replay | Source/budget/path invariants | Selected passage probes |
|---|---|---|---|---|
| Novel | 247,668 characters | 279 artificial continuation steps | Pass | 5/6 |
| Open WebUI export | 42 message nodes, two paths; active path has 40 chat messages | 21 actual user-turn prefixes across paths | Pass, including input/output edit and restore | 5/5 |

Probe questions and expected original substrings were manually selected after
reading the supplied data. They are **not** a held-out recall benchmark or a
measure of generated-answer quality. The novel's remaining miss is an early
passage displaced by other plausible matches under the finite budget. Semantic
helpers were not enabled in these offline runs.

The runs use the default UTF-8 byte counter and a 32,768 context setting, with
2,048 reply and 1,024 safety reserve. This is a deliberately conservative local
text stress test, not actual model token accounting. The novel produced 135
assembly epochs over 279 steps: broad lexical matches and accumulation can cause
frequent capacity rebuilds. Stable-prefix invariants pass, but cost efficiency
under real tokenization and retrieval still needs tuning and measurement.

An attempted o200k vocabulary initialization was blocked by the environment's
network proxy (403). There was no local vocabulary cache. No token-count or billed
cache results from that tokenizer/provider are claimed.

## What the real data caught

1. Many background queries could outvote an explicit current query. Weighted
   fusion and direct-match priority now preserve the current question's influence.
2. Expanded neighbors could make a useful original message too large. Exact-hit
   fallback now preserves the hit and reports omitted neighbors.
3. Old reminders could fill the prefix and block a new direct-query match while
   the base payload still fit. A feasible fresh assembly can now rebuild for
   recall capacity. This corrected the interactive probes from 3/5 to 5/5.
4. Native exports contain branches and host/account fields. Import follows parent
   links, selects the active path last and removes host fields before assembly.

## Still unverified

No live Open WebUI server or model endpoint was called. Integration tests implement
shapes from the official current Filter documentation. Actual prompt normalization,
cache residency, cache-write charges, usage reporting, narrative continuity and
implicit-rule recall require a live provider and further conversation datasets.
The rewrite intentionally breaks v1 APIs and does not migrate its state.
