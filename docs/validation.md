# Alpha validation (2026-11-06)

## Automated checks

- 83 tests pass on Python 3.12; core coverage 94.14%.
- Ruff lint/format and mypy pass.
- Source distribution and wheel build; installation is checked in a separate
  environment. CI tests Python 3.11/3.12/3.13; only 3.12 is executed locally.
- Tests cover exact replay, fixed reminder placement, source/version coverage,
  overlap with recent raw text, preventing weaker-candidate refill, preserving
  edited source versions, bounded resident preference, temporal recall, restoring
  compatible saved layouts, input/output edits, sibling/future exclusion, restart,
  concurrency, tool hops without new retrieval, capacities, cycle economics,
  optional helper failure, Open WebUI lifecycle shapes, exports and text loaders.

## Supplied workloads

Original files remain outside the repository. Scripts read them locally and write
only hashes, counts and positions to reports. No passages, account metadata or
raw chat exports are committed.

| Workload | Size | Replay | Source/budget/path invariants | Selected passage probes |
|---|---|---|---|---|
| Novel | 247,668 characters | 279 artificial continuation steps | Pass; zero repeated source citations | 6/6 |
| Open WebUI export | 42 nodes, two paths; active path has 40 chat messages | 21 actual user-turn prefixes | Pass; zero repeated source citations; edit/restore verified | 5/5 |

Questions and expected substrings were manually selected after reading the input.
These checks are **not** held-out recall benchmarks or generated-answer quality
measurements. Semantic helpers were disabled. Source identity is checked against
message indices and path revisions; this is not automatic semantic deduplication
of different fictional events.

## Optimization comparison

Before is commit `285b7bd` (the initial v2 alpha), after is the reference-working-set
and resident-reuse update. Both use identical input files, default UTF-8 byte
estimates, a 32,768 context setting, 2,048 reply reserve and 1,024 safety reserve.
Prices are unset, so these are capacity/recall experiments, not economic-policy
or provider billing measurements.

| Metric | Novel before | Novel after | Interactive before | Interactive after |
|---|---:|---:|---:|---:|
| Assembly cycles, including initial | 135 | 102 | 11 | 9 |
| Estimated total input | 6,864,870 | 6,903,828 | 417,479 | 420,568 |
| Potential reusable prefix units | 3,098,969 | 4,051,707 | 150,495 | 197,017 |
| Estimated input outside reusable prefixes | 3,765,901 | 2,852,121 | 266,984 | 223,551 |
| Potential prefix share | 45.14% | 58.69% | 36.05% | 46.85% |
| Selected passage probes | 5/6 | 6/6 | 5/5 | 5/5 |

Novel cycles decreased 24.4%, and estimated input outside reusable prefixes
decreased 24.3%; interactive figures decreased 18.2% and 16.3%, respectively.
Total supplied input grew slightly, so the gain is not simply less context sent.
A reusable prefix is only a local message-boundary match against recent attempts.
It does not prove provider cache residency or billed savings. Counts use byte
estimates, not actual model tokens, and the novel still rebuilds frequently.

An o200k vocabulary initialization attempt was blocked by the environment proxy
(403); no local vocabulary cache was available. No results from that tokenizer
or a live model are claimed.

## Bugs caught by workloads and regression tests

1. Many background queries could outvote an explicit current query. Weighted
   fusion, direct-match priority and explicit temporal intent now preserve the
   requested original evidence.
2. Neighbor expansion could make a useful original message too large. Exact-hit
   fallback preserves the hit and reports omitted neighbors.
3. Old reminders could block a new direct-query match even when the base payload
   fit. A feasible fresh assembly can rebuild for recall capacity.
4. Resident evidence was skipped without consuming the query budget, causing
   repeated insertion of weaker candidates. Reference planning now precedes
   missing-only insertion, with a bounded preference for resident optional matches.
5. Source indices already in continuous dialogue could become retrieval candidates
   as the reference recent window moved; they must still not be quoted again.
6. Economically compacted histories could fit the cold full-input threshold and
   accidentally stop searching omitted history. Reference planning now keeps
   retrieval active for those layouts. Compatible persisted layouts remain reusable.
7. Native exports contain branch and account fields. Import follows parent links
   and removes host fields rather than concatenating sibling nodes.

## Still unverified

No live Open WebUI server or main-model endpoint was called. Integration tests use
shapes from the official current Filter documentation. Actual prompt normalization,
cache residency, write charges, billing, narrative continuity and implicit-rule
recall require a live provider and broader datasets. The rewrite breaks v1 APIs
and does not migrate v1 state.
