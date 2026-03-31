"""
LLMContextAnalyzer — sub-agent that classifies message retention priority.

Instead of asking the LLM to produce calibrated floating-point scores
(which it is poor at), the analyzer presents a **5-label classification**
task.  The LLM picks one of five discrete categories per message; the
categories are then mapped to weights internally.

Classification labels
---------------------
K  Keep    — Critical context, directly needed to answer the question.
H  Helpful — Relevant background, useful to include.
M  Maybe   — Tangentially related; include if budget allows.
S  Skip    — Probably not needed; old, off-topic, or superseded.
D  Drop    — Clearly irrelevant to the current question.

Weight mapping: K=1.0, H=0.75, M=0.5, S=0.25, D=0.05

Two-phase entry logic (optional)
---------------------------------
When ``use_entry_logic=True`` the analyzer runs a mini agentic loop:

1. **Phase 1 — Overview**: present a compact digest: all topic-cluster
   summaries, auto-expanded keyword-matching clusters and the most-recent
   cluster, plus the last 3 messages as user context.

2. **Phase 2 — Expansion** (optional): the LLM may call
   ``expand_topic(topic_id)`` up to ``max_expansions`` times to fetch the
   full message list for any cluster it wants to examine in detail.

3. **Classification**: once the LLM stops calling tools it outputs
   ``{"labels": [...]}`` as in the standard mode.

Typical usage — combine with :class:`~lethes.weighting.smart.SmartWeightingStrategy`
via :class:`~lethes.weighting.composite.CompositeWeightStrategy`::

    from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
    from lethes.weighting.llm_analyzer import LLMContextAnalyzer

    weighting = CompositeWeightStrategy([
        (SmartWeightingStrategy(), 0.4),
        (LLMContextAnalyzer(api_base=..., api_key=..., model="gpt-4o-mini"), 0.6),
    ])

Stand-alone usage::

    weighting = LLMContextAnalyzer(
        api_base="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini",
        cache=RedisCache.from_url("redis://localhost"),
    )
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
from typing import TYPE_CHECKING, Any

import httpx

from ..observability import get_logger
from ..utils.ids import cache_key_for_strings

if TYPE_CHECKING:
    from ..cache.base import CacheBackend
    from ..models.conversation import Conversation
    from ..models.message import Message

logger = get_logger(__name__)

_CACHE_PREFIX = "lethes:llm_label:"

# ── Label definitions ──────────────────────────────────────────────────────────

#: Canonical label set (uppercase single letters).
LABELS = ("K", "H", "M", "S", "D")

#: Float weight each label maps to.
LABEL_WEIGHTS: dict[str, float] = {
    "K": 1.00,   # Keep    — critical
    "H": 0.75,   # Helpful — useful background
    "M": 0.50,   # Maybe   — tangential
    "S": 0.25,   # Skip    — unlikely to help
    "D": 0.05,   # Drop    — clearly irrelevant
}

#: Label used for messages outside the analysis window or on LLM failure.
DEFAULT_LABEL = "M"

# ── Keyword extraction helpers ─────────────────────────────────────────────────

_STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "its", "may", "new", "now", "old", "see", "two", "who", "boy",
    "did", "man", "end", "put", "say", "she", "too", "use", "way", "what",
    "when", "with", "have", "this", "that", "from", "they", "will", "been",
    "more", "also", "into", "then", "than", "some", "just", "like", "time",
    "your", "very", "make", "over", "such", "take", "well", "even", "back",
    "good", "know", "most", "tell", "come", "here", "only", "work", "both",
    "life", "many", "need", "want", "long", "down", "look", "does", "much",
    "after", "which", "there", "first", "where", "could", "other", "these",
    "thing", "think", "would", "about", "their", "being", "every", "often",
    "those", "while", "under", "again", "still", "never", "right", "might",
    "should", "through", "another", "because", "before", "between", "without",
    "already", "always", "around", "different", "following", "however",
    "important", "number", "place", "point", "possible", "system", "world",
    "tool", "call", "result", "function",  # suppress lethes-internal tokens
})


def _extract_keywords(text: str, top_n: int = 8) -> list[str]:
    """Extract the top-*top_n* keywords from *text* by frequency, skipping stop words."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    counts: dict[str, int] = {}
    for w in words:
        if w not in _STOP_WORDS:
            counts[w] = counts.get(w, 0) + 1
    return sorted(counts, key=lambda w: -counts[w])[:top_n]


def _jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


# ── Topic cluster dataclass ────────────────────────────────────────────────────

@dataclasses.dataclass
class _TopicCluster:
    topic_id: str          # e.g. "topic_0"
    keywords: list[str]    # representative keywords
    indices: list[int]     # 0-based indices into the window list


# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You classify conversation messages by how important they are for answering \
the current question.  Use exactly one label per message:

  K  Keep    — Critical. Directly answers or is essential to the question.
  H  Helpful — Useful background context; worth including.
  M  Maybe   — Tangentially related; include only if budget allows.
  S  Skip    — Unlikely to help; old context, off-topic, or low-value.
  D  Drop    — Irrelevant to the current question.

Rules:
- The LAST message in the list is always the current question — label it K.
- Tool-call pairs (function call + its result) should share the same label: \
  keep them if the result is still relevant, skip both if superseded.
- Pinned or system messages are not shown; only chat messages appear.
- Output ONLY valid JSON, no markdown, no explanation.

Format: {"labels": ["K", "H", "M", ...]}
The array must have exactly one label per message, in the same order."""

_USER_PROMPT_TEMPLATE = """\
Current question: {query}

Messages (oldest → newest, {n} total):
{messages_text}

Classify all {n} messages."""

# ── Entry-logic prompts & tool definition ─────────────────────────────────────

_ENTRY_SYSTEM_PROMPT = """\
You classify conversation messages by retention priority.

You will receive a compact OVERVIEW of the conversation organised by topic \
clusters.  Use the expand_topic tool to examine clusters in detail before \
deciding.

Labels (one per message, oldest → newest):
  K  Keep    — Critical. Directly needed for the current question.
  H  Helpful — Relevant background worth including.
  M  Maybe   — Tangentially related; include if budget allows.
  S  Skip    — Unlikely to help; old, off-topic, or superseded.
  D  Drop    — Clearly irrelevant.

Rules:
- The LAST message is always the current question — label it K.
- Tool-call pairs share the same label.
- When done examining, output ONLY valid JSON, no markdown, no explanation.

Format: {"labels": ["K", "H", ...]}  (exactly one label per message)"""

_EXPAND_TOPIC_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "expand_topic",
        "description": (
            "Fetch the full message content for a topic cluster. "
            "Use this to examine a cluster in detail before classifying."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic_id": {
                    "type": "string",
                    "description": "The cluster ID shown in the overview (e.g. 'topic_0').",
                }
            },
            "required": ["topic_id"],
        },
    },
}


class LLMContextAnalyzer:
    """
    LLM-powered :class:`~lethes.weighting.base.WeightingStrategy` that
    classifies each message into one of five retention categories.

    Parameters
    ----------
    api_base:
        Base URL of an OpenAI-compatible API.
    api_key:
        Bearer token for the API.
    model:
        Chat completion model.  A fast, cheap model (e.g. ``"gpt-4o-mini"``)
        works well — this is pure classification, not generation.
    cache:
        Optional :class:`~lethes.cache.base.CacheBackend`.  Responses are
        cached by ``(model, query, window_message_texts)``; 30-minute TTL.
    max_messages_in_prompt:
        Sliding window size.  Messages older than this receive
        :data:`DEFAULT_LABEL` (``"M"`` → 0.5) without LLM cost.
    content_truncate_chars:
        Each message snippet is truncated to this many characters to keep
        the prompt compact.
    timeout:
        HTTP timeout in seconds.
    use_entry_logic:
        When ``True``, use the two-phase agentic overview + expand loop
        instead of presenting all messages at once.
    max_expansions:
        Maximum number of ``expand_topic`` tool calls allowed per request.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        cache: "CacheBackend | None" = None,
        max_messages_in_prompt: int = 24,
        content_truncate_chars: int = 200,
        timeout: float = 30.0,
        use_entry_logic: bool = False,
        max_expansions: int = 3,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._cache = cache
        self._max_msgs = max_messages_in_prompt
        self._truncate = content_truncate_chars
        self._client = httpx.AsyncClient(timeout=timeout)
        self._use_entry_logic = use_entry_logic
        self._max_expansions = max_expansions

    async def score(
        self,
        messages: list["Message"],
        query: str,
        conversation: "Conversation",
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if not messages:
            return {}

        # System messages are always kept by constraints; treat them as K.
        non_system = [m for m in messages if m.role != "system"]
        system_ids = {m.id for m in messages if m.role == "system"}

        if not non_system:
            return {m.id: 1.0 for m in messages}

        # Sliding window: messages outside it receive the default label
        window = non_system[-self._max_msgs :]
        out_of_window_ids = {m.id for m in non_system[: -self._max_msgs]}

        # Cache lookup
        cache_key = _CACHE_PREFIX + cache_key_for_strings(
            self._model,
            query,
            *[self._format_snippet(m) for m in window],
        )
        labels: list[str] | None = None
        cache_hit = False
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                try:
                    labels = json.loads(cached)
                    if not isinstance(labels, list):
                        labels = None
                    else:
                        cache_hit = True
                except Exception:
                    pass

        if labels is None:
            t0 = time.perf_counter()
            if self._use_entry_logic:
                labels = await self._call_llm_with_entry(window, query)
            else:
                labels = await self._call_llm(window, query)
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            if labels is not None:
                label_dist = {k: labels.count(k) for k in ("K", "H", "M", "S", "D") if labels.count(k)}
                logger.debug("llm_analyzer.done",
                    model=self._model,
                    n_messages=len(window),
                    label_dist=label_dist,
                    elapsed_ms=elapsed_ms,
                    entry_logic=self._use_entry_logic,
                    message_labels={m.id: label for m, label in zip(window, labels)},
                )
                if self._cache:
                    await self._cache.set(cache_key, json.dumps(labels), ttl=1800)
            else:
                logger.warning("llm_analyzer.failed",
                    model=self._model,
                    n_messages=len(window),
                    elapsed_ms=elapsed_ms,
                )
        else:
            logger.debug("llm_analyzer.cache_hit", model=self._model, n_messages=len(window))

        return self._build_result(
            messages, window, labels, out_of_window_ids, system_ids
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _build_result(
        self,
        all_messages: list["Message"],
        window: list["Message"],
        labels: list[str] | None,
        out_of_window_ids: set[str],
        system_ids: set[str],
    ) -> dict[str, float]:
        window_map: dict[str, float] = {}
        if labels and len(labels) == len(window):
            for msg, label in zip(window, labels):
                window_map[msg.id] = LABEL_WEIGHTS.get(label.upper(), LABEL_WEIGHTS[DEFAULT_LABEL])

        default_weight = LABEL_WEIGHTS[DEFAULT_LABEL]
        result: dict[str, float] = {}
        for msg in all_messages:
            if msg.id in system_ids:
                result[msg.id] = 1.0
            elif msg.id in out_of_window_ids:
                result[msg.id] = default_weight
            elif msg.id in window_map:
                result[msg.id] = window_map[msg.id]
            else:
                result[msg.id] = default_weight
        return result

    async def _call_llm(
        self, window: list["Message"], query: str
    ) -> list[str] | None:
        messages_text = "\n".join(
            f"[{i + 1}] {m.role}: {self._format_snippet(m)}"
            for i, m in enumerate(window)
        )
        user_content = _USER_PROMPT_TEMPLATE.format(
            query=query[:500] if query else "(no query)",
            n=len(window),
            messages_text=messages_text,
        )
        try:
            resp = await self._client.post(
                f"{self._api_base}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    "temperature": 0.0,
                    "max_tokens": max(64, len(window) * 4),
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._parse_labels(raw, expected_length=len(window))
        except Exception as exc:
            logger.warning("llm_analyzer.call_failed", model=self._model, error=str(exc))
            return None

    # ── Entry logic ────────────────────────────────────────────────────────

    def _cluster_messages(self, window: list["Message"]) -> list[_TopicCluster]:
        """
        Group *window* messages into topic clusters using keyword-overlap
        similarity.  Consecutive messages with Jaccard similarity below 0.08
        (and current cluster size ≥ 2) start a new cluster.
        """
        if not window:
            return []

        def kw_set(idx: int) -> set[str]:
            return set(_extract_keywords(self._format_snippet(window[idx]), top_n=10))

        cluster_indices: list[int] = [0]
        cluster_kw: set[str] = kw_set(0)
        all_clusters: list[_TopicCluster] = []
        cluster_count = 0

        def _flush() -> None:
            nonlocal cluster_count
            text = " ".join(self._format_snippet(window[i]) for i in cluster_indices)
            kws = _extract_keywords(text, top_n=6)
            all_clusters.append(_TopicCluster(
                topic_id=f"topic_{cluster_count}",
                keywords=kws,
                indices=list(cluster_indices),
            ))
            cluster_count += 1

        for i in range(1, len(window)):
            msg_kw = kw_set(i)
            sim = _jaccard(cluster_kw, msg_kw)
            # Start a new cluster when overlap drops and current cluster is large enough
            if sim < 0.08 and len(cluster_indices) >= 2:
                _flush()
                cluster_indices = [i]
                cluster_kw = msg_kw
            else:
                cluster_indices.append(i)
                cluster_kw |= msg_kw

        if cluster_indices:
            _flush()

        return all_clusters

    def _build_overview(
        self,
        window: list["Message"],
        query: str,
        clusters: list[_TopicCluster],
        auto_expanded_indices: set[int],
    ) -> str:
        """Build the compact overview prompt for Phase 1."""
        lines: list[str] = [
            f"CONVERSATION OVERVIEW — {len(window)} messages, "
            f"{len(clusters)} topic cluster(s)",
            f"Current question: {(query[:300] if query else '(no query)')}",
            "",
            "── TOPIC CLUSTERS ──",
        ]
        for c in clusters:
            kw_str = ", ".join(c.keywords) if c.keywords else "(no keywords)"
            span = (
                f"msg {c.indices[0]+1}"
                if len(c.indices) == 1
                else f"msgs {c.indices[0]+1}–{c.indices[-1]+1}"
            )
            lines.append(
                f"  [{c.topic_id}]  {kw_str}  ({len(c.indices)} messages, {span})"
            )

        # Auto-expanded section
        lines += ["", "── AUTO-EXPANDED MESSAGES ──"]
        expanded_sorted = sorted(auto_expanded_indices)
        if expanded_sorted:
            for i in expanded_sorted:
                m = window[i]
                lines.append(f"  [{i+1}] {m.role}: {self._format_snippet(m)}")
        else:
            lines.append("  (none)")

        # Recent messages not already shown
        recent_start = max(0, len(window) - 3)
        recent_extra = [
            i for i in range(recent_start, len(window))
            if i not in auto_expanded_indices
        ]
        if recent_extra:
            lines += ["", "── RECENT CONTEXT ──"]
            for i in recent_extra:
                m = window[i]
                lines.append(f"  [{i+1}] {m.role}: {self._format_snippet(m)}")

        lines += [
            "",
            "Use expand_topic(topic_id) to examine any cluster in full detail.",
            f"Then output {{\"labels\": [...]}} with exactly {len(window)} labels "
            "(one per message, oldest → newest).",
        ]
        return "\n".join(lines)

    async def _call_llm_with_entry(
        self, window: list["Message"], query: str
    ) -> list[str] | None:
        """
        Two-phase agentic loop:

        1. Send compact overview + ``expand_topic`` tool.
        2. Handle up to ``max_expansions`` tool calls, each time returning
           the full message list for the requested cluster.
        3. Parse the final ``{"labels": [...]}`` response.
        """
        clusters = self._cluster_messages(window)
        cluster_by_id: dict[str, _TopicCluster] = {c.topic_id: c for c in clusters}

        # Auto-expand: keyword-matching clusters + most recent cluster + last 3 messages
        query_kw = set(_extract_keywords(query, top_n=10)) if query else set()
        auto_expanded: set[int] = set()

        for c in clusters:
            cluster_text = " ".join(self._format_snippet(window[i]) for i in c.indices)
            cluster_kw = set(_extract_keywords(cluster_text, top_n=10))
            if query_kw & cluster_kw:
                auto_expanded.update(c.indices)

        if clusters:  # always expand most recent cluster
            auto_expanded.update(clusters[-1].indices)

        for i in range(max(0, len(window) - 3), len(window)):  # last 3 messages
            auto_expanded.add(i)

        overview = self._build_overview(window, query, clusters, auto_expanded)

        chat_messages: list[dict[str, Any]] = [
            {"role": "system", "content": _ENTRY_SYSTEM_PROMPT},
            {"role": "user", "content": overview},
        ]

        for iteration in range(self._max_expansions + 1):
            try:
                payload: dict[str, Any] = {
                    "model": self._model,
                    "messages": chat_messages,
                    "temperature": 0.0,
                    "max_tokens": max(128, len(window) * 8),
                }
                # Offer the tool on all but the final forced-classification pass
                if iteration < self._max_expansions:
                    payload["tools"] = [_EXPAND_TOPIC_TOOL]
                    payload["tool_choice"] = "auto"

                resp = await self._client.post(
                    f"{self._api_base}/chat/completions",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json=payload,
                )
                resp.raise_for_status()
                choice = resp.json()["choices"][0]
                assistant_msg = choice["message"]
                tool_calls = assistant_msg.get("tool_calls") or []

                if tool_calls:
                    chat_messages.append(assistant_msg)
                    for tc in tool_calls:
                        tc_id = tc.get("id", "tc")
                        fn_args_raw = tc.get("function", {}).get("arguments", "{}")
                        try:
                            fn_args = json.loads(fn_args_raw)
                        except json.JSONDecodeError:
                            fn_args = {}
                        topic_id = fn_args.get("topic_id", "")
                        cluster = cluster_by_id.get(topic_id)
                        if cluster:
                            expanded_lines = "\n".join(
                                f"  [{i+1}] {window[i].role}: {self._format_snippet(window[i])}"
                                for i in cluster.indices
                            )
                            tool_result = (
                                f"Topic {topic_id} ({len(cluster.indices)} messages):\n"
                                f"{expanded_lines}"
                            )
                        else:
                            available = ", ".join(cluster_by_id) or "(none)"
                            tool_result = (
                                f"Unknown topic_id '{topic_id}'. "
                                f"Available: {available}"
                            )
                        chat_messages.append({
                            "role": "tool",
                            "tool_call_id": tc_id,
                            "content": tool_result,
                        })
                    continue  # next iteration

                # No tool calls — expect the labels JSON
                raw = assistant_msg.get("content") or ""
                return self._parse_labels(raw, expected_length=len(window))

            except Exception as exc:
                logger.warning("llm_analyzer.entry_call_failed",
                    model=self._model, iteration=iteration, error=str(exc),
                )
                return None

        return None

    def _format_snippet(self, msg: "Message") -> str:
        from ..utils.content import get_text_content

        if msg.tool_calls:
            names = ", ".join(
                tc.get("function", {}).get("name", "?")
                for tc in msg.tool_calls
                if isinstance(tc, dict)
            )
            text = f"[tool call: {names}]"
        elif msg.role == "tool":
            text = f"[tool result] {get_text_content(msg.content)}"
        else:
            text = get_text_content(msg.content)

        if len(text) > self._truncate:
            text = text[: self._truncate] + "…"
        return text

    def _parse_labels(self, raw: str, expected_length: int) -> list[str] | None:
        """
        Parse the LLM's JSON response into a list of label strings.

        Accepts:
        * ``{"labels": ["K", "H", "M", ...]}``  — primary format
        * A bare JSON array: ``["K", "H", "M", ...]``
        * A compact string of letters (fallback): ``"KHMSD"``

        Returns ``None`` if parsing fails completely.
        """
        raw = raw.strip()
        # Strip markdown fences
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines() if not line.startswith("```")
            ).strip()

        # Try JSON object first
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                labels = data.get("labels", [])
            elif isinstance(data, list):
                labels = data
            else:
                labels = []
            labels = [str(l).strip().upper() for l in labels if str(l).strip()]
            if labels:
                return self._fit_labels(labels, expected_length)
        except json.JSONDecodeError:
            pass

        # Fallback: extract any K/H/M/S/D letters from the raw text
        extracted = re.findall(r"\b([KHMSD])\b", raw.upper())
        if extracted:
            logger.debug("llm_analyzer.parse_fallback", raw_preview=raw[:80])
            return self._fit_labels(extracted, expected_length)

        logger.debug("llm_analyzer.parse_failed", raw_preview=raw[:120])
        return None

    @staticmethod
    def _fit_labels(labels: list[str], expected_length: int) -> list[str]:
        """Trim or pad a label list to exactly *expected_length* entries."""
        valid = [l if l in LABELS else DEFAULT_LABEL for l in labels]
        if len(valid) >= expected_length:
            return valid[:expected_length]
        # Pad short responses with the default label
        return valid + [DEFAULT_LABEL] * (expected_length - len(valid))

    def name(self) -> str:
        return f"llm_classifier({self._model})"
