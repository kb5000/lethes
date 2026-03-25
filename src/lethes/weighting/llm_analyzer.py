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

import json
import logging
import re
from typing import TYPE_CHECKING, Any

import httpx

from ..utils.ids import cache_key_for_strings

if TYPE_CHECKING:
    from ..cache.base import CacheBackend
    from ..models.conversation import Conversation
    from ..models.message import Message

logger = logging.getLogger(__name__)

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
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._cache = cache
        self._max_msgs = max_messages_in_prompt
        self._truncate = content_truncate_chars
        self._client = httpx.AsyncClient(timeout=timeout)

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
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                try:
                    labels = json.loads(cached)
                    if not isinstance(labels, list):
                        labels = None
                except Exception:
                    pass

        if labels is None:
            labels = await self._call_llm(window, query)
            if labels is not None and self._cache:
                await self._cache.set(cache_key, json.dumps(labels), ttl=1800)

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
            logger.warning("LLMContextAnalyzer call failed: %s", exc)
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
            logger.debug("LLMContextAnalyzer: used regex fallback on %r", raw[:80])
            return self._fit_labels(extracted, expected_length)

        logger.debug("LLMContextAnalyzer: could not parse response: %r", raw[:120])
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
