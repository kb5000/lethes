"""
LLMContextAnalyzer — sub-agent that uses an LLM to score message importance.

The analyzer sends a compact conversation digest to any OpenAI-compatible
``/v1/chat/completions`` endpoint and asks the LLM to rate each message's
relevance to the current query.  Results are cached by conversation digest
and query hash so identical turns are free.

Typical usage — combine with :class:`~lethes.weighting.smart.SmartWeightingStrategy`
via :class:`~lethes.weighting.composite.CompositeWeightStrategy`::

    from lethes.weighting import CompositeWeightStrategy, SmartWeightingStrategy
    from lethes.weighting.llm_analyzer import LLMContextAnalyzer

    weighting = CompositeWeightStrategy([
        (SmartWeightingStrategy(), 0.4),
        (LLMContextAnalyzer(api_base=..., api_key=..., model="gpt-4o-mini"), 0.6),
    ])

Stand-alone usage (pure LLM scoring)::

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
from typing import TYPE_CHECKING, Any

import httpx

from ..utils.ids import cache_key_for_strings

if TYPE_CHECKING:
    from ..cache.base import CacheBackend
    from ..models.conversation import Conversation
    from ..models.message import Message

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "lethes:llm_score:"

_SYSTEM_PROMPT = """\
You are a conversation analyst.  Your task is to rate how important each \
message is for answering the current question.

Scoring guide:
  1.0 — Critical: directly relevant, must be kept
  0.7 — Helpful: provides useful context
  0.4 — Marginal: tangentially related
  0.1 — Weak: probably not needed
  0.0 — Irrelevant: unrelated to the current question

Rules:
- Tool-call intermediate messages (role "tool" or function results) should be \
scored relative to whether their result is still needed.
- Score the last user message at 1.0 (it is the current question).
- Respond ONLY with valid JSON in the exact format shown below.
  No markdown, no explanation.

Format: {"scores": [<float>, <float>, ...]}
The array must have exactly one value per message in the same order."""

_USER_PROMPT_TEMPLATE = """\
Current question: {query}

Conversation (oldest → newest, {n} messages):
{messages_text}

Return importance scores for all {n} messages."""


class LLMContextAnalyzer:
    """
    LLM-powered :class:`~lethes.weighting.base.WeightingStrategy`.

    Parameters
    ----------
    api_base:
        Base URL of an OpenAI-compatible API.
    api_key:
        Bearer token for the API.
    model:
        Chat completion model to use.  A cheap, fast model (e.g.
        ``"gpt-4o-mini"``) is recommended — the task is classification,
        not generation.
    cache:
        Optional :class:`~lethes.cache.base.CacheBackend`.  Caches LLM
        responses keyed by ``(model, query, all_message_texts)``.
    max_messages_in_prompt:
        Maximum messages to include in the analysis prompt.  If the
        conversation is longer, the oldest messages outside this window
        receive the ``default_score``.
    content_truncate_chars:
        Each message's text is truncated to this many characters before
        being sent to the LLM (to control prompt size).
    default_score:
        Score assigned to messages that are not sent to the LLM (too far
        back in history) or when the LLM call fails.
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
        default_score: float = 0.5,
        timeout: float = 30.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._cache = cache
        self._max_msgs = max_messages_in_prompt
        self._truncate = content_truncate_chars
        self._default_score = default_score
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

        # System messages are handled by constraints; score them neutrally.
        non_system = [m for m in messages if m.role != "system"]
        system_ids = {m.id for m in messages if m.role == "system"}

        if not non_system:
            return {m.id: 1.0 for m in messages}

        # Only analyze the most recent window; older messages get default score
        window = non_system[-self._max_msgs:]
        out_of_window_ids = {m.id for m in non_system[: -self._max_msgs]}

        # Try cache
        cache_key = _CACHE_PREFIX + cache_key_for_strings(
            self._model,
            query,
            *[self._format_msg(m) for m in window],
        )
        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                try:
                    parsed = json.loads(cached)
                    return self._build_result(
                        messages, window, parsed, out_of_window_ids, system_ids
                    )
                except Exception:
                    pass  # corrupt cache entry — re-query

        # Call LLM
        raw_scores = await self._call_llm(window, query)

        if raw_scores is not None and self._cache:
            await self._cache.set(cache_key, json.dumps(raw_scores), ttl=1800)

        return self._build_result(
            messages, window, raw_scores, out_of_window_ids, system_ids
        )

    def _build_result(
        self,
        all_messages: list["Message"],
        window: list["Message"],
        raw_scores: list[float] | None,
        out_of_window_ids: set[str],
        system_ids: set[str],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        window_map: dict[str, float] = {}
        if raw_scores and len(raw_scores) == len(window):
            for msg, s in zip(window, raw_scores):
                window_map[msg.id] = max(0.0, min(1.0, float(s)))

        for msg in all_messages:
            if msg.id in system_ids:
                result[msg.id] = 1.0
            elif msg.id in out_of_window_ids:
                result[msg.id] = self._default_score
            elif msg.id in window_map:
                result[msg.id] = window_map[msg.id]
            else:
                result[msg.id] = self._default_score
        return result

    async def _call_llm(
        self, window: list["Message"], query: str
    ) -> list[float] | None:
        messages_text = "\n".join(
            f"[{i + 1}] {m.role}: {self._format_msg(m)}"
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
                    "max_tokens": 256,
                },
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            return self._parse_scores(raw, expected_length=len(window))
        except Exception as exc:
            logger.warning("LLMContextAnalyzer call failed: %s", exc)
            return None

    def _format_msg(self, msg: "Message") -> str:
        from ..utils.content import get_text_content

        if msg.tool_calls:
            names = ", ".join(
                tc.get("function", {}).get("name", "?")
                for tc in msg.tool_calls
                if isinstance(tc, dict)
            )
            text = f"[calls: {names}]"
        else:
            text = get_text_content(msg.content)

        if len(text) > self._truncate:
            text = text[: self._truncate] + "…"
        return text

    def _parse_scores(self, raw: str, expected_length: int) -> list[float] | None:
        """Parse LLM JSON output; return None on any failure."""
        raw = raw.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(
                line for line in raw.splitlines()
                if not line.startswith("```")
            )
        try:
            data = json.loads(raw)
            scores = data.get("scores", [])
            if len(scores) == expected_length:
                return [float(s) for s in scores]
            # Truncate or pad to expected length
            if len(scores) > expected_length:
                return [float(s) for s in scores[:expected_length]]
            if len(scores) > 0:
                # Pad with default
                return [float(s) for s in scores] + [
                    self._default_score
                ] * (expected_length - len(scores))
        except Exception as exc:
            logger.debug("Failed to parse LLM score response: %s — %r", exc, raw)
        return None

    def name(self) -> str:
        return f"llm_analyzer({self._model})"
