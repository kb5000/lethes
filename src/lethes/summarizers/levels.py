"""
Multi-level summarisation: turn → segment → conversation.

Each level builds on the previous one, enabling progressive compression
of long conversation histories without losing coherence.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..cache.base import CacheBackend
    from ..models.message import Message
    from .base import Summarizer

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "lethes:summary:"


class TurnSummarizer:
    """
    Summarise a single user+assistant turn pair.

    Checks the cache before calling the backend summariser.

    Parameters
    ----------
    backend:
        The :class:`~lethes.summarizers.base.Summarizer` to call on cache miss.
    cache:
        Optional :class:`~lethes.cache.base.CacheBackend` for storing results.
    target_ratio:
        Compression target passed to the backend.
    cache_ttl:
        Cache TTL in seconds (default: 24 h).
    """

    def __init__(
        self,
        backend: "Summarizer",
        cache: "CacheBackend | None" = None,
        target_ratio: float = 0.3,
        cache_ttl: int = 86400,
    ) -> None:
        self._backend = backend
        self._cache = cache
        self._target_ratio = target_ratio
        self._cache_ttl = cache_ttl

    async def summarize_turn(
        self,
        messages: list["Message"],
        context: list["Message"] | None = None,
    ) -> tuple[str, str]:
        """
        Summarise *messages* (a user+assistant pair or any contiguous slice).

        Returns
        -------
        tuple[str, str]
            ``(role, summary_text)`` where *role* is the role of the last
            message in the turn.
        """
        from ..utils.ids import cache_key_for_messages

        if not messages:
            return "user", ""

        cache_key = _CACHE_PREFIX + cache_key_for_messages(messages)

        if self._cache:
            cached = await self._cache.get(cache_key)
            if cached:
                data = json.loads(cached)
                logger.debug("Cache hit for turn summary %s", cache_key[:12])
                return data["role"], data["text"]

        text = await self._backend.summarize(
            messages,
            target_ratio=self._target_ratio,
            context_messages=context,
        )
        role = messages[-1].role

        if self._cache:
            await self._cache.set(
                cache_key,
                json.dumps({"role": role, "text": text}),
                ttl=self._cache_ttl,
            )

        return role, text


class SegmentSummarizer:
    """
    Summarise a topic segment — a contiguous cluster of turns about
    one subject.

    Two-pass: first each turn is summarised individually, then the turn
    summaries are compressed into a single segment summary.
    """

    def __init__(
        self,
        turn_summarizer: TurnSummarizer,
        backend: "Summarizer",
        target_ratio: float = 0.5,
    ) -> None:
        self._turn_sum = turn_summarizer
        self._backend = backend
        self._target_ratio = target_ratio

    async def summarize_segment(
        self,
        turns: list[list["Message"]],
        context: list["Message"] | None = None,
    ) -> str:
        """
        Parameters
        ----------
        turns:
            A list of turns, each turn being a list of messages
            (e.g. ``[[user_msg, assistant_msg], [...]]``).
        context:
            Prior context passed through to the backend.
        """
        if not turns:
            return ""

        # Summarise each turn concurrently
        tasks = [self._turn_sum.summarize_turn(turn, context=context) for turn in turns]
        turn_results: list[tuple[str, str]] = await asyncio.gather(*tasks)  # type: ignore[assignment]

        # Build synthetic messages from turn summaries for the second pass
        from ..models.message import Message

        summary_messages = [
            Message(role=role, content=text)
            for role, text in turn_results
            if text and text != "-"
        ]

        if not summary_messages:
            return ""

        return await self._backend.summarize(
            summary_messages,
            target_ratio=self._target_ratio,
            context_messages=context,
        )


class ConversationSummarizer:
    """
    High-level summary of the entire conversation history.

    Splits the conversation into segments, summarises each, then produces
    a final overall summary from the segment summaries.
    """

    def __init__(
        self,
        segment_summarizer: SegmentSummarizer,
        backend: "Summarizer",
        segment_size: int = 10,
        target_ratio: float = 0.2,
    ) -> None:
        self._seg_sum = segment_summarizer
        self._backend = backend
        self._segment_size = segment_size
        self._target_ratio = target_ratio

    async def summarize_conversation(
        self,
        messages: list["Message"],
    ) -> str:
        """Return a high-level summary of all *messages*."""
        if not messages:
            return ""

        # Split into segments of `segment_size` turns
        segments = _chunk(messages, self._segment_size)
        turns_per_segment = [_to_turns(seg) for seg in segments]

        seg_tasks = [
            self._seg_sum.summarize_segment(turns)
            for turns in turns_per_segment
        ]
        seg_summaries: list[str] = await asyncio.gather(*seg_tasks)  # type: ignore[assignment]

        from ..models.message import Message

        seg_messages = [
            Message(role="user", content=s)
            for s in seg_summaries
            if s and s != "-"
        ]

        if not seg_messages:
            return ""

        return await self._backend.summarize(
            seg_messages,
            target_ratio=self._target_ratio,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _to_turns(messages: list["Message"]) -> list[list["Message"]]:
    """Split a flat message list into [user, assistant] pairs."""
    turns = []
    buf: list = []
    for m in messages:
        buf.append(m)
        if m.role == "assistant":
            turns.append(buf)
            buf = []
    if buf:
        turns.append(buf)
    return turns
