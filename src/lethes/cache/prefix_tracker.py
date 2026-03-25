"""
Prefix sequence tracker — used by PrefixCacheOptimizedAlgorithm to
maximise KV-cache hit rates by anchoring the longest known common prefix.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import CacheBackend

logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "lethes:prefix_tracker:"


class PrefixSequenceTracker:
    """
    Tracks which message-ID sequences have been sent to the model in prior
    requests so that the selection algorithm can find the longest common
    prefix and anchor it (avoiding cache misses on unchanged history).

    The tracker stores sequences per *session_id*.  On each request it:

    1. Is prepared (async): loads prior sequences from the cache backend
       into memory so the synchronous :meth:`get_longest_prefix` can be called
       inside the algorithm's ``select()`` method.
    2. Records (async): saves the new sequence after assembly.

    Thread-safety: single-threaded asyncio use only.
    """

    def __init__(self, backend: "CacheBackend", max_sequences: int = 5) -> None:
        self._backend = backend
        self._max_sequences = max_sequences
        # session_id → list of recently sent message-ID sequences
        self._loaded: dict[str, list[list[str]]] = {}

    # ── Async I/O ─────────────────────────────────────────────────────────

    async def prepare(self, session_id: str) -> None:
        """Load prior sequences for *session_id* from the cache backend."""
        key = _CACHE_KEY_PREFIX + session_id
        raw = await self._backend.get(key)
        if raw:
            try:
                self._loaded[session_id] = json.loads(raw)
            except Exception:
                logger.warning("Failed to load prefix sequences for session %s", session_id)
                self._loaded[session_id] = []
        else:
            self._loaded[session_id] = []

    async def record(self, session_id: str, message_ids: list[str]) -> None:
        """Save *message_ids* as the latest sent sequence for *session_id*."""
        sequences = self._loaded.get(session_id, [])
        sequences.append(message_ids)
        # Keep only the most recent N sequences
        sequences = sequences[-self._max_sequences :]
        self._loaded[session_id] = sequences
        key = _CACHE_KEY_PREFIX + session_id
        await self._backend.set(key, json.dumps(sequences), ttl=86400)  # 24 h

    # ── Sync query (called inside algorithm.select()) ─────────────────────

    def get_longest_prefix(
        self, session_id: str, candidate_ids: list[str]
    ) -> list[str]:
        """
        Return the longest prefix of *candidate_ids* that matches the start of
        any previously recorded sequence for *session_id*.

        This is a synchronous method so it can be called inside the sync
        ``select()`` method of a selection algorithm.
        Call :meth:`prepare` in an async context before calling this.
        """
        sequences = self._loaded.get(session_id, [])
        best: list[str] = []
        for seq in sequences:
            common = _common_prefix_length(seq, candidate_ids)
            if common > len(best):
                best = candidate_ids[:common]
        return best


def _common_prefix_length(a: list[str], b: list[str]) -> int:
    length = 0
    for x, y in zip(a, b):
        if x == y:
            length += 1
        else:
            break
    return length
