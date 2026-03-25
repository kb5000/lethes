"""In-process async cache — no external dependencies."""

from __future__ import annotations

import time
from typing import NamedTuple


class _Entry(NamedTuple):
    value: str
    expires_at: float | None  # Unix timestamp or None (no expiry)


class InMemoryCache:
    """
    Thread-safe* in-process cache.

    ``*`` Access is not protected by a lock because asyncio is single-threaded
    by default.  Do **not** share an instance across OS threads.
    """

    def __init__(self) -> None:
        self._store: dict[str, _Entry] = {}

    async def get(self, key: str) -> str | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.expires_at is not None and time.monotonic() > entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        expires_at = (time.monotonic() + ttl) if ttl is not None else None
        self._store[key] = _Entry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    def __len__(self) -> int:  # useful for tests
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()
