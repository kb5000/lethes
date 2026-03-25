"""CacheBackend protocol — pluggable async key-value store."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """
    Minimal async key-value store interface.

    Both :class:`~lethes.cache.memory_backend.InMemoryCache` and
    :class:`~lethes.cache.redis_backend.RedisCache` implement this protocol.
    Any object with these four methods is accepted by the engine.
    """

    async def get(self, key: str) -> str | None:
        """Return the cached string value or ``None`` if not present."""
        ...

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        """
        Store *value* under *key*.
        Optional *ttl* specifies expiry in seconds; ``None`` means no expiry.
        """
        ...

    async def delete(self, key: str) -> None:
        """Remove *key*.  No-op if the key does not exist."""
        ...

    async def exists(self, key: str) -> bool:
        """Return ``True`` if *key* is present."""
        ...
