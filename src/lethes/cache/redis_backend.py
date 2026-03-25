"""Redis-backed async cache (optional dependency: ``lethes[redis]``)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import redis.asyncio as aioredis


class RedisCache:
    """
    Async cache backed by Redis.

    Requires the ``redis`` extra::

        pip install lethes[redis]

    Parameters
    ----------
    client:
        A ``redis.asyncio.Redis`` instance.  Create it externally so that
        connection-pool settings remain under the caller's control.

    key_prefix:
        Optional string prepended to every key (useful for namespacing).
    """

    def __init__(
        self,
        client: "aioredis.Redis",
        key_prefix: str = "lethes:",
    ) -> None:
        try:
            import redis.asyncio  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "RedisCache requires the 'redis' extra.  "
                "Install with: pip install lethes[redis]"
            ) from exc
        self._client = client
        self._prefix = key_prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> str | None:
        raw = await self._client.get(self._key(key))
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw

    async def set(self, key: str, value: str, ttl: int | None = None) -> None:
        if ttl is not None:
            await self._client.setex(self._key(key), ttl, value)
        else:
            await self._client.set(self._key(key), value)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(self._key(key)))

    @classmethod
    def from_url(cls, url: str = "redis://localhost:6379/0", **kwargs) -> "RedisCache":
        """Convenience constructor from a Redis URL."""
        try:
            import redis.asyncio as aioredis
        except ImportError as exc:
            raise ImportError(
                "RedisCache requires the 'redis' extra: pip install lethes[redis]"
            ) from exc
        client = aioredis.from_url(url, decode_responses=False)
        return cls(client, **kwargs)
