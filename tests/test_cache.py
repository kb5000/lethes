"""Tests for cache backends and PrefixSequenceTracker."""

from __future__ import annotations

import asyncio

import pytest

from lethes.cache.memory_backend import InMemoryCache
from lethes.cache.prefix_tracker import PrefixSequenceTracker, _common_prefix_length


# ── InMemoryCache ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_in_memory_set_and_get():
    cache = InMemoryCache()
    await cache.set("k", "hello")
    assert await cache.get("k") == "hello"


@pytest.mark.asyncio
async def test_in_memory_get_missing():
    cache = InMemoryCache()
    assert await cache.get("nope") is None


@pytest.mark.asyncio
async def test_in_memory_delete():
    cache = InMemoryCache()
    await cache.set("k", "v")
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_in_memory_delete_nonexistent_is_noop():
    cache = InMemoryCache()
    await cache.delete("ghost")  # must not raise


@pytest.mark.asyncio
async def test_in_memory_exists_true():
    cache = InMemoryCache()
    await cache.set("k", "v")
    assert await cache.exists("k") is True


@pytest.mark.asyncio
async def test_in_memory_exists_false():
    cache = InMemoryCache()
    assert await cache.exists("missing") is False


@pytest.mark.asyncio
async def test_in_memory_overwrite():
    cache = InMemoryCache()
    await cache.set("k", "first")
    await cache.set("k", "second")
    assert await cache.get("k") == "second"


@pytest.mark.asyncio
async def test_in_memory_ttl_expires(monkeypatch):
    """Entry should be gone after TTL elapses (mocked time)."""
    import time

    cache = InMemoryCache()
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base)

    await cache.set("k", "v", ttl=10)
    assert await cache.get("k") == "v"

    # Advance time beyond TTL
    monkeypatch.setattr(time, "monotonic", lambda: base + 11)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_in_memory_no_ttl_does_not_expire(monkeypatch):
    """Entry without TTL should persist indefinitely."""
    import time

    cache = InMemoryCache()
    base = time.monotonic()
    monkeypatch.setattr(time, "monotonic", lambda: base)

    await cache.set("k", "persistent")
    monkeypatch.setattr(time, "monotonic", lambda: base + 999999)
    assert await cache.get("k") == "persistent"


@pytest.mark.asyncio
async def test_in_memory_len():
    cache = InMemoryCache()
    await cache.set("a", "1")
    await cache.set("b", "2")
    assert len(cache) == 2


@pytest.mark.asyncio
async def test_in_memory_clear():
    cache = InMemoryCache()
    await cache.set("a", "1")
    await cache.set("b", "2")
    cache.clear()
    assert len(cache) == 0
    assert await cache.get("a") is None


# ── _common_prefix_length ─────────────────────────────────────────────────────


def test_common_prefix_length_identical():
    assert _common_prefix_length(["a", "b", "c"], ["a", "b", "c"]) == 3


def test_common_prefix_length_partial():
    assert _common_prefix_length(["a", "b", "c"], ["a", "b", "x"]) == 2


def test_common_prefix_length_empty():
    assert _common_prefix_length([], ["a"]) == 0
    assert _common_prefix_length(["a"], []) == 0


def test_common_prefix_length_no_match():
    assert _common_prefix_length(["x"], ["y"]) == 0


def test_common_prefix_length_one_shorter():
    assert _common_prefix_length(["a", "b"], ["a", "b", "c"]) == 2


# ── PrefixSequenceTracker ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_prefix_tracker_empty_before_prepare():
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    # Before prepare(), should return empty
    result = tracker.get_longest_prefix("session1", ["a", "b", "c"])
    assert result == []


@pytest.mark.asyncio
async def test_prefix_tracker_no_history():
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    await tracker.prepare("s1")
    result = tracker.get_longest_prefix("s1", ["a", "b"])
    assert result == []


@pytest.mark.asyncio
async def test_prefix_tracker_record_and_retrieve():
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    await tracker.prepare("s1")
    await tracker.record("s1", ["a", "b", "c"])

    # New tracker, load from cache
    tracker2 = PrefixSequenceTracker(cache)
    await tracker2.prepare("s1")
    result = tracker2.get_longest_prefix("s1", ["a", "b", "c", "d"])
    assert result == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_prefix_tracker_longest_wins():
    """When multiple stored sequences match, the longest prefix wins."""
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    await tracker.prepare("s1")
    await tracker.record("s1", ["a", "b"])
    await tracker.record("s1", ["a", "b", "c", "d"])

    result = tracker.get_longest_prefix("s1", ["a", "b", "c", "d", "e"])
    assert result == ["a", "b", "c", "d"]


@pytest.mark.asyncio
async def test_prefix_tracker_partial_match():
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    await tracker.prepare("s1")
    await tracker.record("s1", ["a", "b", "c"])
    # Current conversation diverges at position 2
    result = tracker.get_longest_prefix("s1", ["a", "b", "x", "y"])
    assert result == ["a", "b"]


@pytest.mark.asyncio
async def test_prefix_tracker_no_overlap():
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    await tracker.prepare("s1")
    await tracker.record("s1", ["a", "b", "c"])
    result = tracker.get_longest_prefix("s1", ["x", "y", "z"])
    assert result == []


@pytest.mark.asyncio
async def test_prefix_tracker_max_sequences_respected():
    """Older sequences are evicted when max_sequences is reached."""
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache, max_sequences=2)
    await tracker.prepare("s1")
    await tracker.record("s1", ["old1", "old2"])
    await tracker.record("s1", ["mid1", "mid2"])
    await tracker.record("s1", ["new1", "new2"])  # evicts ["old1", "old2"]

    # Load fresh tracker
    tracker2 = PrefixSequenceTracker(cache, max_sequences=2)
    await tracker2.prepare("s1")
    # "old1" prefix should no longer match (evicted)
    result = tracker2.get_longest_prefix("s1", ["old1", "old2"])
    assert result == []


@pytest.mark.asyncio
async def test_prefix_tracker_independent_sessions():
    """Different session IDs do not interfere with each other."""
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)

    await tracker.prepare("s1")
    await tracker.prepare("s2")
    await tracker.record("s1", ["a", "b"])
    await tracker.record("s2", ["x", "y"])

    assert tracker.get_longest_prefix("s1", ["a", "b", "c"]) == ["a", "b"]
    assert tracker.get_longest_prefix("s2", ["a", "b", "c"]) == []


@pytest.mark.asyncio
async def test_prefix_tracker_corrupt_cache_handled():
    """Corrupt JSON in cache must not crash the tracker."""
    cache = InMemoryCache()
    await cache.set("lethes:prefix_tracker:bad_session", "not-json{{{")
    tracker = PrefixSequenceTracker(cache)
    await tracker.prepare("bad_session")  # must not raise
    result = tracker.get_longest_prefix("bad_session", ["a"])
    assert result == []
