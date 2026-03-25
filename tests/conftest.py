"""
Test configuration — patches tiktoken with an offline-safe stub.

tiktoken normally downloads encoding files from the internet.  In offline
environments we replace get_encoding with a simple word-count counter.
"""

from __future__ import annotations

import pytest


class _OfflineEncoding:
    """Word-count based stub encoder (1 token ≈ 1 word)."""

    def encode(self, text: str) -> list[int]:
        return list(range(len(text.split())))


@pytest.fixture(autouse=True)
def mock_tiktoken(monkeypatch):
    """Replace tiktoken with an offline stub for all tests."""
    offline = _OfflineEncoding()

    # Patch tiktoken.get_encoding globally
    monkeypatch.setattr("tiktoken.get_encoding", lambda name: offline)

    # Clear lru_cache so the monkeypatch takes effect
    try:
        from lethes.utils import tokens as tokens_mod
        tokens_mod.get_encoding.cache_clear()
    except Exception:
        pass
