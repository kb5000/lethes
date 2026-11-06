"""Explicit estimators; local estimates must never be labelled billed usage."""

from __future__ import annotations

from typing import Protocol

from .history import Message, canonical


class TokenCounter(Protocol):
    def count(self, messages: list[Message]) -> int: ...


class UnsupportedContent(ValueError):
    pass


class ByteTokenCounter:
    """Offline conservative text estimate: UTF-8 bytes plus framing.

    Images/audio need a provider-specific counter, not URL text counting.
    """

    cache_identity = "utf8-byte-counter-v1"

    def count(self, messages: list[Message]) -> int:
        for message in messages:
            content = message.get("content")
            if message.get("images") or message.get("audio"):
                raise UnsupportedContent("Configure a provider-specific multimodal token counter")
            if isinstance(content, list) and any(
                not isinstance(p, dict) or p.get("type") != "text" for p in content
            ):
                raise UnsupportedContent("Configure a provider-specific multimodal token counter")
        return 3 + sum(8 + len(canonical(m).encode()) for m in messages)


class TiktokenCounter(ByteTokenCounter):
    """Optional text counter; framing is estimated, schemas reserved separately."""

    def __init__(self, encoding: str = "o200k_base") -> None:
        import tiktoken

        self.encoding = tiktoken.get_encoding(encoding)
        self.cache_identity = f"tiktoken:{encoding}"

    def count(self, messages: list[Message]) -> int:
        super().count(messages)
        return 3 + sum(
            8 + len(self.encoding.encode(canonical(m), disallowed_special=())) for m in messages
        )


def common_prefix_tokens(a: list[Message], b: list[Message], counter: TokenCounter) -> int:
    shared = []
    for left, right in zip(a, b, strict=False):
        if canonical(left) != canonical(right):
            break
        shared.append(left)
    return max(0, counter.count(shared) - 3) if shared else 0
