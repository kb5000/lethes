"""Token counting utilities backed by tiktoken."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from ..models.message import Message


@lru_cache(maxsize=8)
def get_encoding(encoding_name: str = "o200k_base") -> tiktoken.Encoding:
    """Return a cached tiktoken encoding by name."""
    return tiktoken.get_encoding(encoding_name)


class TokenCounter:
    """
    Stateless token counter that uses a tiktoken encoding.

    A single instance is typically shared across the entire orchestration
    pipeline to avoid re-loading the encoding.
    """

    def __init__(self, encoding_name: str = "o200k_base") -> None:
        self._encoding = get_encoding(encoding_name)

    def count_text(self, text: str) -> int:
        """Count tokens in a plain string."""
        return len(self._encoding.encode(text))

    def count(self, message: "Message") -> int:
        """
        Count tokens for a :class:`~lethes.models.message.Message`.

        Includes text from ``content`` (all text blocks in multimodal messages)
        plus the JSON representation of ``tool_calls`` when present.
        """
        import json

        from ..utils.content import get_text_content

        if message.token_count is not None:
            return message.token_count
        parts = [get_text_content(message.content)]
        if message.tool_calls:
            parts.append(json.dumps(message.tool_calls, ensure_ascii=False))
        return self.count_text(" ".join(p for p in parts if p))

    def count_dict(self, message: dict) -> int:
        """
        Count tokens for a raw OpenAI message dict.

        Includes ``content`` text and ``tool_calls`` JSON when present.
        """
        import json

        from ..utils.content import get_text_content

        parts = [get_text_content(message.get("content"))]
        if message.get("tool_calls"):
            parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
        return self.count_text(" ".join(p for p in parts if p))

    def fill_counts(self, messages: list["Message"]) -> list["Message"]:
        """
        Return the same list with :attr:`~lethes.models.message.Message.token_count`
        populated for any message where it was ``None``.
        Mutates in-place for performance (counts are idempotent).
        """
        import dataclasses

        result = []
        for m in messages:
            if m.token_count is None:
                m = dataclasses.replace(m, token_count=self.count(m))
            result.append(m)
        return result
