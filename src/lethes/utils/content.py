"""Utilities for extracting text from OpenAI-format message content."""

from __future__ import annotations

from typing import Any


def get_text_content(content: str | list[dict[str, Any]] | None) -> str:
    """
    Return plain text from an OpenAI message ``content`` field.

    Handles all OpenAI content variants:

    * ``None`` — tool-call-only assistant messages; returns ``""``
    * Plain string — returned as-is
    * Multimodal list — text blocks are joined; non-text blocks (``image_url``,
      ``image``, ``audio``, etc.) are silently skipped::

          [{"type": "text", "text": "hello"}, {"type": "image_url", ...}]
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts)
