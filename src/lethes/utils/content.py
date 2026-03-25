"""Utilities for extracting text from OpenAI-format message content."""

from __future__ import annotations

from typing import Any


def get_text_content(content: str | list[dict[str, Any]]) -> str:
    """
    Return plain text from an OpenAI message ``content`` field.

    Handles both the simple string form and the multimodal list form::

        [{"type": "text", "text": "hello"}, {"type": "image_url", ...}]

    Non-text blocks are silently skipped.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "\n".join(parts)
