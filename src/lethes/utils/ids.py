"""ID generation and cache-key derivation utilities."""

from __future__ import annotations

import hashlib
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.message import Message


def generate_message_id() -> str:
    """Return a new random UUID string."""
    return str(uuid.uuid4())


def cache_key_for_messages(messages: list["Message"]) -> str:
    """
    Derive a deterministic SHA-256 cache key from a sequence of messages.

    The key is based on the ``role`` and **text** content of each message so
    that structurally identical conversations map to the same key, regardless
    of which lethes-internal metadata fields differ.
    """
    from ..utils.content import get_text_content

    parts: list[str] = []
    for m in messages:
        text = get_text_content(m.content)
        parts.append(f"{m.role}:{text}")
    digest = hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()
    return digest


def cache_key_for_strings(*parts: str) -> str:
    """Derive a SHA-256 cache key from arbitrary strings."""
    combined = "\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
