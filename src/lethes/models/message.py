"""Core Message dataclass — the fundamental unit of the lethes pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# OpenAI-compatible content: either a plain string or a multimodal list
ContentBlock = str | list[dict[str, Any]]


@dataclass
class Message:
    """
    A single conversation message enriched with orchestration metadata.

    The ``role`` and ``content`` fields are standard OpenAI API fields.
    All other fields are lethes-internal and are stripped when serialising
    back to the OpenAI format via :meth:`to_dict`.
    """

    role: str
    """``"system"`` | ``"user"`` | ``"assistant"`` | ``"tool"``"""

    content: ContentBlock
    """Plain string or OpenAI multimodal content list."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Stable identifier — used for dependency references and cache keys."""

    # ── Orchestration metadata ─────────────────────────────────────────────
    base_weight: float = 1.0
    """
    User-supplied static priority (via ``!weight=N`` flag or the Python API).
    Higher = more important.  Combined with the dynamic relevance score by the
    weighting layer to produce the final :attr:`weight`.
    """

    weight: float = 1.0
    """
    Effective weight used by selection algorithms.
    Recomputed on every orchestration pass:
    ``weight = base_weight * relevance_score``.
    Do **not** set this directly — use :attr:`base_weight` instead.
    """

    tags: set[str] = field(default_factory=set)
    """Free-form labels for categorisation (e.g. ``{"important", "tool_result"}``). """

    dependencies: list[str] = field(default_factory=list)
    """
    IDs of messages that *must* be kept whenever this message is kept.
    Enforced by :class:`~lethes.algorithms.dependency.DependencyAwareAlgorithm`.
    """

    pinned: bool = False
    """If ``True`` the message is never dropped or summarised."""

    summary: str | None = None
    """
    Pre-computed compressed representation of :attr:`content`.
    When the engine decides to *summarise* this message it replaces the
    outgoing content with this string (or requests a new summary if ``None``).
    """

    # ── Tracking ──────────────────────────────────────────────────────────
    token_count: int | None = None
    """Cached token count — populated by the token-counting step."""

    created_at: float = field(default_factory=time.time)
    sequence_index: int = 0
    """Original position in the conversation (0-based).  Used for final assembly."""

    # ── Helpers ───────────────────────────────────────────────────────────

    def get_text_content(self) -> str:
        """Return plain text from :attr:`content`, regardless of multimodal format."""
        if isinstance(self.content, str):
            return self.content
        parts: list[str] = []
        for item in self.content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts)

    def with_summary_content(self) -> "Message":
        """Return a shallow copy whose ``content`` is replaced by :attr:`summary`."""
        import dataclasses

        if self.summary is None:
            raise ValueError(f"Message {self.id!r} has no summary to substitute")
        return dataclasses.replace(self, content=self.summary)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the OpenAI message dict format (strips lethes-only fields)."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def from_dict(cls, d: dict[str, Any], **kwargs: Any) -> "Message":
        """
        Construct a :class:`Message` from an OpenAI-format dict.
        Extra keyword arguments are forwarded to the dataclass constructor
        (e.g. ``base_weight``, ``pinned``, ``tags``).
        """
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            **kwargs,
        )
