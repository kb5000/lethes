"""Core Message dataclass — the fundamental unit of the lethes pipeline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# OpenAI-compatible content: plain string, multimodal list, or null (for tool-call-only turns)
ContentBlock = str | list[dict[str, Any]] | None


@dataclass
class Message:
    """
    A single conversation message enriched with orchestration metadata.

    The ``role``, ``content``, ``tool_calls``, ``tool_call_id``, and ``name``
    fields are standard OpenAI API fields.  All other fields are lethes-internal
    and are stripped when serialising back to the OpenAI format via
    :meth:`to_dict`.

    Tool call flow (OpenAI format)::

        # 1. Assistant decides to call a tool
        assistant_msg = Message(
            role="assistant",
            content=None,          # null when only tool_calls is present
            tool_calls=[{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }],
        )

        # 2. Caller executes the tool and returns the result
        tool_result = Message(
            role="tool",
            content="22°C, cloudy",
            tool_call_id="call_abc",
        )
    """

    role: str
    """``"system"`` | ``"user"`` | ``"assistant"`` | ``"tool"``"""

    content: ContentBlock = None
    """
    Plain string, OpenAI multimodal content list, or ``None``.

    ``None`` is valid for ``assistant`` messages that contain only
    ``tool_calls`` (no text reply).
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    """Stable identifier — used for dependency references and cache keys."""

    # ── Tool call fields (OpenAI API) ──────────────────────────────────────
    tool_calls: list[dict[str, Any]] | None = None
    """
    Present on ``assistant`` messages that invoke one or more tools.
    Each entry follows the OpenAI schema::

        {"id": "call_abc", "type": "function",
         "function": {"name": "fn", "arguments": "<json-string>"}}
    """

    tool_call_id: str | None = None
    """
    Present on ``tool`` result messages.  Must match the ``id`` of the
    corresponding entry in the preceding assistant ``tool_calls`` list.
    """

    name: str | None = None
    """
    Optional function name on ``tool`` result messages (recommended by
    the OpenAI spec for clarity).
    """

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
        """
        Return plain text from :attr:`content`, regardless of multimodal format.

        Returns an empty string for ``None`` content (e.g. tool-call-only
        assistant messages) or when no text blocks are present.
        """
        if self.content is None:
            return ""
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
        """
        Serialise to the OpenAI message dict format (strips lethes-only fields).

        Fields included only when non-``None``:
        ``tool_calls``, ``tool_call_id``, ``name``.
        ``content`` is always included (may be ``null`` for tool-call-only turns).
        """
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], **kwargs: Any) -> "Message":
        """
        Construct a :class:`Message` from an OpenAI-format dict.

        Handles all OpenAI message variants:

        * Plain text: ``{"role": "user", "content": "hello"}``
        * Multimodal: ``{"role": "user", "content": [{"type": "text", ...}, {"type": "image_url", ...}]}``
        * Tool call: ``{"role": "assistant", "content": null, "tool_calls": [...]}``
        * Tool result: ``{"role": "tool", "tool_call_id": "call_abc", "content": "result"}``

        Extra keyword arguments are forwarded to the dataclass constructor
        (e.g. ``base_weight``, ``pinned``, ``tags``).
        """
        return cls(
            role=d["role"],
            content=d.get("content"),  # None is valid for tool-call-only assistant msgs
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            **kwargs,
        )
