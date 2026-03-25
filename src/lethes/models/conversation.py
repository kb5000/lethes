"""Conversation — an ordered, immutable-style collection of Messages."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from .message import Message

if TYPE_CHECKING:
    from ..utils.tokens import TokenCounter


class Conversation:
    """
    An ordered collection of :class:`~lethes.models.message.Message` objects
    that represents the full conversation history.

    All mutation helpers return **new** ``Conversation`` instances so that
    objects can be safely shared across async tasks without defensive copies.
    """

    def __init__(
        self,
        messages: list[Message] | tuple[Message, ...] = (),
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._messages: tuple[Message, ...] = tuple(messages)
        self._session_id = session_id or str(uuid.uuid4())
        self._metadata: dict[str, Any] = metadata or {}
        # Build lookup index
        self._index: dict[str, Message] = {m.id: m for m in self._messages}

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_openai_messages(
        cls,
        messages: list[dict[str, Any]],
        session_id: str | None = None,
        **msg_kwargs: Any,
    ) -> "Conversation":
        """
        Convert a list of OpenAI-format dicts into a :class:`Conversation`.

        ``msg_kwargs`` are forwarded to every :class:`Message` constructor call.

        **Tool call dependency linking** — bidirectional dependencies are
        automatically set between assistant ``tool_calls`` messages and their
        corresponding ``tool`` result messages.  This ensures the orchestration
        engine always keeps or drops the pair together, preventing invalid
        sequences where a tool result appears without its preceding tool call
        (or vice versa), which the OpenAI API rejects.
        """
        objs = [
            Message.from_dict(d, sequence_index=i, **msg_kwargs)
            for i, d in enumerate(messages)
        ]
        objs = _link_tool_dependencies(objs)
        return cls(objs, session_id=session_id)

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Serialise back to the standard OpenAI ``messages`` list."""
        return [m.to_dict() for m in self._messages]

    # ── Querying ──────────────────────────────────────────────────────────

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._messages

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self):
        return iter(self._messages)

    def get_by_id(self, message_id: str) -> Message | None:
        return self._index.get(message_id)

    def system_messages(self) -> list[Message]:
        return [m for m in self._messages if m.role == "system"]

    def chat_messages(self) -> list[Message]:
        """All non-system messages."""
        return [m for m in self._messages if m.role != "system"]

    def last_user_message(self) -> Message | None:
        for m in reversed(self._messages):
            if m.role == "user":
                return m
        return None

    # ── Immutable-style mutation ───────────────────────────────────────────

    def append(self, message: Message) -> "Conversation":
        """Return new Conversation with *message* appended."""
        msgs = list(self._messages)
        msg = _with_index(message, len(msgs))
        msgs.append(msg)
        return Conversation(msgs, session_id=self._session_id, metadata=self._metadata)

    def replace(self, message: Message) -> "Conversation":
        """Return new Conversation with the message matching *message.id* replaced."""
        msgs = [message if m.id == message.id else m for m in self._messages]
        return Conversation(msgs, session_id=self._session_id, metadata=self._metadata)

    def without(self, ids: set[str]) -> "Conversation":
        """Return new Conversation with all messages whose id is in *ids* removed."""
        msgs = [m for m in self._messages if m.id not in ids]
        return Conversation(msgs, session_id=self._session_id, metadata=self._metadata)

    def with_messages(self, messages: list[Message]) -> "Conversation":
        """Return new Conversation with *messages* as the full message list."""
        return Conversation(messages, session_id=self._session_id, metadata=self._metadata)

    def with_metadata(self, **kwargs: Any) -> "Conversation":
        meta = {**self._metadata, **kwargs}
        return Conversation(self._messages, session_id=self._session_id, metadata=meta)

    # ── Token accounting ──────────────────────────────────────────────────

    def total_tokens(self, counter: "TokenCounter") -> int:
        return sum(counter.count(m) for m in self._messages)

    # ── Repr ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Conversation(session_id={self._session_id!r}, "
            f"messages={len(self._messages)})"
        )


def _with_index(message: Message, index: int) -> Message:
    import dataclasses

    if message.sequence_index == index:
        return message
    return dataclasses.replace(message, sequence_index=index)


def _link_tool_dependencies(messages: list[Message]) -> list[Message]:
    """
    Auto-set bidirectional dependencies between tool-call and tool-result messages.

    For each ``tool`` result message, its ``tool_call_id`` is matched against
    ``tool_calls[*].id`` in preceding ``assistant`` messages.  The matched pair
    gets mutual ``dependencies`` entries so the orchestration engine always
    keeps or drops them together.

    This runs once at parse time (``from_openai_messages``).  Messages that
    already have explicit ``dependencies`` set are left untouched for those
    entries; the auto-linked IDs are appended.
    """
    import dataclasses

    # Build maps needed for cross-referencing
    # tool_call_id → id of the tool-result Message
    result_by_call_id: dict[str, str] = {}
    for msg in messages:
        if msg.role == "tool" and msg.tool_call_id:
            result_by_call_id[msg.tool_call_id] = msg.id

    # assistant message id → list of tool-result message ids
    assistant_results: dict[str, list[str]] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            linked = [
                result_by_call_id[tc["id"]]
                for tc in msg.tool_calls
                if isinstance(tc, dict) and tc.get("id") in result_by_call_id
            ]
            if linked:
                assistant_results[msg.id] = linked

    # tool_call_id → id of the assistant Message that issued it
    assistant_by_call_id: dict[str, str] = {}
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and tc.get("id"):
                    assistant_by_call_id[tc["id"]] = msg.id

    if not assistant_results and not assistant_by_call_id:
        return messages

    updated: dict[str, Message] = {}

    for msg in messages:
        if msg.role == "assistant" and msg.id in assistant_results:
            extra = assistant_results[msg.id]
            new_deps = list(dict.fromkeys(msg.dependencies + extra))
            updated[msg.id] = dataclasses.replace(msg, dependencies=new_deps)

        if msg.role == "tool" and msg.tool_call_id:
            asst_id = assistant_by_call_id.get(msg.tool_call_id)
            if asst_id:
                new_deps = list(dict.fromkeys(msg.dependencies + [asst_id]))
                updated[msg.id] = dataclasses.replace(msg, dependencies=new_deps)

    return [updated.get(m.id, m) for m in messages]
