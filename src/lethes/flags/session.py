"""Persistent flag state across conversation turns."""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING

from .parser import FlagMap, extract_flags
from .schema import WellKnownFlag

if TYPE_CHECKING:
    from ..models.conversation import Conversation
    from ..models.message import Message

logger = logging.getLogger(__name__)


class SessionFlags:
    """
    Maintains per-session flag state built by replaying the message history.

    **Persistence rules** (mirroring ``example.py``):

    - ``+key[=value]``  → add *key* to the persistent set
    - ``-key``          → remove *key* from the persistent set
    - ``key[=value]``   → temporary flag; effective only for the current turn

    The effective flags for the *last* turn are
    ``persistent_flags ∪ current_turn_temporary_flags``.

    Usage::

        session_flags, modified_conv = SessionFlags.from_conversation(conversation)
        flags = session_flags.effective_flags()
        if WellKnownFlag.FULL in flags:
            ...
    """

    def __init__(self) -> None:
        self._persistent: FlagMap = {}
        self._current_turn: FlagMap = {}

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_conversation(
        cls, conversation: "Conversation"
    ) -> tuple["SessionFlags", "Conversation"]:
        """
        Replay all messages to reconstruct session flag state.

        Flag prefixes are **stripped** from message content.  Because
        :class:`~lethes.models.conversation.Conversation` is immutable-style,
        a new ``Conversation`` (with cleaned content) is returned alongside
        the ``SessionFlags``.

        Returns
        -------
        tuple[SessionFlags, Conversation]
            ``(session_flags, modified_conversation)``
        """
        instance = cls()
        modified_messages: list[Message] = []

        for msg in conversation.messages:
            if msg.role in ("user", "system") and isinstance(msg.content, str):
                raw_flags, remaining = extract_flags(msg.content)
                # Strip flags from message content
                cleaned_msg = dataclasses.replace(msg, content=remaining)
                instance._apply_raw_flags(raw_flags, is_last=(msg is conversation.messages[-1]))
                modified_messages.append(cleaned_msg)
            else:
                modified_messages.append(msg)

        modified_conv = conversation.with_messages(modified_messages)
        return instance, modified_conv

    # ── Public API ────────────────────────────────────────────────────────

    def effective_flags(self) -> FlagMap:
        """
        Snapshot of the flags effective for the *current* (last processed) turn:
        persistent flags merged with the last turn's temporary flags.
        """
        return {**self._persistent, **self._current_turn}

    def get(self, key: str | WellKnownFlag, default: str | None = None) -> str | None:
        return self.effective_flags().get(str(key), default)

    def __contains__(self, key: str | WellKnownFlag) -> bool:
        return str(key) in self.effective_flags()

    # ── Internal ─────────────────────────────────────────────────────────

    def _apply_raw_flags(self, raw: FlagMap, *, is_last: bool) -> None:
        """Update persistent state and, if *is_last*, the current-turn flags."""
        temp: FlagMap = {}

        for key, value in raw.items():
            if key.startswith("+"):
                clean = key[1:]
                self._persistent[clean] = value
            elif key.startswith("-"):
                clean = key[1:]
                self._persistent.pop(clean, None)
            else:
                temp[key] = value

        if is_last:
            self._current_turn = temp

    def _apply_well_known_to_message(
        self, msg: "Message"
    ) -> "Message":
        """
        Apply well-known flags (pin, weight, tag) to a message.
        Called by the orchestrator on the last user message after flag parsing.
        """
        effective = self.effective_flags()
        changes: dict = {}

        if WellKnownFlag.PIN in effective:
            changes["pinned"] = True

        if WellKnownFlag.WEIGHT in effective:
            try:
                changes["base_weight"] = float(effective[WellKnownFlag.WEIGHT])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                logger.warning("Invalid weight flag value: %r", effective[WellKnownFlag.WEIGHT])

        if WellKnownFlag.TAG in effective:
            tag_val = effective[WellKnownFlag.TAG]
            if tag_val:
                new_tags = set(msg.tags) | {tag_val}
                changes["tags"] = new_tags

        if changes:
            return dataclasses.replace(msg, **changes)
        return msg
