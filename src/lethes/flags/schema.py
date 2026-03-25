"""Well-known flag names used by the orchestration engine."""

from __future__ import annotations

from enum import StrEnum


class WellKnownFlag(StrEnum):
    """
    Built-in flags recognised by the orchestrator.

    Users can define their own flags; unrecognised flags are stored in
    :attr:`~lethes.flags.session.SessionFlags.effective_flags` but ignored
    by the engine unless a custom algorithm or integration reads them.
    """

    # ── Truncation control ────────────────────────────────────────────────
    FULL = "full"
    """Bypass all truncation for this turn. Equivalent to an unlimited budget."""

    CONTEXT = "context"
    """``context=N`` — override the turn limit for this turn."""

    NOSUM = "nosum"
    """Disable summarisation — dropped messages are hard-deleted."""

    # ── Message metadata ──────────────────────────────────────────────────
    PIN = "pin"
    """Pin the current user message so it is never dropped or summarised."""

    WEIGHT = "weight"
    """``weight=N`` — set the ``base_weight`` of the current user message."""

    TAG = "tag"
    """``tag=label`` — add a tag to the current user message."""
