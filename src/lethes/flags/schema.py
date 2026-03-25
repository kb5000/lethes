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

    # ── Truncation / budget control ───────────────────────────────────────
    FULL = "full"
    """Bypass all truncation for this turn. Equivalent to an unlimited budget."""

    CONTEXT = "context"
    """``context=N`` — keep only the last N conversation turns."""

    NOSUM = "nosum"
    """Disable summarisation — dropped messages are hard-deleted."""

    TARGET = "target"
    """
    ``target=N`` — aim for approximately N tokens of context.

    Unlike :attr:`FULL` (unlimited) or the configured budget (hard cap),
    this sets a *target*: the engine fills as close to N tokens as possible.
    Useful for keeping context size predictable across turns.

    Example: ``!target=8000``
    """

    # ── Anchoring / forced-keep ───────────────────────────────────────────
    RECENT = "recent"
    """
    ``recent=N`` — unconditionally pin the last N non-system messages.

    These messages are never dropped or summarised regardless of weight.
    Other messages are still subject to normal selection.

    Example: ``!recent=6`` (always keep the last 3 Q&A turns)
    """

    KEEP_TAG = "keep_tag"
    """
    ``keep_tag=label`` — pin all messages that carry the given tag.

    Works with the ``!tag=label`` flag used in earlier messages.  A message
    tagged ``!+tag=important`` in a previous turn can be anchored here with
    ``!keep_tag=important``.

    Example: ``!keep_tag=important``
    """

    # ── Weighting overrides ───────────────────────────────────────────────
    TOOL_PENALTY = "tool_penalty"
    """
    ``tool_penalty=F`` — per-turn multiplier for tool-call intermediate messages.

    Overrides the default configured in :class:`~lethes.weighting.smart.SmartWeightingStrategy`.
    Lower values make tool intermediates less likely to be kept when budget is tight.

    Example: ``!tool_penalty=0.2``
    """

    PAIR_COHERENCE = "pair_coherence"
    """
    ``pair_coherence=F`` — fraction of a user message's score transferred to
    its immediately following assistant reply.

    Overrides the default configured in :class:`~lethes.weighting.smart.SmartWeightingStrategy`.
    Set to 0 to score assistant messages purely on their own keyword relevance.

    Example: ``!pair_coherence=0.9``
    """

    # ── Message metadata ──────────────────────────────────────────────────
    PIN = "pin"
    """Pin the current user message so it is never dropped or summarised."""

    WEIGHT = "weight"
    """``weight=N`` — set the ``base_weight`` of the current user message."""

    TAG = "tag"
    """``tag=label`` — add a tag to the current user message."""
