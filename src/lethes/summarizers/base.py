"""Summarizer protocol — pluggable text compression backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..models.message import Message


@runtime_checkable
class Summarizer(Protocol):
    """
    Compress a sequence of messages into a single summary string.

    All implementations must be **async** because LLM-based backends
    make network calls.

    Parameters
    ----------
    messages:
        The messages to summarise (typically a user+assistant pair or a
        segment of the conversation).
    target_ratio:
        Desired compression ratio (``0.3`` = keep ~30 % of original length).
        Implementations treat this as a *guideline*, not a hard constraint.
    context_messages:
        Additional prior messages provided as context for coherence, but
        which are **not** themselves summarised.
    """

    async def summarize(
        self,
        messages: list["Message"],
        *,
        target_ratio: float = 0.3,
        context_messages: list["Message"] | None = None,
    ) -> str:
        """Return the summary string."""
        ...

    def name(self) -> str:
        """Short identifier for logging."""
        ...
