"""
SelectionAlgorithm protocol and SelectionResult dataclass.

Every algorithm must be **synchronous** — I/O (summarisation, caching) is
handled by the orchestrator around the algorithm call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..engine.constraints import ConstraintSet
    from ..models.budget import Budget
    from ..models.conversation import Conversation
    from ..utils.tokens import TokenCounter


@dataclass
class SelectionResult:
    """
    The algorithm's decision for each message in the conversation.

    All three lists are disjoint and together cover every non-system message
    (system messages are always kept by the orchestrator regardless).
    """

    keep_full: list[str] = field(default_factory=list)
    """Message IDs to include at their full content."""

    summarize: list[str] = field(default_factory=list)
    """Message IDs whose content should be replaced by their summary."""

    drop: list[str] = field(default_factory=list)
    """Message IDs to exclude from the outgoing context entirely."""

    estimated_tokens: int = 0
    """Estimated total tokens after applying this selection."""

    estimated_cost_usd: float = 0.0
    """Estimated API cost (USD) for the resulting context."""


@runtime_checkable
class SelectionAlgorithm(Protocol):
    """
    Protocol for context selection algorithms.

    Implement this to create a custom algorithm.  No inheritance required —
    structural (duck-type) compatibility is sufficient.

    All implementations must:

    * Honour ``message.pinned == True`` (never place pinned messages in ``drop``
      or ``summarize``).
    * Be deterministic for the same input (required for testing).
    * Be **synchronous** (the orchestrator wraps the call in a thread if needed).
    """

    def select(
        self,
        conversation: "Conversation",
        budget: "Budget",
        constraints: "ConstraintSet",
        token_counter: "TokenCounter",
    ) -> SelectionResult:
        """
        Decide which messages to keep, summarise, or drop.

        Parameters
        ----------
        conversation:
            The full conversation with ``message.weight`` already set by the
            weighting layer.
        budget:
            Token/cost limits to stay within.
        constraints:
            Hard rules that the result *must* satisfy (the orchestrator's
            :class:`~lethes.engine.constraints.ConstraintChecker` will repair
            any violations, but algorithms should try to avoid them).
        token_counter:
            Shared token counter — use this rather than creating a new one.
        """
        ...

    def name(self) -> str:
        """Short identifier for logging and the ``ContextPlan``."""
        ...
