"""
Weighting strategy protocol and supporting types.

A WeightingStrategy computes a **relevance score** for each message in the
conversation relative to the current user query and conversation topic.
The orchestrator multiplies this score by the message's ``base_weight`` to
derive the final ``weight`` used by the selection algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..models.conversation import Conversation
    from ..models.message import Message


@runtime_checkable
class WeightingStrategy(Protocol):
    """
    Compute per-message relevance scores for the current turn.

    Implementations must be **async** because strategies such as
    :class:`~lethes.weighting.embedding.EmbeddingSimilarityStrategy` make
    network calls.

    Returns
    -------
    dict[str, float]
        Mapping of ``message_id → score``.  Scores should be in ``[0.0, 1.0]``
        but this is not enforced — algorithms treat them as relative weights.
        Messages not present in the returned dict receive a default score of
        ``1.0`` (i.e. no relevance penalty).
    """

    async def score(
        self,
        messages: list["Message"],
        query: str,
        conversation: "Conversation",
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        """Return ``{message_id: relevance_score}`` for each message."""
        ...

    def name(self) -> str:
        """Human-readable strategy name for logging and observability."""
        ...


@runtime_checkable
class TopicDetector(Protocol):
    """
    Segment the conversation into topic clusters.

    Returns
    -------
    dict[str, str]
        ``{message_id: topic_label}``
    """

    def detect(self, conversation: "Conversation") -> dict[str, str]:
        ...


def apply_scores(
    messages: list["Message"],
    scores: dict[str, float],
) -> list["Message"]:
    """
    Return a new list with ``message.weight = message.base_weight * score``
    applied for every message.  Messages absent from *scores* keep their
    current ``weight`` unchanged.

    This is called by the orchestrator after :meth:`WeightingStrategy.score`.
    """
    import dataclasses

    result = []
    for m in messages:
        s = scores.get(m.id)
        if s is None:
            result.append(m)
        else:
            result.append(dataclasses.replace(m, weight=m.base_weight * s))
    return result
