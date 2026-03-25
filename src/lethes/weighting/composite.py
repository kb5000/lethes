"""Composite weighting — linear combination of multiple strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.conversation import Conversation
    from ..models.message import Message
    from .base import WeightingStrategy


class CompositeWeightStrategy:
    """
    Weighted linear combination of multiple :class:`WeightingStrategy` instances.

    Example::

        from lethes.weighting import CompositeWeightStrategy
        from lethes.weighting.keyword import KeywordRelevanceStrategy
        from lethes.weighting.embedding import EmbeddingSimilarityStrategy

        strategy = CompositeWeightStrategy([
            (KeywordRelevanceStrategy(), 0.3),
            (EmbeddingSimilarityStrategy(...), 0.7),
        ])

    Each strategy's scores are normalised independently and then combined
    as a weighted sum.  The final scores are normalised to ``[0, 1]``.
    """

    def __init__(
        self,
        strategies: list[tuple["WeightingStrategy", float]],
    ) -> None:
        if not strategies:
            raise ValueError("At least one strategy is required")
        self._strategies = strategies

    async def score(
        self,
        messages: list["Message"],
        query: str,
        conversation: "Conversation",
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        if not messages:
            return {}

        # Gather scores from all strategies concurrently
        import asyncio

        tasks = [
            s.score(messages, query, conversation, context)
            for s, _ in self._strategies
        ]
        all_scores: list[dict[str, float]] = await asyncio.gather(*tasks)

        # Weighted combination
        combined: dict[str, float] = {m.id: 0.0 for m in messages}
        total_weight = sum(w for _, w in self._strategies)

        for (_, w), scores in zip(self._strategies, all_scores):
            norm = w / total_weight
            for msg_id, s in scores.items():
                combined[msg_id] = combined.get(msg_id, 0.0) + s * norm

        # Messages missing from any strategy keep their partial sum
        return combined

    def name(self) -> str:
        parts = [f"{s.name()}×{w:.2f}" for s, w in self._strategies]
        return f"composite({', '.join(parts)})"
