"""
RecencyBiasedAlgorithm — apply a recency multiplier then delegate to greedy.

Ensures that more recent messages receive a weight bonus proportional to
their position from the end of the conversation.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

from .base import SelectionResult
from .greedy import GreedyByWeightAlgorithm

if TYPE_CHECKING:
    from ..engine.constraints import ConstraintSet
    from ..models.budget import Budget
    from ..models.conversation import Conversation
    from ..utils.tokens import TokenCounter


class RecencyBiasedAlgorithm:
    """
    Wraps :class:`GreedyByWeightAlgorithm` after adjusting weights by recency.

    The adjusted weight for message at position ``i`` (0 = oldest) in a
    conversation of length ``n`` is::

        weight *= 1 + recency_factor * (i / (n - 1))

    This means the *most recent* message gets a multiplier of
    ``1 + recency_factor`` and the oldest gets ``1.0``.

    Parameters
    ----------
    recency_factor:
        How strongly to bias towards recent messages.
        ``0.0`` = no bias (same as plain greedy).
        ``2.0`` (default) = most recent message has 3× the weight bonus of oldest.
    """

    def __init__(
        self,
        recency_factor: float = 2.0,
        prefer_summarize: bool = True,
    ) -> None:
        self._factor = recency_factor
        self._inner = GreedyByWeightAlgorithm(prefer_summarize=prefer_summarize)

    def select(
        self,
        conversation: "Conversation",
        budget: "Budget",
        constraints: "ConstraintSet",
        token_counter: "TokenCounter",
    ) -> SelectionResult:
        chat = conversation.chat_messages()
        n = len(chat)
        if n == 0:
            return self._inner.select(conversation, budget, constraints, token_counter)

        # Apply recency multiplier to .weight (not base_weight)
        adjusted: list = []
        for i, msg in enumerate(chat):
            multiplier = 1.0 + self._factor * (i / max(n - 1, 1))
            adjusted.append(dataclasses.replace(msg, weight=msg.weight * multiplier))

        # Build a modified conversation with adjusted weights
        all_msgs = list(conversation.system_messages()) + adjusted
        adjusted_conv = conversation.with_messages(all_msgs)

        return self._inner.select(adjusted_conv, budget, constraints, token_counter)

    def name(self) -> str:
        return f"recency_biased(factor={self._factor})"
