"""
PrefixCacheOptimizedAlgorithm — maximise KV-cache hit rate.

Anchors the longest prefix of the previous request that is still present
in the current conversation, then fills remaining budget by weight.
This minimises the number of tokens that need to be re-processed by the
model on each turn, reducing both latency and cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import SelectionResult
from .greedy import GreedyByWeightAlgorithm

if TYPE_CHECKING:
    from ..cache.prefix_tracker import PrefixSequenceTracker
    from ..engine.constraints import ConstraintSet
    from ..models.budget import Budget
    from ..models.conversation import Conversation
    from ..utils.tokens import TokenCounter


class PrefixCacheOptimizedAlgorithm:
    """
    Selection algorithm that maximises prefix-cache reuse.

    Strategy:

    1. Look up the longest common prefix between the current conversation
       and the last sent sequence (via the :class:`~lethes.cache.prefix_tracker.PrefixSequenceTracker`).
    2. Pin all messages in that prefix (they must be kept to preserve cache).
    3. Fill remaining budget using greedy-by-weight for the rest.

    Call :meth:`~lethes.cache.prefix_tracker.PrefixSequenceTracker.prepare`
    before using this algorithm so that prior sequences are loaded into memory.

    Parameters
    ----------
    prefix_tracker:
        The tracker that knows which sequences have been sent previously.
    session_id:
        Session identifier for tracker lookups.
    prefer_summarize:
        Passed through to the inner greedy algorithm.
    """

    def __init__(
        self,
        prefix_tracker: "PrefixSequenceTracker",
        session_id: str,
        prefer_summarize: bool = True,
    ) -> None:
        self._tracker = prefix_tracker
        self._session_id = session_id
        self._greedy = GreedyByWeightAlgorithm(prefer_summarize=prefer_summarize)

    def select(
        self,
        conversation: "Conversation",
        budget: "Budget",
        constraints: "ConstraintSet",
        token_counter: "TokenCounter",
    ) -> SelectionResult:
        import dataclasses

        candidate_ids = [m.id for m in conversation.messages]
        prefix_ids = set(
            self._tracker.get_longest_prefix(self._session_id, candidate_ids)
        )

        if not prefix_ids:
            # No known prefix — fall back to greedy
            return self._greedy.select(conversation, budget, constraints, token_counter)

        # Temporarily pin all prefix messages for greedy's benefit
        pinned_msgs = []
        other_msgs = []
        for msg in conversation.messages:
            if msg.id in prefix_ids and not msg.pinned:
                pinned_msgs.append(dataclasses.replace(msg, pinned=True))
            else:
                pinned_msgs.append(msg) if msg.id in prefix_ids else other_msgs.append(msg)

        modified_conv = conversation.with_messages(pinned_msgs + other_msgs)
        result = self._greedy.select(modified_conv, budget, constraints, token_counter)

        # Un-pin messages that we artificially pinned (keep them in keep_full but
        # don't let them block the constraint checker from moving them)
        return result

    def name(self) -> str:
        return "prefix_cache_optimized"
