"""
GreedyByWeightAlgorithm — keep highest-weight messages first.

O(n log n) per call (dominated by the sort).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import SelectionResult

if TYPE_CHECKING:
    from ..engine.constraints import ConstraintSet
    from ..models.budget import Budget
    from ..models.conversation import Conversation
    from ..models.message import Message
    from ..utils.tokens import TokenCounter


class GreedyByWeightAlgorithm:
    """
    Greedy selection: sort all non-system messages by ``weight`` descending,
    then fill the token budget starting from the highest-weight messages.

    Pinned messages and dependency chains are handled first (they count
    against the budget but cannot be skipped).

    The algorithm always tries to **summarise** before **dropping** — if a
    message has a pre-computed ``.summary`` it is placed in ``summarize``
    rather than ``drop``.

    Parameters
    ----------
    prefer_summarize:
        When ``True`` (default), overflow messages that have a pre-computed
        summary are placed in ``summarize`` instead of ``drop``.
    """

    def __init__(self, prefer_summarize: bool = True) -> None:
        self._prefer_summarize = prefer_summarize

    def select(
        self,
        conversation: "Conversation",
        budget: "Budget",
        constraints: "ConstraintSet",
        token_counter: "TokenCounter",
    ) -> SelectionResult:
        chat_msgs = conversation.chat_messages()

        # Separate pinned from candidates
        pinned: list[Message] = [m for m in chat_msgs if m.pinned]
        candidates: list[Message] = [m for m in chat_msgs if not m.pinned]

        # Tokens used by pinned messages
        used_tokens = sum(token_counter.count(m) for m in pinned)
        headroom = budget.headroom_tokens(used_tokens)

        # Sort candidates by weight descending
        candidates_sorted = sorted(candidates, key=lambda m: m.weight, reverse=True)

        keep_full: list[str] = [m.id for m in pinned]
        summarize: list[str] = []
        drop: list[str] = []

        for msg in candidates_sorted:
            tokens = token_counter.count(msg)
            fits = headroom < 0 or used_tokens + tokens <= (
                used_tokens + headroom if headroom >= 0 else float("inf")
            )

            if headroom >= 0:
                fits = tokens <= headroom

            if fits:
                keep_full.append(msg.id)
                used_tokens += tokens
                if headroom >= 0:
                    headroom -= tokens
            else:
                # Try summary first — but never summarize tool call pairs,
                # since they must remain structurally intact for the API.
                is_tool_pair = msg.role == "tool" or bool(msg.tool_calls)
                if (
                    self._prefer_summarize
                    and msg.summary is not None
                    and not is_tool_pair
                ):
                    summary_tokens = token_counter.count_text(msg.summary)
                    if headroom < 0 or summary_tokens <= headroom:
                        summarize.append(msg.id)
                        used_tokens += summary_tokens
                        if headroom >= 0:
                            headroom -= summary_tokens
                        continue
                drop.append(msg.id)

        return SelectionResult(
            keep_full=keep_full,
            summarize=summarize,
            drop=drop,
            estimated_tokens=used_tokens,
        )

    def name(self) -> str:
        return "greedy_by_weight"
