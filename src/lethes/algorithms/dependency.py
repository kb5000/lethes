"""
DependencyAwareAlgorithm — decorator that enforces message dependency chains.

Wraps any SelectionAlgorithm and post-processes its result to ensure that
if a message is kept, all of its declared dependencies are also kept.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import SelectionAlgorithm, SelectionResult

if TYPE_CHECKING:
    from ..engine.constraints import ConstraintSet
    from ..models.budget import Budget
    from ..models.conversation import Conversation
    from ..utils.tokens import TokenCounter


class DependencyAwareAlgorithm:
    """
    Wraps another algorithm and enforces ``message.dependencies`` chains.

    Rules applied after the inner algorithm runs:

    1. If a message is in ``keep_full``, all its dependencies are also moved
       to ``keep_full`` (recursively).
    2. If a dependency is in ``drop`` but the dependent is kept, the dependency
       is promoted to ``summarize`` (if it has a summary) or ``keep_full``.
    3. If a dependency is in ``summarize`` and the dependent is in ``keep_full``,
       the dependency stays in ``summarize`` (acceptable — summary is kept).

    Parameters
    ----------
    inner:
        The underlying :class:`SelectionAlgorithm` to decorate.
    """

    def __init__(self, inner: SelectionAlgorithm) -> None:
        self._inner = inner

    def select(
        self,
        conversation: "Conversation",
        budget: "Budget",
        constraints: "ConstraintSet",
        token_counter: "TokenCounter",
    ) -> SelectionResult:
        result = self._inner.select(conversation, budget, constraints, token_counter)
        return self._resolve_dependencies(result, conversation)

    def _resolve_dependencies(
        self, result: SelectionResult, conversation: "Conversation"
    ) -> SelectionResult:
        keep_full = set(result.keep_full)
        summarize = set(result.summarize)
        drop = set(result.drop)

        # Build dependency map
        dep_map: dict[str, list[str]] = {}
        for msg in conversation.messages:
            dep_map[msg.id] = msg.dependencies

        # Iteratively resolve: if a kept message has a dropped dependency, promote it
        changed = True
        while changed:
            changed = False
            for msg_id in list(keep_full | summarize):
                for dep_id in dep_map.get(msg_id, []):
                    if dep_id in drop:
                        # Promote dependency
                        dep_msg = conversation.get_by_id(dep_id)
                        drop.discard(dep_id)
                        if dep_msg and dep_msg.summary is not None:
                            summarize.add(dep_id)
                        else:
                            keep_full.add(dep_id)
                        changed = True

        return SelectionResult(
            keep_full=list(keep_full),
            summarize=list(summarize),
            drop=list(drop),
            estimated_tokens=result.estimated_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
        )

    def name(self) -> str:
        return f"dependency_aware({self._inner.name()})"
