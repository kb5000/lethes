"""Constraint definitions and the ConstraintChecker / repair logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..algorithms.base import SelectionResult
    from ..models.conversation import Conversation


@dataclass(frozen=True)
class ConstraintSet:
    """
    Hard rules that a :class:`~lethes.algorithms.base.SelectionResult` must
    satisfy.  The :class:`ConstraintChecker` validates and repairs violations.
    """

    min_chat_messages: int = 1
    """Always keep at least this many non-system messages."""

    require_last_user: bool = True
    """The last user message must never be dropped or summarised."""

    require_all_system: bool = True
    """System messages are always kept (they are never passed to algorithms)."""


@dataclass
class ConstraintViolation:
    rule: str
    message_id: str
    description: str


class ConstraintChecker:
    """Validate a :class:`~lethes.algorithms.base.SelectionResult` and repair it."""

    def validate(
        self,
        result: "SelectionResult",
        conversation: "Conversation",
        constraints: ConstraintSet,
    ) -> list[ConstraintViolation]:
        violations: list[ConstraintViolation] = []
        drop_set = set(result.drop)
        sum_set = set(result.summarize)

        if constraints.require_last_user:
            last_user = conversation.last_user_message()
            if last_user and last_user.id in drop_set:
                violations.append(
                    ConstraintViolation(
                        "require_last_user",
                        last_user.id,
                        "Last user message is in drop set",
                    )
                )
            if last_user and last_user.id in sum_set:
                violations.append(
                    ConstraintViolation(
                        "require_last_user",
                        last_user.id,
                        "Last user message is in summarize set",
                    )
                )

        if constraints.min_chat_messages > 0:
            kept_count = len(result.keep_full) + len(result.summarize)
            if kept_count < constraints.min_chat_messages:
                violations.append(
                    ConstraintViolation(
                        "min_chat_messages",
                        "",
                        f"Only {kept_count} messages kept, minimum is {constraints.min_chat_messages}",
                    )
                )

        return violations

    def repair(
        self,
        result: "SelectionResult",
        conversation: "Conversation",
        constraints: ConstraintSet,
    ) -> "SelectionResult":
        """
        Promote messages from ``drop → summarize → keep_full`` until all
        constraint violations are resolved.

        In addition to explicit constraints, this method always resolves
        ``message.dependencies``: if a kept message has a dependency that was
        dropped, the dependency is promoted to ``keep_full`` (or ``summarize``
        if it already has a summary).  This ensures tool-call / tool-result
        pairs are never separated even when using algorithms that do not
        inherently understand dependencies.
        """
        from ..algorithms.base import SelectionResult

        keep_full = list(result.keep_full)
        summarize = list(result.summarize)
        drop = list(result.drop)

        def _promote(msg_id: str) -> None:
            """Move msg_id from drop/summarize to keep_full."""
            if msg_id in drop:
                drop.remove(msg_id)
                keep_full.append(msg_id)
            elif msg_id in summarize:
                summarize.remove(msg_id)
                keep_full.append(msg_id)

        if constraints.require_last_user:
            last_user = conversation.last_user_message()
            if last_user and (last_user.id in drop or last_user.id in summarize):
                _promote(last_user.id)

        if constraints.min_chat_messages > 0:
            while len(keep_full) + len(summarize) < constraints.min_chat_messages:
                if not drop:
                    break
                # Promote the first dropped message (oldest by position)
                promoted_id = drop.pop(0)
                keep_full.insert(0, promoted_id)

        # Resolve message.dependencies: ensure that anything kept has its
        # declared dependencies also kept.  Iterates until stable.
        dep_map: dict[str, list[str]] = {
            m.id: m.dependencies for m in conversation.messages
        }
        drop_set = set(drop)
        changed = True
        while changed:
            changed = False
            for msg_id in list(keep_full) + list(summarize):
                for dep_id in dep_map.get(msg_id, []):
                    if dep_id in drop_set:
                        dep_msg = conversation.get_by_id(dep_id)
                        drop_set.discard(dep_id)
                        if dep_msg and dep_msg.summary is not None:
                            summarize.append(dep_id)
                        else:
                            keep_full.append(dep_id)
                        changed = True
        drop = list(drop_set)

        return SelectionResult(
            keep_full=keep_full,
            summarize=summarize,
            drop=drop,
            estimated_tokens=result.estimated_tokens,
            estimated_cost_usd=result.estimated_cost_usd,
        )
