"""Static (passthrough) weighting strategy — uses base_weight as-is."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.conversation import Conversation
    from ..models.message import Message


class StaticWeightStrategy:
    """
    No-op weighting: every message receives a relevance score of ``1.0``,
    so the final ``weight == base_weight``.

    This is the default strategy when no weighting is configured.
    It adds zero latency and requires no external services.
    """

    async def score(
        self,
        messages: list["Message"],
        query: str,
        conversation: "Conversation",
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        return {m.id: 1.0 for m in messages}

    def name(self) -> str:
        return "static"
