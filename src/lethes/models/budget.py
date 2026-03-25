"""Budget types — define how much context the orchestrator may use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Budget(Protocol):
    """
    Implement this protocol to define a custom budget policy.

    All concrete budgets are frozen dataclasses so they can be safely
    shared and replaced without mutation.
    """

    def is_exceeded(self, tokens: int, cost_usd: float) -> bool:
        """Return ``True`` if the given usage exceeds this budget."""
        ...

    def headroom_tokens(self, current_tokens: int) -> int:
        """
        How many *additional* tokens may still be added.
        Returns ``-1`` to signal *unlimited*.
        """
        ...


@dataclass(frozen=True)
class TokenBudget:
    """Hard limit on the number of tokens in the context window."""

    max_tokens: int
    """Set to ``0`` for unlimited."""

    def is_exceeded(self, tokens: int, cost_usd: float = 0.0) -> bool:
        return self.max_tokens > 0 and tokens > self.max_tokens

    def headroom_tokens(self, current_tokens: int) -> int:
        if self.max_tokens <= 0:
            return -1
        return max(0, self.max_tokens - current_tokens)


@dataclass(frozen=True)
class TokenTargetBudget:
    """
    Aim for a *specific* token count rather than enforcing a hard maximum.

    Semantically different from :class:`TokenBudget`: the intent is to fill
    the context window **as close to** ``target_tokens`` as possible, allowing
    a small ``overshoot`` to avoid stopping just short of the target on a
    message boundary.

    Use this when you want predictable, consistent context sizes turn-over-turn
    (e.g. always sending ~8 000 tokens to a model with a 16 k context window).

    Equivalent flag: ``!target=N``
    """

    target_tokens: int
    """Desired token count."""

    overshoot: int = 150
    """Allow going up to ``target_tokens + overshoot`` to avoid message splits."""

    def is_exceeded(self, tokens: int, cost_usd: float = 0.0) -> bool:
        return tokens > self.target_tokens + self.overshoot

    def headroom_tokens(self, current_tokens: int) -> int:
        return max(0, self.target_tokens + self.overshoot - current_tokens)


@dataclass(frozen=True)
class CostBudget:
    """Soft limit on estimated API cost per request (USD)."""

    max_cost_usd: float
    """Approximate dollars.  Set to ``0.0`` for unlimited."""

    price_per_token: float = 0.0
    """Used to convert token headroom ↔ cost headroom."""

    def is_exceeded(self, tokens: int, cost_usd: float) -> bool:
        return self.max_cost_usd > 0 and cost_usd > self.max_cost_usd

    def headroom_tokens(self, current_tokens: int) -> int:
        # Cannot derive token headroom without knowing the current cost,
        # so return unlimited here; cost is checked via is_exceeded.
        return -1


@dataclass(frozen=True)
class CompositeBudget:
    """Both the token budget **and** the cost budget must be satisfied."""

    token_budget: TokenBudget
    cost_budget: CostBudget | None = None

    def is_exceeded(self, tokens: int, cost_usd: float) -> bool:
        if self.token_budget.is_exceeded(tokens, cost_usd):
            return True
        if self.cost_budget and self.cost_budget.is_exceeded(tokens, cost_usd):
            return True
        return False

    def headroom_tokens(self, current_tokens: int) -> int:
        return self.token_budget.headroom_tokens(current_tokens)

    @classmethod
    def unlimited(cls) -> "CompositeBudget":
        return cls(token_budget=TokenBudget(max_tokens=0))
