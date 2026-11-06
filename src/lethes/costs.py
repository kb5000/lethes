"""Renewal/replacement economics for an indefinitely continuing conversation.

The horizon is a cache cycle, bounded by capacity, not a predicted user departure.
Estimates assume constant measured growth and cache eligibility during a cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CachePricing:
    uncached_per_million: float
    cached_per_million: float
    minimum_cache_tokens: int = 1024
    cache_block_tokens: int = 128
    ttl_seconds: float = 300

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) and v >= 0
            for v in (
                self.uncached_per_million,
                self.cached_per_million,
            )
        ):
            raise ValueError("Prices must be finite and nonnegative")
        if self.minimum_cache_tokens < 0 or self.cache_block_tokens <= 0 or self.ttl_seconds <= 0:
            raise ValueError("Invalid cache eligibility settings")

    def eligible(self, tokens: int) -> int:
        if tokens < self.minimum_cache_tokens:
            return 0
        return tokens // self.cache_block_tokens * self.cache_block_tokens

    def cost(self, tokens: int, shared: int) -> float:
        cached = min(tokens, self.eligible(max(0, shared)))
        return (
            (tokens - cached) * self.uncached_per_million + cached * self.cached_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class RebuildDecision:
    rebuild: bool
    cycle_rounds: int
    average_cost: float
    keep_excess: float
    rebuild_excess: float


class RenewalPolicy:
    """Average-cost replacement test with the renewal rate as opportunity cost.

    First minimize C_new(T)/T over feasible cycle lengths T. Then compare the
    minimum excess cost C_action(T)-rate*T for keeping vs rebuilding now.
    Both actions return to the same future renewal process. Sunk costs are absent.
    """

    def __init__(self, pricing: CachePricing, margin: float = 0.05) -> None:
        if not 0 <= margin < 1:
            raise ValueError("Margin must be in [0, 1)")
        self.pricing = pricing
        self.margin = margin

    def _cycles(
        self, tokens: int, first_shared: int, growth: int, capacity: int, overhead: float = 0
    ) -> list[tuple[int, float]]:
        if tokens > capacity:
            return []
        result = []
        total = overhead
        for turn in range(1, (capacity - tokens) // growth + 2):
            size = tokens + (turn - 1) * growth
            shared = first_shared if turn == 1 else size - growth
            total += self.pricing.cost(size, shared)
            result.append((turn, total))
        return result

    def decide(
        self,
        *,
        old_tokens: int,
        new_tokens: int,
        old_shared: int,
        new_shared: int,
        growth: float,
        capacity: int,
        rebuild_overhead: float = 0,
    ) -> RebuildDecision:
        if growth <= 0 or not math.isfinite(growth):
            raise ValueError("Growth must be positive")
        step = max(1, math.ceil(growth))
        cold = self._cycles(new_tokens, 0, step, capacity, rebuild_overhead)
        if not cold:
            return RebuildDecision(False, 0, math.inf, math.inf, math.inf)
        rounds, total = min(cold, key=lambda pair: pair[1] / pair[0])
        rate = total / rounds
        keep = self._cycles(old_tokens, old_shared, step, capacity)
        rebuild = self._cycles(new_tokens, new_shared, step, capacity, rebuild_overhead)
        keep_excess = min((cost - rate * n for n, cost in keep), default=math.inf)
        new_excess = min(cost - rate * n for n, cost in rebuild)
        margin = max(rate * self.margin, 1e-12)
        return RebuildDecision(
            keep_excess > new_excess + margin, rounds, rate, keep_excess, new_excess
        )


def parse_usage(usage: dict) -> dict:
    """Keep provider measurements distinct from estimates. No fabricated zeros."""
    result = {"raw": usage}
    if "prompt_tokens" in usage:
        result["input_tokens"] = usage["prompt_tokens"]
    elif "input_tokens" in usage:
        # Anthropic input_tokens excludes cache read/write tokens.
        result["input_tokens"] = (
            usage["input_tokens"]
            + usage.get("cache_read_input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
        )
    details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
    if not isinstance(details, dict):
        details = {}
    cached = details.get("cached_tokens", usage.get("cache_read_input_tokens"))
    if cached is not None:
        result["cached_tokens"] = cached
    output = usage.get("completion_tokens", usage.get("output_tokens"))
    if output is not None:
        result["output_tokens"] = output
    return result
