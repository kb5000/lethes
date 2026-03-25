"""Model pricing table — for cost estimation and cache-optimised selection."""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelPricingEntry:
    """Per-token prices for a single model (USD per 1 M tokens)."""

    model_id: str
    """Exact or glob pattern, e.g. ``"claude-3-5-sonnet*"``."""

    input_price_per_1m: float
    """Price for uncached input tokens."""

    cached_price_per_1m: float
    """Price for prefix-cache hit tokens (typically 10–20 % of input price)."""

    output_price_per_1m: float


class ModelPricingTable:
    """
    Loaded from a JSON file.  Supports glob matching on ``model_id``
    so a single entry like ``"gpt-4o*"`` covers all GPT-4o variants.
    """

    def __init__(self, entries: list[ModelPricingEntry]) -> None:
        self._entries = entries

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelPricingTable":
        with open(path, encoding="utf-8") as f:
            data: list[dict[str, Any]] = json.load(f)
        return cls.from_list(data)

    @classmethod
    def from_list(cls, data: list[dict[str, Any]]) -> "ModelPricingTable":
        entries = [
            ModelPricingEntry(
                model_id=d["model_id"],
                input_price_per_1m=float(d.get("input_price_per_1m", 0)),
                cached_price_per_1m=float(d.get("cached_price_per_1m", 0)),
                output_price_per_1m=float(d.get("output_price_per_1m", 0)),
            )
            for d in data
        ]
        return cls(entries)

    @classmethod
    def default(cls) -> "ModelPricingTable":
        """Load the bundled default pricing table."""
        default_path = Path(__file__).parent.parent / "config" / "pricing" / "default_pricing.json"
        if default_path.exists():
            return cls.from_json(default_path)
        return cls([])

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, model_id: str) -> ModelPricingEntry | None:
        """Exact match first, then glob, then ``None``."""
        # Exact
        for entry in self._entries:
            if entry.model_id == model_id:
                return entry
        # Glob
        for entry in self._entries:
            if fnmatch.fnmatch(model_id, entry.model_id):
                return entry
        return None

    # ── Cost estimation ───────────────────────────────────────────────────

    def estimate_cost(
        self,
        model_id: str,
        input_tokens: int,
        cached_tokens: int = 0,
        output_tokens: int = 0,
    ) -> float:
        """
        Return estimated USD cost.
        ``cached_tokens`` is the number of prefix-cache-hit tokens (cheaper).
        ``input_tokens`` should be the *total* input count; the cached portion
        is billed at the lower rate and the remainder at the standard rate.
        """
        entry = self.get(model_id)
        if entry is None:
            return 0.0
        uncached = max(0, input_tokens - cached_tokens)
        cost = (
            uncached * entry.input_price_per_1m
            + cached_tokens * entry.cached_price_per_1m
            + output_tokens * entry.output_price_per_1m
        ) / 1_000_000
        return cost
