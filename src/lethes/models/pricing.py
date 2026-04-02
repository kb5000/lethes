"""Model pricing table — for cost estimation and cache-optimised selection."""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _normalize(s: str) -> str:
    """Lowercase and strip all non-alphanumeric chars, keeping ``*`` for glob patterns."""
    return re.sub(r"[^a-z0-9*]", "", s.lower())

if TYPE_CHECKING:
    import httpx

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


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
    Pricing table for LLM cost estimation.  Supports glob matching on
    ``model_id`` so a single entry like ``"gpt-4o*"`` covers all variants.

    Recommended usage — fetch live prices from OpenRouter::

        # sync
        table = ModelPricingTable.from_openrouter()

        # async (preferred in async code paths)
        table = await ModelPricingTable.from_openrouter_async()

    Or build from a custom list::

        table = ModelPricingTable.from_list([
            {"model_id": "my-model", "input_price_per_1m": 1.0,
             "cached_price_per_1m": 0.1, "output_price_per_1m": 3.0},
        ])
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
    def empty(cls) -> "ModelPricingTable":
        """Return an empty table (cost estimation always returns 0.0)."""
        return cls([])

    @classmethod
    def from_openrouter(
        cls,
        *,
        strip_provider_prefix: bool = True,
        timeout: float = 10.0,
    ) -> "ModelPricingTable":
        """Fetch live pricing from the OpenRouter ``/api/v1/models`` endpoint (sync).

        OpenRouter returns prices as USD *per token*; they are converted to the
        per-1 M token unit used internally.

        Parameters
        ----------
        strip_provider_prefix:
            When ``True`` (default) the ``provider/`` prefix is removed so that
            e.g. ``"openai/gpt-4o"`` is stored as ``"gpt-4o"``, matching the
            same way callers normally pass model IDs to OpenAI/Anthropic SDKs.
            Set to ``False`` to keep the full ``"provider/model"`` form.
        timeout:
            HTTP request timeout in seconds.

        Raises
        ------
        httpx.HTTPError
            On network or HTTP-level failures.  Callers that want a silent
            fallback can catch this and use :meth:`default` instead.
        """
        import httpx

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(_OPENROUTER_MODELS_URL)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        return cls._parse_openrouter_response(data, strip_provider_prefix=strip_provider_prefix)

    @classmethod
    async def from_openrouter_async(
        cls,
        *,
        strip_provider_prefix: bool = True,
        timeout: float = 10.0,
    ) -> "ModelPricingTable":
        """Async version of :meth:`from_openrouter`."""
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(_OPENROUTER_MODELS_URL)
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()

        return cls._parse_openrouter_response(data, strip_provider_prefix=strip_provider_prefix)

    @classmethod
    def _parse_openrouter_response(
        cls,
        data: dict[str, Any],
        *,
        strip_provider_prefix: bool,
    ) -> "ModelPricingTable":
        entries: list[ModelPricingEntry] = []
        for model in data.get("data", []):
            pricing = model.get("pricing") or {}
            prompt_str = pricing.get("prompt")
            completion_str = pricing.get("completion")
            if not prompt_str or not completion_str:
                continue

            try:
                # OpenRouter prices are USD per token; convert to USD per 1 M
                input_price = float(prompt_str) * 1_000_000
                output_price = float(completion_str) * 1_000_000
            except (ValueError, TypeError):
                continue

            cache_read_str = pricing.get("input_cache_read")
            try:
                cached_price = (
                    float(cache_read_str) * 1_000_000
                    if cache_read_str
                    else input_price * 0.1
                )
            except (ValueError, TypeError):
                cached_price = input_price * 0.1

            raw_id: str = model.get("id", "")
            if not raw_id:
                continue
            model_id = (
                raw_id.split("/", 1)[1] if strip_provider_prefix and "/" in raw_id else raw_id
            )

            entries.append(
                ModelPricingEntry(
                    model_id=model_id,
                    input_price_per_1m=input_price,
                    cached_price_per_1m=cached_price,
                    output_price_per_1m=output_price,
                )
            )
        return cls(entries)

    # ── Lookup ────────────────────────────────────────────────────────────

    def get(self, model_id: str) -> ModelPricingEntry | None:
        """Exact match → glob → normalized exact → normalized glob → ``None``."""
        # 1. Exact
        for entry in self._entries:
            if entry.model_id == model_id:
                return entry
        # 2. Glob
        for entry in self._entries:
            if fnmatch.fnmatch(model_id, entry.model_id):
                return entry
        # 3 & 4. Normalize both sides (lowercase, strip symbols, keep * for globs)
        norm = _normalize(model_id)
        for entry in self._entries:
            if _normalize(entry.model_id) == norm:
                return entry
        for entry in self._entries:
            if fnmatch.fnmatch(norm, _normalize(entry.model_id)):
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
