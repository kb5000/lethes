"""
Open WebUI Filter integration — standalone version using installed lethes package.

Usage in Open WebUI (after `pip install lethes`):

    Copy this file into your Open WebUI functions directory.

Or import directly:

    from open_webui_filter import Filter
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.request
from typing import Any, Awaitable, Callable, Literal, Optional

import re as _re

from lethes.observability import configure_logging, get_logger, make_formatter

from pydantic import BaseModel, Field


def _strip_model_prefix(model_id: str) -> str:
    """Remove a leading ``identifier.`` prefix added by Open WebUI providers.

    Examples: ``poe.claude-3-5-sonnet`` → ``claude-3-5-sonnet``
              ``openrouter.openai/gpt-4o`` → ``openai/gpt-4o``
    """
    return _re.sub(r"^[A-Za-z0-9_-]+\.", "", model_id)


def _apply_model_aliases(raw_model_id: str, aliases_json: str) -> str:
    """Look up *raw_model_id* in the JSON alias map; return the mapped name or the
    original if no alias is defined.  Silently ignores malformed JSON."""
    try:
        aliases: dict = json.loads(aliases_json) if aliases_json.strip() else {}
    except Exception:
        aliases = {}
    return aliases.get(raw_model_id, raw_model_id)

from lethes.algorithms.dependency import DependencyAwareAlgorithm
from lethes.algorithms.greedy import GreedyByWeightAlgorithm
from lethes.algorithms.recency import RecencyBiasedAlgorithm
from lethes.cache.memory_backend import InMemoryCache
from lethes.cache.redis_backend import RedisCache
from lethes.engine.constraints import ConstraintSet
from lethes.engine.orchestrator import ContextOrchestrator
from lethes.models.budget import TokenBudget
from lethes.models.conversation import Conversation
from lethes.models.pricing import ModelPricingTable
from lethes.summarizers.levels import TurnSummarizer
from lethes.summarizers.llm import LLMSummarizer
from lethes.utils.tokens import TokenCounter
from lethes.weighting.composite import CompositeWeightStrategy
from lethes.weighting.keyword import KeywordRelevanceStrategy
from lethes.weighting.llm_analyzer import LLMContextAnalyzer
from lethes.weighting.smart import SmartWeightingStrategy
from lethes.weighting.static import StaticWeightStrategy

logger = get_logger("lethes.filter")


class _LethesObserverHandler(logging.Handler):
    """
    Minimal inline handler that POSTs structlog JSON events to a lethes-observer
    server. Non-blocking: uses a background daemon thread + bounded queue.
    All errors are silently swallowed. Zero external dependencies (stdlib only).
    """

    def __init__(self, url: str, timeout: float = 2.0) -> None:
        super().__init__()
        self._endpoint = url.rstrip("/") + "/ingest"
        self._timeout = timeout
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=500)
        self._thread = threading.Thread(target=self._worker, daemon=True, name="lethes-obs")
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(self.format(record))
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._queue.put(None, timeout=1.0)
            self._thread.join(timeout=2.0)
        except Exception:
            pass
        super().close()

    def _worker(self) -> None:
        while True:
            try:
                item = self._queue.get()
                if item is None:
                    break
                data = item.encode("utf-8")
                req = urllib.request.Request(
                    self._endpoint, data=data,
                    headers={"Content-Type": "application/json"}, method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout):
                    pass
            except Exception:
                pass


class Filter:
    """
    Plug-and-play Open WebUI filter that wraps
    :class:`~lethes.engine.orchestrator.ContextOrchestrator`.

    Configure via the ``Valves`` inner class (exposed by Open WebUI's settings UI).
    """

    class Valves(BaseModel):
        priority: int = Field(default=0, description="Filter execution priority")
        max_tokens: int = Field(
            default=10_000,
            description="Token limit for the context window. 0 = unlimited.",
        )
        max_turns: int = Field(
            default=25,
            description="Max conversation turns to keep. 0 = unlimited.",
        )
        algorithm: Literal[
            "greedy_by_weight",
            "recency_biased",
            "dependency_aware",
        ] = Field(
            default="recency_biased",
            description=(
                "Context selection algorithm. "
                "'recency_biased' (default) prioritises recent messages; "
                "'greedy_by_weight' picks by relevance score alone; "
                "'dependency_aware' enforces explicit dependency chains."
            ),
        )
        recency_factor: float = Field(
            default=1.5,
            description=(
                "Recency bias strength for 'recency_biased' algorithm. "
                "0 = no bias; 1.5 = newest message has 2.5× the weight of oldest. "
                "Increase for stronger recency preference."
            ),
        )
        weighting: Literal["static", "keyword", "smart"] = Field(
            default="smart",
            description=(
                "Message relevance weighting strategy. "
                "'smart' (default) combines keyword relevance, Q&A pair coherence, "
                "and tool-call penalties. "
                "'keyword' uses BM25/overlap only. "
                "'static' disables relevance scoring."
            ),
        )
        tool_penalty: float = Field(
            default=0.5,
            description=(
                "Weight multiplier for tool-call intermediate messages "
                "(role='tool' or assistant messages with tool_calls). "
                "0.5 = half the weight of regular messages. 1.0 = no penalty."
            ),
        )
        pair_coherence: float = Field(
            default=0.8,
            description=(
                "Fraction of a user message's relevance score transferred to its "
                "following assistant reply (0.0 – 1.0). "
                "Higher values keep Q&A pairs together more aggressively."
            ),
        )
        # ── Summarisation ─────────────────────────────────────────────────
        summary_api_base: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI-compatible API base URL for summarisation",
        )
        summary_api_key: str = Field(default="", description="API key for summarisation model")
        summary_model: str = Field(default="gpt-4o-mini", description="Model used for summarisation")
        summary_target_ratio: float = Field(default=0.3, description="Summarisation compression ratio")
        retry_attempts: int = Field(default=3, description="Retry attempts for summarisation")
        nosum_by_default: bool = Field(default=False, description="Disable summarisation globally")
        # ── LLM context analyser (sub-agent) ──────────────────────────────
        llm_analysis: bool = Field(
            default=False,
            description=(
                "Enable LLM-powered context analysis (sub-agent). "
                "When enabled, a fast LLM scores each message's importance for "
                "the current query. Uses summary_api_key and summary_model. "
                "Adds ~200–400 ms latency; results are cached."
            ),
        )
        llm_analysis_weight: float = Field(
            default=0.6,
            description=(
                "Weight of the LLM analysis score in the composite weighting "
                "(1 − llm_analysis_weight goes to the base weighting strategy). "
                "Only used when llm_analysis=True."
            ),
        )
        # ── Cache ─────────────────────────────────────────────────────────
        cache_backend: Literal["memory", "redis"] = Field(
            default="memory", description="Cache backend for summaries and embeddings"
        )
        redis_url: str = Field(
            default="redis://redis:6379/0", description="Redis URL (only used with redis cache)"
        )
        pricing_config_path: str = Field(
            default="", description="Path to custom pricing JSON (takes priority over OpenRouter fetch)"
        )
        use_openrouter_pricing: bool = Field(
            default=True,
            description=(
                "Fetch live model pricing from OpenRouter API. "
                "Falls back to bundled defaults if the fetch fails. "
                "Ignored when pricing_config_path is set."
            ),
        )
        pricing_cache_ttl_hours: float = Field(
            default=24.0,
            description=(
                "How long to cache fetched OpenRouter pricing data (hours). "
                "0 = re-fetch on every request (not recommended)."
            ),
        )
        # ── Observability ─────────────────────────────────────────────────
        observer_url: str = Field(
            default="",
            description=(
                "lethes-observer server URL (e.g. http://localhost:7456). "
                "Leave empty to disable structured log forwarding."
            ),
        )
        observer_log_level: Literal["DEBUG", "INFO", "WARNING"] = Field(
            default="DEBUG",
            description=(
                "Log level sent to the observer. "
                "DEBUG includes per-step pipeline details, weights and API call timings. "
                "INFO sends only pipeline start/selection/complete events."
            ),
        )
        # ── Model aliases ──────────────────────────────────────────────────
        model_aliases: str = Field(
            default="{}",
            description=(
                "JSON map of raw model IDs (before prefix stripping) to canonical names. "
                "Applied first, then the normal prefix-stripping runs on the result. "
                'Example: {"poe.gemini-3-flash": "gemini-3-flash-preview", '
                '"openrouter.meta-llama/llama-3-8b": "meta-llama/llama-3-8b"}'
            ),
        )

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._orchestrator: ContextOrchestrator | None = None
        self._orchestrator_config_key: tuple | None = None
        self._observer_configured: bool = False
        self._pricing_table: ModelPricingTable | None = None
        self._pricing_cache_time: float = 0.0
        # Telemetry
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._start_time: float | None = None
        self._user: str = "unknown"
        self._model_id: str = "unknown"
        self._input_message_count: int = 0
        self._run_id: str | None = None

    # ── Open WebUI hooks ──────────────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Pre-process: orchestrate the message list before sending to LLM."""
        self._maybe_setup_observer()
        self._start_time = time.time()
        self._user = (__user__ or {}).get("email", "unknown")
        raw_model_id = (__model__ or {}).get("id", "unknown")
        self._model_id = _strip_model_prefix(_apply_model_aliases(raw_model_id, self.valves.model_aliases))

        messages = body.get("messages", [])
        if not messages:
            return body

        conversation = Conversation.from_openai_messages(messages)

        pricing = await self._get_pricing_table()
        result = await self._get_orchestrator(pricing).process(
            conversation,
            model_id=self._model_id,
            event_emitter=__event_emitter__,
        )

        final_msgs = result.conversation.to_openai_messages()
        self._input_tokens = result.token_count
        self._input_message_count = len(final_msgs)
        self._run_id = result.run_id

        body["messages"] = final_msgs
        return body

    def outlet(
        self,
        body: dict,
        __model__: Optional[dict] = None,
    ) -> dict:
        """Post-process: attach usage stats to the response message."""
        messages = body.get("messages", [])
        if not messages:
            return body

        last_msg = messages[-1]

        # Prefer actual usage data from the LLM response body (e.g. OpenAI/OpenRouter
        # return prompt_tokens, completion_tokens, and optionally cost in usage).
        # Fall back to local token counting if not present.
        resp_usage: dict = body.get("usage") or last_msg.get("usage") or {}
        actual_prompt_tokens: int | None = resp_usage.get("prompt_tokens")
        actual_completion_tokens: int | None = resp_usage.get("completion_tokens")
        actual_cost_usd: float | None = resp_usage.get("cost")  # OpenRouter extension

        counter = TokenCounter()
        self._output_tokens = (
            actual_completion_tokens
            if actual_completion_tokens is not None
            else counter.count_dict(last_msg)
        )
        if actual_prompt_tokens is not None:
            self._input_tokens = actual_prompt_tokens

        if "usage" not in last_msg:
            last_msg["usage"] = {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }

        if self._start_time:
            elapsed = round(time.time() - self._start_time, 1)
            logger.info(
                "chat_turn",
                user=self._user,
                model=self._model_id,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                elapsed_s=elapsed,
                message_count=self._input_message_count,
            )
            # Log actual usage so the observer can update the run with real cost/tokens.
            if self._run_id:
                logger.info(
                    "pipeline.outlet",
                    run_id=self._run_id,
                    actual_output_tokens=self._output_tokens,
                    actual_input_tokens=self._input_tokens,
                    actual_cost_usd=actual_cost_usd,
                    from_response_body=resp_usage != {},
                )

        return body

    # ── Observer setup (lazy, once) ───────────────────────────────────────

    def _maybe_setup_observer(self) -> None:
        """Configure lethes structured logging to send to the observer server."""
        if self._observer_configured or not self.valves.observer_url:
            return
        handler = _LethesObserverHandler(self.valves.observer_url)
        configure_logging(level=self.valves.observer_log_level, handlers=[handler])
        self._observer_configured = True

    # ── Pricing (lazy, TTL-cached) ─────────────────────────────────────────

    async def _get_pricing_table(self) -> ModelPricingTable:
        """Return a cached pricing table, refreshing when the TTL has expired.

        Priority:
        1. Custom JSON file (``pricing_config_path``) — never cached, always re-read.
        2. OpenRouter live fetch (``use_openrouter_pricing``) — cached per TTL.
        3. Bundled defaults.
        """
        if self.valves.pricing_config_path:
            try:
                return ModelPricingTable.from_json(self.valves.pricing_config_path)
            except Exception:
                return ModelPricingTable.default()

        now = time.time()
        ttl = self.valves.pricing_cache_ttl_hours * 3600
        if self._pricing_table is not None and (now - self._pricing_cache_time) <= ttl:
            return self._pricing_table

        if self.valves.use_openrouter_pricing:
            try:
                self._pricing_table = await ModelPricingTable.from_openrouter_async()
                self._pricing_cache_time = now
                logger.info("pricing.refreshed", source="openrouter", models=len(self._pricing_table._entries))
            except Exception as exc:
                logger.warning("pricing.fetch_failed", error=str(exc), fallback="cached_or_default")
                if self._pricing_table is None:
                    self._pricing_table = ModelPricingTable.default()
        else:
            self._pricing_table = ModelPricingTable.default()
            self._pricing_cache_time = now

        return self._pricing_table

    # ── Orchestrator construction (lazy, config-keyed) ────────────────────

    def _get_orchestrator(self, pricing: ModelPricingTable) -> ContextOrchestrator:
        """Return (or rebuild) the orchestrator based on current Valves."""
        config_key = (
            self.valves.max_tokens,
            self.valves.algorithm,
            self.valves.recency_factor,
            self.valves.weighting,
            self.valves.tool_penalty,
            self.valves.pair_coherence,
            self.valves.summary_api_base,
            self.valves.summary_model,
            self.valves.cache_backend,
            self.valves.redis_url,
            self.valves.nosum_by_default,
            self.valves.llm_analysis,
            self.valves.llm_analysis_weight,
            id(pricing),  # rebuild when pricing table is refreshed
        )
        if self._orchestrator is not None and self._orchestrator_config_key == config_key:
            return self._orchestrator

        # Cache backend
        if self.valves.cache_backend == "redis":
            cache = RedisCache.from_url(self.valves.redis_url)
        else:
            cache = InMemoryCache()

        # Summariser (only if API key is set)
        turn_summarizer: TurnSummarizer | None = None
        if self.valves.summary_api_key and not self.valves.nosum_by_default:
            llm = LLMSummarizer(
                api_base=self.valves.summary_api_base,
                api_key=self.valves.summary_api_key,
                model=self.valves.summary_model,
                retry_attempts=self.valves.retry_attempts,
            )
            turn_summarizer = TurnSummarizer(
                backend=llm,
                cache=cache,
                target_ratio=self.valves.summary_target_ratio,
            )

        # Algorithm
        base_algo = GreedyByWeightAlgorithm()
        if self.valves.algorithm == "recency_biased":
            algo = RecencyBiasedAlgorithm(recency_factor=self.valves.recency_factor)
        elif self.valves.algorithm == "dependency_aware":
            algo = DependencyAwareAlgorithm(inner=base_algo)
        else:
            algo = base_algo

        # Base weighting strategy
        if self.valves.weighting == "keyword":
            base_weighting = KeywordRelevanceStrategy()
        elif self.valves.weighting == "smart":
            base_weighting = SmartWeightingStrategy(
                tool_penalty=self.valves.tool_penalty,
                pair_coherence=self.valves.pair_coherence,
            )
        else:
            base_weighting = StaticWeightStrategy()

        # Optionally wrap with LLM context analyser
        weighting = base_weighting
        if self.valves.llm_analysis and self.valves.summary_api_key:
            analyzer = LLMContextAnalyzer(
                api_base=self.valves.summary_api_base,
                api_key=self.valves.summary_api_key,
                model=self.valves.summary_model,
                cache=cache,
            )
            w = self.valves.llm_analysis_weight
            weighting = CompositeWeightStrategy([
                (base_weighting, 1.0 - w),
                (analyzer, w),
            ])

        self._orchestrator = ContextOrchestrator(
            budget=TokenBudget(max_tokens=self.valves.max_tokens),
            algorithm=algo,
            weighting=weighting,
            turn_summarizer=turn_summarizer,
            cache=cache,
            pricing_table=pricing,
            constraints=ConstraintSet(
                min_chat_messages=1,
                require_last_user=True,
            ),
        )
        self._orchestrator_config_key = config_key
        return self._orchestrator
