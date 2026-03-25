"""
Open WebUI Filter integration — drop-in replacement for example.py's Filter.

Usage in Open WebUI:

    from lethes.integrations.open_webui import OpenWebUIFilter as Filter

Or copy just this file into your Open WebUI functions directory.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable, Literal, Optional

from pydantic import BaseModel, Field

from ..algorithms.dependency import DependencyAwareAlgorithm
from ..algorithms.greedy import GreedyByWeightAlgorithm
from ..algorithms.recency import RecencyBiasedAlgorithm
from ..cache.memory_backend import InMemoryCache
from ..cache.redis_backend import RedisCache
from ..engine.constraints import ConstraintSet
from ..engine.orchestrator import ContextOrchestrator
from ..models.budget import TokenBudget
from ..models.conversation import Conversation
from ..models.pricing import ModelPricingTable
from ..summarizers.levels import TurnSummarizer
from ..summarizers.llm import LLMSummarizer
from ..utils.tokens import TokenCounter
from ..weighting.keyword import KeywordRelevanceStrategy
from ..weighting.static import StaticWeightStrategy

logger = logging.getLogger(__name__)


class OpenWebUIFilter:
    """
    Plug-and-play Open WebUI filter that wraps :class:`~lethes.engine.orchestrator.ContextOrchestrator`.

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
        ] = Field(default="greedy_by_weight", description="Context selection algorithm")
        recency_factor: float = Field(
            default=2.0,
            description="Recency bias factor (only used with recency_biased algorithm)",
        )
        weighting: Literal["static", "keyword"] = Field(
            default="keyword",
            description="Message relevance weighting strategy",
        )
        summary_api_base: str = Field(
            default="https://api.openai.com/v1",
            description="OpenAI-compatible API base URL for summarisation",
        )
        summary_api_key: str = Field(default="", description="API key for summarisation model")
        summary_model: str = Field(default="gpt-4o-mini", description="Model used for summarisation")
        summary_target_ratio: float = Field(default=0.3, description="Summarisation compression ratio")
        retry_attempts: int = Field(default=3, description="Retry attempts for summarisation")
        cache_backend: Literal["memory", "redis"] = Field(
            default="memory", description="Cache backend for summaries"
        )
        redis_url: str = Field(
            default="redis://redis:6379/0", description="Redis URL (only used with redis cache)"
        )
        pricing_config_path: str = Field(
            default="", description="Path to custom pricing JSON (leave empty for defaults)"
        )
        nosum_by_default: bool = Field(default=False, description="Disable summarisation globally")

    def __init__(self) -> None:
        self.valves = self.Valves()
        self._orchestrator: ContextOrchestrator | None = None
        self._orchestrator_config_key: tuple | None = None
        # Telemetry
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._start_time: float | None = None
        self._user: str = "unknown"
        self._model_id: str = "unknown"
        self._input_message_count: int = 0

    # ── Open WebUI hooks ──────────────────────────────────────────────────

    async def inlet(
        self,
        body: dict,
        __event_emitter__: Callable[[Any], Awaitable[None]],
        __model__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        """Pre-process: orchestrate the message list before sending to LLM."""
        self._start_time = time.time()
        self._user = (__user__ or {}).get("email", "unknown")
        self._model_id = (__model__ or {}).get("id", "unknown")

        messages = body.get("messages", [])
        if not messages:
            return body

        conversation = Conversation.from_openai_messages(messages)
        orchestrator = self._get_orchestrator()

        result = await orchestrator.process(
            conversation,
            model_id=self._model_id,
            event_emitter=__event_emitter__,
        )

        final_msgs = result.conversation.to_openai_messages()
        self._input_tokens = result.token_count
        self._input_message_count = len(final_msgs)

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
        counter = TokenCounter()
        self._output_tokens = counter.count_dict(last_msg)

        if "usage" not in last_msg:
            last_msg["usage"] = {
                "prompt_tokens": self._input_tokens,
                "completion_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }

        if self._start_time:
            elapsed = round(time.time() - self._start_time, 1)
            logger.info(
                '{"log_type":"chat_turn","user":"%s","model":"%s",'
                '"input_tokens":%d,"output_tokens":%d,"elapsed_s":%s,'
                '"message_count":%d}',
                self._user,
                self._model_id,
                self._input_tokens,
                self._output_tokens,
                elapsed,
                self._input_message_count,
            )

        return body

    # ── Orchestrator construction (lazy, config-keyed) ────────────────────

    def _get_orchestrator(self) -> ContextOrchestrator:
        """Return (or rebuild) the orchestrator based on current Valves."""
        config_key = (
            self.valves.max_tokens,
            self.valves.algorithm,
            self.valves.recency_factor,
            self.valves.weighting,
            self.valves.summary_api_base,
            self.valves.summary_model,
            self.valves.cache_backend,
            self.valves.redis_url,
            self.valves.nosum_by_default,
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

        # Weighting
        if self.valves.weighting == "keyword":
            weighting = KeywordRelevanceStrategy()
        else:
            weighting = StaticWeightStrategy()

        # Pricing
        pricing: ModelPricingTable | None = None
        if self.valves.pricing_config_path:
            try:
                pricing = ModelPricingTable.from_json(self.valves.pricing_config_path)
            except Exception:
                pricing = ModelPricingTable.default()
        else:
            pricing = ModelPricingTable.default()

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
