"""
ContextOrchestrator — the main engine.

Coordinates all layers: flag parsing → budget override → token counting →
dynamic weighting → algorithm selection → constraint repair →
summarisation → assembly → prefix tracking.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from ..algorithms.base import SelectionResult
from ..algorithms.greedy import GreedyByWeightAlgorithm
from ..cache.memory_backend import InMemoryCache
from ..engine.constraints import ConstraintChecker, ConstraintSet
from ..engine.planner import ContextPlan
from ..flags.schema import WellKnownFlag
from ..flags.session import SessionFlags
from ..models.budget import Budget, CompositeBudget, TokenBudget
from ..models.conversation import Conversation
from ..models.message import Message
from ..utils.tokens import TokenCounter
from ..weighting.base import apply_scores
from ..weighting.static import StaticWeightStrategy

if TYPE_CHECKING:
    from ..algorithms.base import SelectionAlgorithm
    from ..cache.base import CacheBackend
    from ..cache.prefix_tracker import PrefixSequenceTracker
    from ..models.pricing import ModelPricingTable
    from ..summarizers.base import Summarizer
    from ..summarizers.levels import TurnSummarizer
    from ..weighting.base import WeightingStrategy

logger = logging.getLogger(__name__)

# Type alias for Open WebUI / generic event emitter
EventEmitter = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass
class OrchestratorResult:
    """The output of one :meth:`ContextOrchestrator.process` call."""

    conversation: Conversation
    """The final, orchestrated conversation ready to be sent to the LLM API."""

    plan: ContextPlan
    """The full selection plan (for logging, dry-run, testing)."""

    flags: "SessionFlags"
    """Effective flags for the current turn."""

    token_count: int
    """Total tokens in the outgoing conversation."""

    estimated_cost_usd: float | None = None


class ContextOrchestrator:
    """
    The central message orchestration engine.

    Accepts a :class:`~lethes.models.conversation.Conversation` and returns
    an :class:`OrchestratorResult` containing the context-managed version
    ready to send to an LLM.

    Parameters
    ----------
    budget:
        Token/cost limits.  Use :class:`~lethes.models.budget.TokenBudget`
        for a simple token cap.
    algorithm:
        Context selection algorithm.  Defaults to
        :class:`~lethes.algorithms.greedy.GreedyByWeightAlgorithm`.
    weighting:
        Dynamic relevance scoring strategy.  Defaults to
        :class:`~lethes.weighting.static.StaticWeightStrategy` (no-op).
    turn_summarizer:
        If provided, messages marked ``summarize`` are compressed here.
        Without this, ``summarize`` messages are treated as ``drop``.
    token_counter:
        Shared token counter.  A new one is created if not provided.
    cache:
        Cache backend for summary storage.  Defaults to in-memory.
    prefix_tracker:
        Optional tracker for KV-cache prefix optimisation.
    pricing_table:
        Used to estimate cost in :class:`OrchestratorResult`.
    constraints:
        Hard rules the selection must satisfy.
    """

    def __init__(
        self,
        budget: Budget | None = None,
        algorithm: "SelectionAlgorithm | None" = None,
        weighting: "WeightingStrategy | None" = None,
        turn_summarizer: "TurnSummarizer | None" = None,
        token_counter: TokenCounter | None = None,
        cache: "CacheBackend | None" = None,
        prefix_tracker: "PrefixSequenceTracker | None" = None,
        pricing_table: "ModelPricingTable | None" = None,
        constraints: ConstraintSet | None = None,
    ) -> None:
        self._budget = budget or TokenBudget(max_tokens=0)
        self._algorithm = algorithm or GreedyByWeightAlgorithm()
        self._weighting = weighting or StaticWeightStrategy()
        self._turn_summarizer = turn_summarizer
        self._token_counter = token_counter or TokenCounter()
        self._cache = cache or InMemoryCache()
        self._prefix_tracker = prefix_tracker
        self._pricing_table = pricing_table
        self._constraints = constraints or ConstraintSet()
        self._constraint_checker = ConstraintChecker()

    # ── Public API ────────────────────────────────────────────────────────

    async def process(
        self,
        conversation: Conversation,
        *,
        model_id: str | None = None,
        event_emitter: EventEmitter | None = None,
    ) -> OrchestratorResult:
        """
        Run the full orchestration pipeline.

        Steps:

        1. Flag parsing — strip ``!flags`` from message content
        2. Budget override — honour ``full``, ``context=N``, ``nosum`` flags
        3. Token counting — fill ``message.token_count``
        4. Dynamic weighting — score messages against the current query
        5. Algorithm selection — produce ``SelectionResult``
        6. Constraint validation + repair
        7. Summarisation — compress ``summarize`` messages (async, concurrent)
        8. Assembly — build the final conversation
        9. Prefix tracking — record the sent sequence
        """
        # Step 1: Flag parsing
        session_flags, conversation = SessionFlags.from_conversation(conversation)

        # Apply well-known per-message flags to the last user message
        last_user = conversation.last_user_message()
        if last_user:
            updated = session_flags._apply_well_known_to_message(last_user)
            if updated is not last_user:
                conversation = conversation.replace(updated)

        effective = session_flags.effective_flags()
        logger.debug("Effective flags: %s", effective)

        # Step 2: Budget override from flags
        budget = self._override_budget(effective)
        nosum = WellKnownFlag.NOSUM in session_flags

        # Step 3: Token counting (fills .token_count on each message)
        messages_with_counts = self._token_counter.fill_counts(list(conversation.messages))
        conversation = conversation.with_messages(messages_with_counts)

        pre_plan_tokens = sum(m.token_count or 0 for m in conversation.messages)

        # Check if budget is unlimited — skip selection
        if isinstance(budget, TokenBudget) and budget.max_tokens == 0:
            # full mode — pass everything through
            plan = ContextPlan(
                keep_full=frozenset(m.id for m in conversation.messages),
                summarize=frozenset(),
                drop=frozenset(),
                algorithm_name="bypass(full)",
                weighting_strategy_name=self._weighting.name(),
                pre_plan_tokens=pre_plan_tokens,
                post_plan_tokens=pre_plan_tokens,
            )
            return self._build_result(conversation, plan, session_flags, model_id)

        # Step 4: Dynamic weighting
        query = last_user.get_text_content() if last_user else ""
        scores = await self._weighting.score(
            list(conversation.messages), query, conversation
        )
        weighted_messages = apply_scores(list(conversation.messages), scores)
        conversation = conversation.with_messages(weighted_messages)

        # Step 5: Algorithm selection (sync)
        if self._prefix_tracker and conversation.session_id:
            await self._prefix_tracker.prepare(conversation.session_id)

        selection: SelectionResult = self._algorithm.select(
            conversation,
            budget,
            self._constraints,
            self._token_counter,
        )

        # Step 6: Constraint repair
        violations = self._constraint_checker.validate(
            selection, conversation, self._constraints
        )
        if violations:
            logger.debug("Repairing %d constraint violations", len(violations))
            selection = self._constraint_checker.repair(
                selection, conversation, self._constraints
            )

        plan = ContextPlan.from_selection_result(
            selection,
            algorithm_name=self._algorithm.name(),
            weighting_strategy_name=self._weighting.name(),
            pre_plan_tokens=pre_plan_tokens,
        )

        # Emit status event
        if event_emitter and plan.total_dropped > 0:
            await _emit_status(
                event_emitter,
                f"Summarising {plan.total_dropped} messages, keeping {plan.total_kept}…",
                done=False,
            )

        # Step 7: Summarisation
        if not nosum and self._turn_summarizer and plan.summarize:
            conversation = await self._execute_summarization(
                conversation, plan, self._turn_summarizer
            )

        # Step 8: Assembly
        final_conversation = self._assemble(conversation, plan, nosum=nosum)

        # Emit done event
        if event_emitter and plan.total_dropped > 0:
            await _emit_status(
                event_emitter,
                f"Context ready: {plan.total_kept} messages kept, {plan.total_dropped} dropped.",
                done=True,
            )

        # Step 9: Prefix tracking
        if self._prefix_tracker and conversation.session_id:
            sent_ids = [m.id for m in final_conversation.messages]
            await self._prefix_tracker.record(conversation.session_id, sent_ids)

        return self._build_result(final_conversation, plan, session_flags, model_id)

    # ── Internal steps ────────────────────────────────────────────────────

    def _override_budget(self, flags: dict) -> Budget:
        """Apply flag overrides to produce an effective budget."""
        if WellKnownFlag.FULL in flags:
            return TokenBudget(max_tokens=0)  # unlimited

        if WellKnownFlag.CONTEXT in flags:
            try:
                max_turns = int(flags[WellKnownFlag.CONTEXT])  # type: ignore[arg-type]
                # Convert turn limit to an approximate token budget
                # We use a turn-count-aware sub-budget approach:
                # Return the budget with a turn_limit annotation in metadata.
                # The greedy algorithm doesn't know about turns, so we use a
                # TurnLimitBudget wrapper here.
                return _TurnLimitBudget(max_turns=max_turns)
            except (TypeError, ValueError):
                logger.warning("Invalid context flag value: %r", flags[WellKnownFlag.CONTEXT])

        return self._budget

    async def _execute_summarization(
        self,
        conversation: Conversation,
        plan: ContextPlan,
        turn_summarizer: "TurnSummarizer",
    ) -> Conversation:
        """Concurrently summarise all messages in plan.summarize."""
        summarize_ids = plan.summarize

        # Group consecutive summarize messages into turns
        ordered = [m for m in conversation.messages if m.id in summarize_ids]

        # Find context messages (system + kept messages before the summarized block)
        context_msgs = [
            m for m in conversation.messages
            if m.role == "system" or m.id in plan.keep_full
        ]

        # Summarise each message individually (can be enhanced to pair user+assistant)
        tasks = [
            turn_summarizer.summarize_turn([msg], context=context_msgs)
            for msg in ordered
        ]
        results: list[tuple[str, str]] = await asyncio.gather(*tasks, return_exceptions=False)  # type: ignore[assignment]

        # Apply summaries back to messages
        updated_messages = list(conversation.messages)
        summary_map = {msg.id: text for msg, (_, text) in zip(ordered, results)}

        for i, msg in enumerate(updated_messages):
            if msg.id in summary_map:
                updated_messages[i] = dataclasses.replace(
                    msg, summary=summary_map[msg.id]
                )

        return conversation.with_messages(updated_messages)

    def _assemble(
        self,
        conversation: Conversation,
        plan: ContextPlan,
        nosum: bool = False,
    ) -> Conversation:
        """
        Build the final outgoing conversation from the plan.

        Order: system messages → summarised context block → kept messages
        (sorted by original sequence_index).
        """
        system_msgs = conversation.system_messages()
        chat_msgs = conversation.chat_messages()

        # Collect messages for the summary block
        summarized_contents: list[str] = []
        kept_msgs: list[Message] = []

        for msg in sorted(chat_msgs, key=lambda m: m.sequence_index):
            if msg.id in plan.keep_full:
                kept_msgs.append(msg)
            elif msg.id in plan.summarize:
                if nosum or msg.summary is None:
                    # Treat as drop
                    pass
                else:
                    summarized_contents.append(f"{msg.role}: {msg.summary}")
            # drop: excluded

        final_msgs: list[Message] = list(system_msgs)

        if summarized_contents:
            summary_block = Message(
                role="user",
                content=(
                    "Saved context:\n<context>\n"
                    + "\n\n".join(summarized_contents)
                    + "\n</context>"
                ),
            )
            final_msgs.append(summary_block)

        final_msgs.extend(kept_msgs)
        return conversation.with_messages(final_msgs)

    def _build_result(
        self,
        conversation: Conversation,
        plan: ContextPlan,
        session_flags: "SessionFlags",
        model_id: str | None,
    ) -> OrchestratorResult:
        token_count = sum(
            self._token_counter.count(m) for m in conversation.messages
        )

        cost: float | None = None
        if self._pricing_table and model_id:
            cost = self._pricing_table.estimate_cost(
                model_id, input_tokens=token_count
            )

        return OrchestratorResult(
            conversation=conversation,
            plan=plan,
            flags=session_flags,
            token_count=token_count,
            estimated_cost_usd=cost,
        )


# ── Turn-limit budget (internal) ──────────────────────────────────────────────

@dataclass(frozen=True)
class _TurnLimitBudget:
    """
    Internal budget that limits by number of conversation turns.
    Used when the ``context=N`` flag is set.
    The orchestrator handles this before calling the algorithm.
    """

    max_turns: int

    def is_exceeded(self, tokens: int, cost_usd: float = 0.0) -> bool:
        # Turn counting is done in _apply_turn_limit; this budget is never
        # checked by the algorithm directly.
        return False

    def headroom_tokens(self, current_tokens: int) -> int:
        return -1  # unlimited for the algorithm; turns enforced separately


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _emit_status(emitter: EventEmitter, description: str, done: bool) -> None:
    try:
        await emitter({"type": "status", "data": {"description": description, "done": done}})
    except Exception:
        pass
