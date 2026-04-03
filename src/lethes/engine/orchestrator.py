"""
ContextOrchestrator — the main engine.

Coordinates all layers: flag parsing → budget override → token counting →
dynamic weighting → algorithm selection → constraint repair →
summarisation → assembly → prefix tracking.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Awaitable

from ..algorithms.base import SelectionResult
from ..algorithms.greedy import GreedyByWeightAlgorithm
from ..cache.memory_backend import InMemoryCache
from ..engine.constraints import ConstraintChecker, ConstraintSet
from ..engine.planner import ContextPlan
from ..flags.schema import WellKnownFlag
from ..flags.session import SessionFlags
from ..models.budget import Budget, CompositeBudget, TokenBudget, TokenTargetBudget
from ..models.conversation import Conversation
from ..models.message import Message
from ..observability import get_logger
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

logger = get_logger(__name__)

# Generic status callback: (description, done) — framework-agnostic
StatusCallback = Callable[[str, bool], Awaitable[None]]


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
    run_id: str | None = None
    """Correlation ID for this pipeline run (matches observability log events)."""


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
        status_callback: StatusCallback | None = None,
    ) -> OrchestratorResult:
        """
        Run the full orchestration pipeline.

        Steps:

        1. Flag parsing — strip ``!flags`` from message content
        2. Budget override — honour ``full``, ``context=N``, ``target=N``, ``nosum`` flags
        3. Anchor pinning — honour ``recent=N``, ``keep_tag=label`` flags
        4. Token counting — fill ``message.token_count``
        5. Dynamic weighting — score messages (passes flag overrides as context)
        6. Algorithm selection — produce ``SelectionResult``
        7. Constraint validation + repair (also resolves dependencies)
        8. Summarisation — compress ``summarize`` messages (async, concurrent)
        9. Assembly — build the final conversation
        10. Prefix tracking — record the sent sequence
        """
        t0 = time.perf_counter()
        run_id = str(uuid.uuid4())
        # Bind run_id into the contextvars context so all sub-component loggers
        # (llm_analyzer, summarizer, embedding) automatically carry it.
        import structlog as _structlog
        _structlog.contextvars.bind_contextvars(run_id=run_id)

        log = logger.bind(
            model_id=model_id,
            session_id=str(conversation.session_id) if conversation.session_id else None,
            n_input=len(conversation.messages),
        )
        log.info("pipeline.start")

        # Step 1: Flag parsing
        session_flags, conversation = SessionFlags.from_conversation(conversation)

        # Apply well-known per-message flags to the last user message
        last_user = conversation.last_user_message()
        if last_user:
            updated = session_flags._apply_well_known_to_message(last_user)
            if updated is not last_user:
                conversation = conversation.replace(updated)

        effective = session_flags.effective_flags()
        log.debug("pipeline.flags", flags=[str(k) for k in effective.keys()])

        # Step 2: Budget override from flags
        budget = self._override_budget(effective)
        nosum = WellKnownFlag.NOSUM in session_flags

        # Step 3: Anchor pinning — !recent=N and !keep_tag=label
        conversation = self._apply_anchor_flags(conversation, effective)

        # Step 4: Token counting (fills .token_count on each message)
        messages_with_counts = self._token_counter.fill_counts(list(conversation.messages))
        conversation = conversation.with_messages(messages_with_counts)

        pre_plan_tokens = sum(m.token_count or 0 for m in conversation.messages)
        n_messages = len(conversation.messages)
        log.debug("pipeline.tokens", pre_plan_tokens=pre_plan_tokens, n_messages=n_messages)
        log.debug("pipeline.messages_in",
            messages=[_msg_summary(m) for m in conversation.messages],
        )

        if status_callback:
            await _emit_status(
                status_callback,
                f"Analysing context — {pre_plan_tokens} tokens, {n_messages} messages…",
                done=False,
            )

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
            log.info("pipeline.complete",
                mode="full_bypass",
                output_tokens=pre_plan_tokens,
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                kept=len(conversation.messages),
                dropped=0,
                summarized=0,
            )
            return self._build_result(conversation, plan, session_flags, model_id, run_id=run_id)

        # Step 5: Dynamic weighting — skip if budget is not exceeded (nothing to trim)
        last_user = conversation.last_user_message()
        query = last_user.get_text_content() if last_user else ""
        budget_exceeded = budget.is_exceeded(pre_plan_tokens, 0.0)
        if budget_exceeded:
            if status_callback:
                await _emit_status(status_callback, "Scoring message relevance…", done=False)
            weighting_context = _build_weighting_context(effective)
            scores = await self._weighting.score(
                list(conversation.messages), query, conversation, context=weighting_context
            )
            weighted_messages = apply_scores(list(conversation.messages), scores)
            conversation = conversation.with_messages(weighted_messages)
        else:
            log.debug("pipeline.weighting_skipped", reason="budget_not_exceeded", pre_plan_tokens=pre_plan_tokens)
            scores = {}
        if scores:
            vals = list(scores.values())
            log.debug("pipeline.weights",
                strategy=self._weighting.name(),
                n_scored=len(scores),
                weight_min=round(min(vals), 3),
                weight_max=round(max(vals), 3),
                weight_mean=round(sum(vals) / len(vals), 3),
            )

        # Step 6: Algorithm selection (sync)
        if self._prefix_tracker and conversation.session_id:
            await self._prefix_tracker.prepare(conversation.session_id)

        selection: SelectionResult = self._algorithm.select(
            conversation,
            budget,
            self._constraints,
            self._token_counter,
        )
        log.info("pipeline.selection",
            algorithm=self._algorithm.name(),
            keep=len(selection.keep_full),
            summarize=len(selection.summarize),
            drop=len(selection.drop),
            estimated_tokens=selection.estimated_tokens,
        )

        # Step 7: Constraint repair (always run — also resolves tool-pair dependencies)
        violations = self._constraint_checker.validate(
            selection, conversation, self._constraints
        )
        if violations:
            log.warning("pipeline.constraint_repair", violations=len(violations))
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
        if status_callback:
            if plan.total_dropped > 0:
                await _emit_status(
                    status_callback,
                    f"Trimming context — keeping {plan.total_kept}, compressing {plan.total_dropped} messages…",
                    done=False,
                )
            else:
                await _emit_status(
                    status_callback,
                    f"Context OK — {plan.post_plan_tokens} tokens, {plan.total_kept} messages",
                    done=True,
                )

        # Step 8: Summarisation
        if not nosum and self._turn_summarizer and plan.summarize:
            log.debug("pipeline.summarize_start", n_messages=len(plan.summarize))
            if status_callback:
                await _emit_status(
                    status_callback,
                    f"Summarising {len(plan.summarize)} messages…",
                    done=False,
                )
            conversation = await self._execute_summarization(
                conversation, plan, self._turn_summarizer
            )

        # Log messages_plan after summarisation so summary text is included.
        msg_plan: list[dict[str, Any]] = []
        for _m in conversation.messages:
            _info = _msg_summary(_m)
            if _m.id in plan.summarize:
                _info["disposition"] = "summarized"
                if _m.summary is not None:
                    _info["summary_tokens"] = self._token_counter.count_text(_m.summary)
            elif _m.id in plan.drop:
                _info["disposition"] = "dropped"
            else:
                _info["disposition"] = "kept"
            msg_plan.append(_info)
        log.debug("pipeline.messages_plan", messages=msg_plan)

        # Step 9: Assembly
        final_conversation = self._assemble(conversation, plan, nosum=nosum)

        # Emit done event (only needed when we emitted a non-done status above)
        if status_callback and plan.total_dropped > 0:
            await _emit_status(
                status_callback,
                f"Context ready — {plan.total_kept} messages kept, {plan.total_dropped} compressed.",
                done=True,
            )

        # Step 10: Prefix tracking
        if self._prefix_tracker and conversation.session_id:
            sent_ids = [m.id for m in final_conversation.messages]
            await self._prefix_tracker.record(conversation.session_id, sent_ids)

        result = self._build_result(final_conversation, plan, session_flags, model_id, run_id=run_id)
        log.info("pipeline.complete",
            output_tokens=result.token_count,
            cost_usd=result.estimated_cost_usd,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            kept=plan.total_kept,
            dropped=plan.total_dropped,
            summarized=len(plan.summarize),
        )
        return result

    # ── Internal steps ────────────────────────────────────────────────────

    def _override_budget(self, flags: dict) -> Budget:
        """Apply flag overrides to produce an effective budget."""
        if WellKnownFlag.FULL in flags:
            return TokenBudget(max_tokens=0)  # unlimited

        if WellKnownFlag.TARGET in flags:
            try:
                target = int(flags[WellKnownFlag.TARGET])  # type: ignore[arg-type]
                return TokenTargetBudget(target_tokens=target)
            except (TypeError, ValueError):
                logger.warning("invalid_flag_value", flag="target", value=flags[WellKnownFlag.TARGET])

        if WellKnownFlag.CONTEXT in flags:
            try:
                max_turns = int(flags[WellKnownFlag.CONTEXT])  # type: ignore[arg-type]
                return _TurnLimitBudget(max_turns=max_turns)
            except (TypeError, ValueError):
                logger.warning("invalid_flag_value", flag="context", value=flags[WellKnownFlag.CONTEXT])

        return self._budget

    @staticmethod
    def _apply_anchor_flags(
        conversation: Conversation, flags: dict
    ) -> Conversation:
        """
        Pin messages based on ``recent=N`` and ``keep_tag=label`` flags.

        * ``recent=N`` — pins the last N non-system messages so the algorithm
          never drops or summarises them.
        * ``keep_tag=label`` — pins every message that carries the given tag.
        """
        updated = list(conversation.messages)
        changed = False

        # !keep_tag=label — pin all messages with that tag
        keep_tag = flags.get(str(WellKnownFlag.KEEP_TAG))
        if keep_tag:
            for i, m in enumerate(updated):
                if keep_tag in m.tags and not m.pinned:
                    updated[i] = dataclasses.replace(m, pinned=True)
                    changed = True

        # !recent=N — pin last N chat messages
        recent_val = flags.get(str(WellKnownFlag.RECENT))
        if recent_val is not None:
            try:
                n_recent = int(recent_val)
                chat_indices = [
                    i for i, m in enumerate(updated) if m.role != "system"
                ]
                to_pin = set(chat_indices[-n_recent:]) if n_recent > 0 else set()
                for i in to_pin:
                    if not updated[i].pinned:
                        updated[i] = dataclasses.replace(updated[i], pinned=True)
                        changed = True
            except (TypeError, ValueError):
                logger.warning("invalid_flag_value", flag="recent", value=recent_val)

        return conversation.with_messages(updated) if changed else conversation

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
        run_id: str | None = None,
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
            run_id=run_id,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _msg_summary(m: Message, preview_len: int = 180) -> dict[str, Any]:
    """Compact message descriptor for observability log events."""
    text = m.get_text_content()
    preview = text[:preview_len] + ("…" if len(text) > preview_len else "")
    d: dict[str, Any] = {
        "id": m.id,
        "role": m.role,
        "tokens": m.token_count or 0,
        "preview": preview,
        "content": text,
        "weight": round(m.weight, 3),
    }
    if m.pinned:
        d["pinned"] = True
    if m.tool_calls:
        d["tool_calls"] = [
            tc.get("function", {}).get("name", "?") for tc in m.tool_calls
        ]
    if m.summary is not None:
        d["summary"] = m.summary
    return d


def _build_weighting_context(flags: dict) -> dict[str, Any]:
    """
    Extract flag values that weighting strategies can consume via their
    ``context`` parameter.  Currently passes through:

    * ``tool_penalty`` — overrides :attr:`~lethes.weighting.smart.SmartWeightingStrategy.tool_penalty`
    * ``pair_coherence`` — overrides :attr:`~lethes.weighting.smart.SmartWeightingStrategy.pair_coherence`
    """
    ctx: dict[str, Any] = {}
    for flag_name, ctx_key in [
        (WellKnownFlag.TOOL_PENALTY, "tool_penalty"),
        (WellKnownFlag.PAIR_COHERENCE, "pair_coherence"),
    ]:
        val = flags.get(str(flag_name))
        if val is not None:
            try:
                ctx[ctx_key] = float(val)
            except (TypeError, ValueError):
                pass
    return ctx


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


async def _emit_status(callback: StatusCallback, description: str, done: bool) -> None:
    try:
        await callback(description, done)
    except Exception:
        pass
