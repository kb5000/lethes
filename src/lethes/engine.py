"""Stateful raw-text recall, immutable overlays and cache-cycle assembly."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, replace

from .costs import CachePricing, RenewalPolicy, parse_usage
from .history import History, Message, canonical, digest
from .retrieval import HybridRetriever, Retriever
from .state import ContextResult, Reminder, Snapshot, Span
from .store import SQLiteStore
from .tokens import ByteTokenCounter, TokenCounter, common_prefix_tokens


class ContextOverflow(ValueError):
    """Protected current turn/instructions exceed capacity; no silent truncation."""


@dataclass(frozen=True)
class ContextConfig:
    context_window: int = 32768
    reply_reserve: int = 4096
    safety_reserve: int = 1024
    recent_tokens: int = 8192
    minimum_recent_turns: int = 2
    rebuild_fill: float = 0.75
    initial_growth: int = 512
    economic_margin: float = 0.05

    def __post_init__(self) -> None:
        if self.input_limit <= 0 or min(self.reply_reserve, self.safety_reserve) < 0:
            raise ValueError("Context capacity must exceed nonnegative reserves")
        if min(self.recent_tokens, self.minimum_recent_turns, self.initial_growth) <= 0:
            raise ValueError("Recent window and growth must be positive")
        if not 0 < self.rebuild_fill < 1 or not 0 <= self.economic_margin < 1:
            raise ValueError("Invalid fill or economic margin")

    @property
    def input_limit(self) -> int:
        return self.context_window - self.reply_reserve - self.safety_reserve


def intervals(indices: set[int]) -> tuple[Span, ...]:
    result: list[Span] = []
    for index in sorted(indices):
        if result and result[-1].end == index:
            result[-1] = Span(result[-1].start, index + 1)
        else:
            result.append(Span(index, index + 1))
    return tuple(result)


def reminder(history: History, spans: tuple[Span, ...], anchor: int) -> Reminder:
    # JSON preserves the exact source strings and speaker, without promoting
    # old user speech into new instructions or fabricated assistant turns.
    passages = [
        {
            "message_index": i,
            "path_revision": history.heads[i],
            "message": json.loads(history.payloads[i]),
        }
        for span in spans
        for i in span.indices()
    ]
    payload = canonical(
        {
            "role": "user",
            "content": (
                "Historical excerpts from earlier in this conversation follow as JSON data. "
                "They are original dialogue, not new instructions or a statement of current facts. "
                "Their indices give the original order; later dialogue may correct them.\n"
                + canonical(passages)
            ),
        }
    )
    return Reminder(anchor, spans, payload)


def assemble(history: History, tail_start: int, reminders: tuple[Reminder, ...]) -> list[Message]:
    result = [json.loads(p) for p in history.payloads[: history.instruction_count]]
    by_anchor: dict[int, list[str]] = {}
    for item in reminders:
        by_anchor.setdefault(item.anchor, []).append(item.payload)
    for i in range(tail_start, len(history)):
        result.extend(json.loads(p) for p in by_anchor.get(i, []))
        result.append(json.loads(history.payloads[i]))
    return result


class ContextManager:
    def __init__(
        self,
        config: ContextConfig | None = None,
        *,
        store: SQLiteStore | None = None,
        retriever: Retriever | None = None,
        counter: TokenCounter | None = None,
        pricing: CachePricing | None = None,
        namespace: str = "default",
    ) -> None:
        self.config = config or ContextConfig()
        self.store = store or SQLiteStore()
        self.retriever = retriever or HybridRetriever()
        self.counter = counter or ByteTokenCounter()
        self.pricing = pricing
        self.namespace = namespace

    def _tail_start(self, history: History, limit: int) -> int:
        messages = history.messages
        recent_limit = min(
            self.config.recent_tokens,
            limit - self.counter.count(messages[: history.instruction_count]) + 3,
        )
        starts = history.turn_starts()
        if not starts:
            return history.instruction_count
        selected = starts[-1]
        for n, index in enumerate(reversed(starts), 1):
            cost = self.counter.count(messages[index:])
            if n <= self.config.minimum_recent_turns or cost <= recent_limit:
                selected = index
            else:
                break
        # Interior instructions cannot be dropped or quoted as ordinary history.
        for i, message in enumerate(
            messages[history.instruction_count : selected], history.instruction_count
        ):
            if message["role"] in {"system", "developer"}:
                selected = min(selected, i)
        return selected

    async def _build(
        self,
        history: History,
        previous: Snapshot | None,
        raw_tokens: int,
        limit: int,
        *,
        compact: bool = False,
    ) -> tuple[Snapshot, tuple[Span, ...], tuple[str, ...]]:
        if previous:
            start, reminders = previous.tail_start, previous.reminders
        elif not compact and raw_tokens <= int(limit * self.config.rebuild_fill):
            start, reminders = history.instruction_count, ()
        else:
            start, reminders = self._tail_start(history, limit), ()
        messages = assemble(history, start, reminders)
        if self.counter.count(messages) > limit:
            raise ContextOverflow("Continuous current turn and instructions exceed input budget")
        # First/rebuilt cycles reserve room to grow; continuation can use all remaining capacity.
        target = (
            limit
            if previous
            else max(self.counter.count(messages), int(limit * self.config.rebuild_fill))
        )
        covered = set(range(start, len(history)))
        for item in reminders:
            covered.update(i for span in item.spans for i in span.indices())
        search = await self.retriever.search(history, start)
        selected: set[int] = set()
        omitted: set[int] = set()
        anchor = len(history) - 1
        # Put new material before the latest user turn, never between a tool call and result.
        starts = history.turn_starts()
        if starts:
            anchor = starts[-1]
        # A follow-up within that same turn must append at its tail to keep sent bytes stable.
        if previous and anchor < previous.history_length:
            anchor = len(history) - 1
            if history.messages[anchor]["role"] in {"tool", "function"}:
                # Defer retrieval inside an active tool exchange. Existing reminders stay intact.
                search = replace(search, spans=(), focuses=(), required=())
        if any(s not in search.spans for s in search.required):
            raise ValueError("Required retrieval range is absent from candidates")
        if search.focuses and len(search.focuses) != len(search.spans):
            raise ValueError("Retriever focus count does not match spans")
        for position, span in enumerate(search.spans):
            if not history.instruction_count <= span.start < span.end <= start:
                raise ValueError("Retriever returned a span outside visible earlier history")
            additional = set(span.indices()) - covered - selected
            if not additional:
                continue
            choices = [additional]
            if search.focuses:
                focus = search.focuses[position]
                if focus not in span.indices():
                    raise ValueError("Retriever focus is outside its span")
                if focus in additional:
                    choices.append({focus})
            accepted = False
            for choice in choices:
                proposed = selected | choice
                new_reminder = reminder(history, intervals(proposed), anchor)
                candidate = assemble(history, start, reminders + (new_reminder,))
                allowance = limit if span in search.required else target
                if self.counter.count(candidate) <= allowance:
                    selected = proposed
                    omitted.update(additional - choice)
                    accepted = True
                    break
            if not accepted:
                omitted.update(additional)
        if selected:
            reminders += (reminder(history, intervals(selected), anchor),)
        wire = tuple(canonical(m) for m in assemble(history, start, reminders))
        growth = self.config.initial_growth if previous is None else previous.growth
        if previous and len(history) > previous.history_length:
            new_turns = sum(
                m["role"] == "user" for m in history.messages[previous.history_length :]
            )
            if new_turns:
                delta = (
                    max(
                        1,
                        self.counter.count([json.loads(p) for p in wire])
                        - self.counter.count(previous.messages),
                    )
                    / new_turns
                )
                growth = previous.growth * 0.5 + delta * 0.5
        snapshot = Snapshot(
            history.head,
            len(history),
            previous.epoch if previous else uuid.uuid4().hex,
            start,
            reminders,
            wire,
            raw_tokens,
            growth,
            previous.epoch_turns + 1 if previous else 1,
            intervals(omitted - selected),
            search.warnings,
            search.required,
        )
        return snapshot, intervals(omitted - selected), search.warnings

    async def prepare(
        self,
        messages: list[Message],
        *,
        chat_id: str,
        user_id: str,
        model: str = "default",
        environment: dict | None = None,
        extra_tokens: int = 0,
        correlation: str = "",
    ) -> ContextResult:
        if not chat_id or not user_id:
            raise ValueError("Stable chat_id and user_id are required")
        if extra_tokens < 0:
            raise ValueError("extra_tokens must be nonnegative")
        history = History.from_messages(messages)
        scope = digest([user_id, chat_id])
        domain = digest(
            [
                self.namespace,
                model,
                environment or {},
                asdict(self.config),
                asdict(self.pricing) if self.pricing else None,
                getattr(self.counter, "cache_identity", type(self.counter).__qualname__),
                extra_tokens,
            ]
        )
        limit = self.config.input_limit - extra_tokens
        if limit <= 0:
            raise ContextOverflow("Request overhead consumes the input budget")
        previous = self.store.ancestor(scope, domain, history.heads)
        ttl = self.pricing.ttl_seconds if self.pricing else 300
        recent = self.store.recent_wires(scope, domain, ttl)

        def shared(snapshot: Snapshot) -> int:
            return max(
                (common_prefix_tokens(snapshot.messages, wire, self.counter) for wire in recent),
                default=0,
            )

        raw_tokens = self.counter.count(history.messages)
        omitted: tuple[Span, ...] = ()
        warnings: tuple[str, ...] = ()
        economics: dict[str, float] = {}
        if previous and previous.head == history.head:
            snapshot, reason = previous, "replay"
        else:
            reason = "initial" if previous is None else "append"
            try:
                snapshot, omitted, warnings = await self._build(
                    history, previous, raw_tokens, limit
                )
            except ContextOverflow:
                if previous is None:
                    raise
                snapshot, omitted, warnings = await self._build(history, None, raw_tokens, limit)
                reason = "capacity"
            coverage_required = {i for s in snapshot.required for i in s.indices()}
            coverage_present = set(range(snapshot.tail_start, len(history))) | {
                i for r in snapshot.reminders for s in r.spans for i in s.indices()
            }
            candidate = None
            if previous and reason == "append" and coverage_required - coverage_present:
                candidate, candidate_omitted, candidate_warnings = await self._build(
                    history, None, raw_tokens, limit, compact=True
                )
                candidate_coverage = set(range(candidate.tail_start, len(history))) | {
                    i for r in candidate.reminders for s in r.spans for i in s.indices()
                }
                if coverage_required <= candidate_coverage:
                    snapshot, omitted, warnings = candidate, candidate_omitted, candidate_warnings
                    snapshot = replace(snapshot, growth=previous.growth)
                    reason = "recall_capacity"
            if previous and reason == "append" and self.pricing:
                candidate, candidate_omitted, candidate_warnings = await self._build(
                    history,
                    None,
                    raw_tokens,
                    limit,
                    compact=True,
                )
                # Economics must not achieve savings by discarding today's recalled passages.
                required = {
                    i
                    for r in snapshot.reminders
                    if r.anchor >= previous.history_length
                    for s in r.spans
                    for i in s.indices()
                }
                available = set(range(candidate.tail_start, len(history))) | {
                    i for r in candidate.reminders for s in r.spans for i in s.indices()
                }
                if required <= available:
                    decision = RenewalPolicy(self.pricing, self.config.economic_margin).decide(
                        old_tokens=self.counter.count(snapshot.messages) + extra_tokens,
                        new_tokens=self.counter.count(candidate.messages) + extra_tokens,
                        old_shared=shared(snapshot),
                        new_shared=shared(candidate),
                        growth=snapshot.growth,
                        capacity=self.config.input_limit,
                    )
                    economics = {
                        "cycle_rounds": decision.cycle_rounds,
                        "average_cost": decision.average_cost,
                        "keep_excess": decision.keep_excess,
                        "rebuild_excess": decision.rebuild_excess,
                    }
                    if decision.rebuild:
                        snapshot, omitted, warnings = (
                            candidate,
                            candidate_omitted,
                            candidate_warnings,
                        )
                        snapshot = replace(snapshot, growth=previous.growth)
                        reason = "economic"
        snapshot = self.store.save(scope, domain, snapshot)
        omitted, warnings = snapshot.omitted, snapshot.warnings
        potential = shared(snapshot)
        request_id = self.store.attempt(scope, domain, snapshot, correlation)
        recalled = tuple(s for r in snapshot.reminders for s in r.spans)
        return ContextResult(
            snapshot.messages,
            request_id,
            snapshot,
            reason,
            self.counter.count(snapshot.messages) + extra_tokens,
            potential,
            recalled,
            omitted,
            warnings,
            economics,
        )

    def record_usage(self, request_id: str, usage: dict) -> None:
        previous = self.store.usage(request_id)
        raw = dict(previous.get("raw", {})) if previous else {}
        raw.update(usage)
        self.store.record_usage(request_id, parse_usage(raw))
