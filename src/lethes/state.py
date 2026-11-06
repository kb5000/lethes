"""Immutable snapshots describe assembly, not inferred fictional facts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .history import Message, canonical


@dataclass(frozen=True, order=True)
class Span:
    start: int
    end: int

    def indices(self) -> range:
        return range(self.start, self.end)


@dataclass(frozen=True)
class Reminder:
    anchor: int
    spans: tuple[Span, ...]
    payload: str


@dataclass(frozen=True)
class Snapshot:
    head: str
    history_length: int
    epoch: str
    tail_start: int
    reminders: tuple[Reminder, ...]
    wire: tuple[str, ...]
    raw_tokens: int
    growth: float
    epoch_turns: int = 1
    omitted: tuple[Span, ...] = ()
    warnings: tuple[str, ...] = ()
    required: tuple[Span, ...] = ()

    @property
    def messages(self) -> list[Message]:
        import json

        return [json.loads(p) for p in self.wire]

    def serialize(self) -> str:
        return canonical(asdict(self))

    @classmethod
    def deserialize(cls, payload: str) -> Snapshot:
        import json

        d = json.loads(payload)
        d["wire"] = tuple(d["wire"])
        d["omitted"] = tuple(Span(**s) for s in d.get("omitted", ()))
        d["warnings"] = tuple(d.get("warnings", ()))
        d["required"] = tuple(Span(**s) for s in d.get("required", ()))
        d["reminders"] = tuple(
            Reminder(r["anchor"], tuple(Span(**s) for s in r["spans"]), r["payload"])
            for r in d["reminders"]
        )
        return cls(**d)


@dataclass(frozen=True)
class ContextResult:
    messages: list[Message]
    request_id: str
    snapshot: Snapshot
    reason: str
    estimated_tokens: int
    potential_cached_tokens: int
    recalled: tuple[Span, ...] = ()
    omitted: tuple[Span, ...] = ()
    warnings: tuple[str, ...] = ()
    economics: dict[str, float] = field(default_factory=dict)
