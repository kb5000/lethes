"""Canonical, content-versioned paths. Host message IDs are not revision IDs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

Message = dict[str, Any]
HOST_FIELDS = frozenset(
    {
        "id",
        "parentId",
        "childrenIds",
        "timestamp",
        "created_at",
        "updated_at",
        "usage",
        "done",
        "statusHistory",
        "model",
        "modelName",
        "modelIdx",
        "models",
        "user_id",
        "user",
        "userContext",
        "lastSentence",
    }
)


def canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalize(message: Message) -> Message:
    if message.get("role") not in {"system", "developer", "user", "assistant", "tool", "function"}:
        raise ValueError("Unsupported or missing message role")
    # Preserve provider extensions (cache_control, reasoning_details, etc.).
    result = {k: v for k, v in message.items() if k not in HOST_FIELDS}
    result.setdefault("content", None)
    return json.loads(canonical(result))


def text(message: Message) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


@dataclass(frozen=True)
class History:
    payloads: tuple[str, ...]
    heads: tuple[str, ...]
    instruction_count: int

    @classmethod
    def from_messages(cls, messages: list[Message]) -> History:
        if not messages:
            raise ValueError("History must not be empty")
        payloads = tuple(canonical(normalize(m)) for m in messages)
        heads: list[str] = []
        parent = "lethes-history-v2"
        instructions = 0
        for i, payload in enumerate(payloads):
            parent = digest([parent, payload])
            heads.append(parent)
            if i == instructions and json.loads(payload)["role"] in {"system", "developer"}:
                instructions += 1
        return cls(payloads, tuple(heads), instructions)

    @property
    def messages(self) -> list[Message]:
        return [json.loads(p) for p in self.payloads]

    @property
    def head(self) -> str:
        return self.heads[-1]

    def __len__(self) -> int:
        return len(self.payloads)

    def turn_starts(self) -> list[int]:
        starts = [
            i
            for i, p in enumerate(self.payloads)
            if i >= self.instruction_count and json.loads(p)["role"] == "user"
        ]
        if self.instruction_count < len(self) and self.instruction_count not in starts:
            starts.insert(0, self.instruction_count)
        return starts
