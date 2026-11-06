"""Read actual Open WebUI export paths without flattening sibling branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .history import Message, normalize


@dataclass(frozen=True)
class ExportPath:
    chat_id: str
    leaf_id: str
    active: bool
    messages: list[Message]


def open_webui_paths(data: Any, *, all_branches: bool = True) -> list[ExportPath]:
    chats = data if isinstance(data, list) else [data]
    result = []
    for number, entry in enumerate(chats):
        if not isinstance(entry, dict):
            raise ValueError("Export chat must be an object")
        chat = entry.get("chat", entry)
        history = chat.get("history") or {}
        nodes = history.get("messages") or {}
        chat_id = str(entry.get("id") or chat.get("id") or f"import-{number}")
        system = (chat.get("params") or {}).get("system")
        leading = [{"role": "system", "content": system}] if system else []
        if not nodes:
            messages = [normalize(m) for m in chat.get("messages", [])]
            if not messages:
                raise ValueError("Export contains no message history")
            result.append(ExportPath(chat_id, "flat", True, leading + messages))
            continue
        current = history.get("currentId") or entry.get("current_message_id")
        parents = {node.get("parentId") for node in nodes.values()}
        leaves = [key for key in nodes if key not in parents]
        if not current:
            if len(leaves) != 1:
                raise ValueError("Ambiguous export: currentId is missing")
            current = leaves[0]
        if current not in nodes:
            raise ValueError("Export currentId references a missing node")
        targets = [key for key in leaves if key != current] if all_branches else []
        targets.append(current)  # finish on the user's selected path
        for target in targets:
            reverse = []
            visited = set()
            cursor = target
            while cursor is not None:
                if cursor in visited:
                    raise ValueError("Export history contains a parent cycle")
                if cursor not in nodes:
                    raise ValueError("Export parent references a missing node")
                visited.add(cursor)
                node = nodes[cursor]
                reverse.append(normalize(node))
                cursor = node.get("parentId")
            result.append(
                ExportPath(chat_id, target, target == current, leading + list(reversed(reverse)))
            )
    return result
