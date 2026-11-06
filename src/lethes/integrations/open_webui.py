"""Open WebUI Filter contract checked against official docs on 2026-11-06.

Assembly runs in request(), after pipeline enrichment, on every provider hop.
No per-chat mutable telemetry lives on the shared Filter instance.
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import asdict

from pydantic import BaseModel, Field, SecretStr

from lethes.costs import CachePricing
from lethes.engine import ContextConfig, ContextManager
from lethes.history import canonical, digest, normalize
from lethes.providers import OpenAIEmbeddings, OpenAIQueryExpander
from lethes.retrieval import HybridRetriever
from lethes.store import SQLiteStore
from lethes.tokens import ByteTokenCounter, TiktokenCounter

log = logging.getLogger(__name__)
Emitter = Callable[[dict], Awaitable[None]]


def is_prefix(prefix: list[dict], messages: list[dict]) -> bool:
    return len(prefix) <= len(messages) and all(
        canonical(normalize(a)) == canonical(normalize(b))
        for a, b in zip(prefix, messages, strict=False)
    )


def instruction_count(messages: list[dict]) -> int:
    count = 0
    for message in messages:
        if message.get("role") not in {"system", "developer"}:
            break
        count += 1
    return count


class OpenWebUIFilter:
    class Valves(BaseModel):
        priority: int = Field(default=100, description="Run after other context-changing filters")
        state_path: str = Field(default="/app/backend/data/lethes/state.sqlite3")
        context_window: int = Field(default=32768, gt=0)
        reply_reserve: int = Field(default=4096, ge=0)
        safety_reserve: int = Field(default=1024, ge=0)
        recent_tokens: int = Field(default=8192, gt=0)
        minimum_recent_turns: int = Field(default=2, gt=0)
        rebuild_fill: float = Field(default=0.75, gt=0, lt=1)
        tokenizer: str = Field(default="bytes", description="bytes or a tiktoken encoding name")
        input_price: float | None = Field(default=None, ge=0, description="USD per million tokens")
        cached_input_price: float | None = Field(default=None, ge=0)
        cache_minimum: int = Field(default=1024, ge=0)
        cache_block: int = Field(default=128, gt=0)
        cache_ttl_seconds: float = Field(default=300, gt=0)
        embedding_api_base: str = "https://api.openai.com/v1"
        embedding_api_key: SecretStr = SecretStr("")
        embedding_model: str = ""
        query_api_base: str = "https://api.openai.com/v1"
        query_api_key: SecretStr = SecretStr("")
        query_model: str = ""

    def __init__(self) -> None:
        self.valves = self.Valves()
        self.toggle = True
        self._manager: ContextManager | None = None
        self._config_key = ""

    def _get_manager(self) -> ContextManager:
        config = self.valves.model_dump(mode="json")
        # SecretStr serialization masks values; hash actual secrets for invalidation, never log.
        config["embedding_api_key"] = digest(self.valves.embedding_api_key.get_secret_value())
        config["query_api_key"] = digest(self.valves.query_api_key.get_secret_value())
        key = digest(config)
        if self._manager is not None and key == self._config_key:
            return self._manager
        valves = self.valves
        pricing = None
        if valves.input_price is not None and valves.cached_input_price is not None:
            pricing = CachePricing(
                valves.input_price,
                valves.cached_input_price,
                valves.cache_minimum,
                valves.cache_block,
                valves.cache_ttl_seconds,
            )
        embedder = (
            OpenAIEmbeddings(
                valves.embedding_api_base,
                valves.embedding_api_key.get_secret_value(),
                valves.embedding_model,
            )
            if valves.embedding_model
            else None
        )
        expander = (
            OpenAIQueryExpander(
                valves.query_api_base, valves.query_api_key.get_secret_value(), valves.query_model
            )
            if valves.query_model
            else None
        )
        manager = ContextManager(
            ContextConfig(
                context_window=valves.context_window,
                reply_reserve=valves.reply_reserve,
                safety_reserve=valves.safety_reserve,
                recent_tokens=valves.recent_tokens,
                minimum_recent_turns=valves.minimum_recent_turns,
                rebuild_fill=valves.rebuild_fill,
            ),
            store=SQLiteStore(valves.state_path),
            retriever=HybridRetriever(embedder=embedder, expander=expander),
            pricing=pricing,
            counter=ByteTokenCounter()
            if valves.tokenizer == "bytes"
            else TiktokenCounter(valves.tokenizer),
            namespace=key,
        )
        # Do not close the previous manager while an in-flight request may still use it.
        self._manager, self._config_key = manager, key
        return manager

    @staticmethod
    def _metadata(body: dict, metadata: dict | None) -> dict:
        if metadata is not None:
            return metadata
        if not isinstance(body.get("metadata"), dict):
            body["metadata"] = {}
        return body["metadata"]

    @staticmethod
    def _key(function_id: str | None) -> str:
        return f"_lethes_v2_{function_id or 'default'}"

    @staticmethod
    def _correlation(metadata: dict, user: dict | None, body: dict) -> str:
        chat = metadata.get("chat_id") or body.get("chat_id")
        message = metadata.get("message_id") or body.get("id")
        if not chat or not message:
            return ""
        return digest(
            [(user or {}).get("id") or metadata.get("user_id"), chat, message, body.get("model")]
        )

    async def inlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __task__: str | None = None,
        __id__: str | None = None,
    ) -> dict:
        metadata = self._metadata(body, __metadata__)
        # Internal title/query/tag requests may carry the same chat_id as real dialogue.
        metadata[self._key(__id__)] = {"skip": bool(__task__ or metadata.get("task"))}
        return body

    async def request(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __user__: dict | None = None,
        __model__: dict | None = None,
        __event_emitter__: Emitter | None = None,
        __chat_id__: str | None = None,
        __message_id__: str | None = None,
        __task__: str | None = None,
        __id__: str | None = None,
    ) -> dict:
        metadata = self._metadata(body, __metadata__)
        key = self._key(__id__)
        context = metadata.setdefault(key, {})
        if __task__ or metadata.get("task") or context.get("skip"):
            return body
        chat_id = __chat_id__ or metadata.get("chat_id") or body.get("chat_id")
        user_id = (__user__ or {}).get("id") or metadata.get("user_id")
        if not chat_id or not user_id or not body.get("messages"):
            # No invented session identity: unscoped API calls remain unchanged.
            return body
        manager = self._get_manager()
        incoming = copy.deepcopy(body["messages"])
        prior_wire = context.get("wire", [])
        prior_source = context.get("source", [])
        if prior_wire and is_prefix(prior_wire, incoming):
            incoming = prior_source + incoming[len(prior_wire) :]
        elif prior_source and not is_prefix(prior_source, incoming):
            # Open WebUI merges instruction messages between provider hops. Restore
            # omitted originals even when that leading instruction shape has changed.
            head = instruction_count(incoming)
            old_head = instruction_count(prior_wire)
            if is_prefix(prior_wire[old_head:], incoming[head:]):
                incoming = (
                    incoming[:head]
                    + prior_source[instruction_count(prior_source) :]
                    + incoming[head + len(prior_wire) - old_head :]
                )
            elif any(canonical(normalize(m)) in set(context.get("overlays", [])) for m in incoming):
                raise ValueError(
                    "Host rewrote compiled history; cannot safely restore original path"
                )
        model_info = (__model__ or {}).get("info") or {}
        environment = {
            "model_info": model_info,
            "provider": (__model__ or {}).get("owned_by"),
            "tools": body.get("tools"),
            "tool_choice": body.get("tool_choice"),
            "response_format": body.get("response_format"),
        }
        overhead = len(canonical(body.get("tools") or []).encode())
        if body.get("response_format"):
            overhead += len(canonical(body["response_format"]).encode())
        workspace_system = (model_info.get("params") or {}).get("system")
        if workspace_system:
            overhead += manager.counter.count([{"role": "system", "content": workspace_system}])
        reply = body.get("max_completion_tokens", body.get("max_tokens", 0)) or 0
        if not isinstance(reply, int) or reply < 0:
            raise ValueError("Invalid reply token limit")
        overhead += max(0, reply - manager.config.reply_reserve)
        if __message_id__ and not metadata.get("message_id"):
            metadata["message_id"] = __message_id__
        correlation = self._correlation({**metadata, "chat_id": chat_id}, __user__, body)
        result = await manager.prepare(
            incoming,
            chat_id=str(chat_id),
            user_id=str(user_id),
            model=str(body.get("model") or (__model__ or {}).get("id") or "default"),
            environment=environment,
            extra_tokens=overhead,
            correlation=correlation,
        )
        context.update(
            {
                "source": incoming,
                "wire": copy.deepcopy(result.messages),
                "overlays": [r.payload for r in result.snapshot.reminders],
                "request_id": result.request_id,
                "state_path": self.valves.state_path,
            }
        )
        body["messages"] = result.messages
        log.info(
            "lethes.context",
            extra={
                "lethes": {
                    "request_id": result.request_id,
                    "reason": result.reason,
                    "estimated_tokens": result.estimated_tokens,
                    "potential_cached_tokens": result.potential_cached_tokens,
                    "omitted": [asdict(s) for s in result.omitted],
                    "warnings": result.warnings,
                    "economics": result.economics,
                }
            },
        )
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Historical context ready",
                        "done": True,
                        "action": "lethes.context",
                    },
                }
            )
        return body

    async def stream(
        self, event: dict, __metadata__: dict | None = None, __id__: str | None = None
    ) -> dict:
        context = (__metadata__ or {}).get(self._key(__id__), {})
        data = event.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data.removeprefix("data:").strip())
            except ValueError:
                data = {}
        if not isinstance(data, dict):
            data = {}
        message = event.get("message") or data.get("message") or {}
        usage = event.get("usage") or data.get("usage")
        if not usage and isinstance(message, dict):
            usage = message.get("usage")
        if isinstance(usage, dict) and context.get("request_id"):
            self._record_usage(context, context["request_id"], usage)
        return event

    def _record_usage(self, context: dict, request_id: str, usage: dict) -> None:
        path = context.get("state_path", self.valves.state_path)
        if path == self.valves.state_path:
            self._get_manager().record_usage(request_id, usage)
        else:
            # A valve change must not redirect an in-flight response to a different database.
            store = SQLiteStore(path)
            try:
                ContextManager(store=store).record_usage(request_id, usage)
            finally:
                store.close()

    async def outlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __user__: dict | None = None,
        __id__: str | None = None,
    ) -> dict:
        metadata = __metadata__ or {}
        context = metadata.get(self._key(__id__), {})
        usage = body.get("usage")
        if not usage and body.get("messages"):
            usage = body["messages"][-1].get("usage")
        if isinstance(usage, dict):
            manager = self._get_manager()
            request_id = context.get("request_id") or manager.store.correlated_request(
                self._correlation(metadata, __user__, body)
            )
            if request_id:
                self._record_usage(context, request_id, usage)
        return body
