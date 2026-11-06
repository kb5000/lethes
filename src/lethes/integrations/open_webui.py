"""Open WebUI Filter contract checked against official docs on 2026-11-06.

Assembly runs in request(), after pipeline enrichment, on every provider hop.
No per-chat mutable telemetry lives on the shared Filter instance.
"""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, Field, SecretStr

from lethes.costs import CachePricing
from lethes.engine import ContextConfig, ContextManager
from lethes.history import canonical, digest, normalize
from lethes.observability import context_observation, status_line
from lethes.providers import OpenAIEmbeddings, OpenAIQueryExpander
from lethes.retrieval import HybridRetriever
from lethes.store import SQLiteStore
from lethes.tokens import ByteTokenCounter, LiteLLMTokenCounter, TiktokenCounter

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
        tokenizer: str = Field(
            default="bytes", description="bytes, litellm, or a tiktoken encoding"
        )
        tokenizer_model: str = Field(
            default="", description="Actual LiteLLM model; defaults to workspace base_model_id"
        )
        show_status: bool = Field(default=True, description="Compact token status in chat")
        input_price: float | None = Field(default=None, ge=0, description="USD per million tokens")
        cached_input_price: float | None = Field(default=None, ge=0)
        cache_minimum: int = Field(default=1024, ge=0)
        cache_block: int = Field(default=128, gt=0)
        cache_ttl_seconds: float = Field(default=300, gt=0)
        embedding_api_base: str = "https://api.openai.com/v1"
        embedding_api_key: SecretStr = SecretStr("")
        embedding_model: str = ""
        embedding_chunk_chars: int = Field(default=0, ge=0, description="0 disables windowing")
        embedding_chunk_overlap: int = Field(default=64, ge=0)
        embedding_query_prefix: str = ""
        embedding_document_prefix: str = ""
        semantic_floor: float = Field(default=0.25, ge=-1, le=1)
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
        config.pop("show_status")  # Presentation changes must not invalidate a model prefix.
        for name, default in {
            "embedding_chunk_chars": 0,
            "embedding_chunk_overlap": 64,
            "embedding_query_prefix": "",
            "embedding_document_prefix": "",
            "semantic_floor": 0.25,
        }.items():
            if config[name] == default:
                config.pop(name)  # Preserve existing domains with default helper behavior.
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
            retriever=HybridRetriever(
                embedder=embedder,
                expander=expander,
                semantic_floor=valves.semantic_floor,
                embedding_chunk_chars=valves.embedding_chunk_chars,
                embedding_chunk_overlap=valves.embedding_chunk_overlap,
                embedding_query_prefix=valves.embedding_query_prefix,
                embedding_document_prefix=valves.embedding_document_prefix,
            ),
            pricing=pricing,
            counter=ByteTokenCounter()
            if valves.tokenizer in {"bytes", "litellm"}
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
        started = time.perf_counter()
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
        workspace_system = (model_info.get("params") or {}).get("system")
        token_context = {
            k: v
            for k, v in {
                "tools": body.get("tools"),
                "tool_choice": body.get("tool_choice"),
                "response_format": body.get("response_format"),
                "prefix_messages": [{"role": "system", "content": workspace_system}]
                if workspace_system
                else [],
            }.items()
            if v
        }
        if self.valves.tokenizer == "litellm":
            # Bind a request-local model/envelope; never mutate a shared counter
            # while another chat/model is awaiting retrieval or a response.
            model = (
                self.valves.tokenizer_model
                or model_info.get("base_model_id")
                or body.get("model")
                or (__model__ or {}).get("id")
            )
            manager = copy.copy(manager)
            manager.counter = LiteLLMTokenCounter(
                str(model or ""),
                tools=body.get("tools"),
                tool_choice=body.get("tool_choice"),
                response_format=body.get("response_format"),
                prefix_messages=[{"role": "system", "content": workspace_system}]
                if workspace_system
                else [],
            )
            overhead = 0
        else:
            overhead = len(canonical(body.get("tools") or []).encode())
            if body.get("response_format"):
                overhead += len(canonical(body["response_format"]).encode())
            if workspace_system:
                overhead += manager.counter.count([{"role": "system", "content": workspace_system}])
        reply = body.get("max_completion_tokens", body.get("max_tokens", 0)) or 0
        if not isinstance(reply, int) or reply < 0:
            raise ValueError("Invalid reply token limit")
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
            extra_reply_tokens=max(0, reply - manager.config.reply_reserve),
            request_context=token_context,
            correlation=correlation,
        )
        observation = context_observation(
            result,
            body.get("model"),
            manager.config.input_limit - max(0, reply - manager.config.reply_reserve),
            (time.perf_counter() - started) * 1000,
        )
        manager.store.record_observation(result.request_id, observation)
        context.update(
            {
                "source": incoming,
                "wire": copy.deepcopy(result.messages),
                "overlays": [r.payload for r in result.snapshot.reminders],
                "request_id": result.request_id,
                "state_path": self.valves.state_path,
                "provider_hops": context.get("provider_hops", 0) + 1,
                "response_model": None,
                "status_usage": None,
            }
        )
        body["messages"] = result.messages
        log.info("lethes.context %s", canonical(observation), extra={"lethes": observation})
        await self._emit_status(observation, __event_emitter__)
        return body

    async def _emit_status(self, observation: dict, emitter: Emitter | None) -> None:
        if not emitter or not self.valves.show_status:
            return
        try:
            await emitter(
                {
                    "type": "status",
                    "data": {
                        "description": status_line(observation),
                        "done": True,
                        "hidden": False,
                        "action": "lethes.context",
                    },
                }
            )
        except Exception as exc:
            log.warning(
                "lethes.status_failed request_id=%s error=%s",
                observation["request_id"],
                type(exc).__name__,
            )

    async def _usage_status(
        self, context: dict, recorded: tuple[dict, dict | None] | None, emitter: Emitter | None
    ) -> None:
        if not recorded:
            return
        usage, observation = recorded
        details = {
            "request_id": observation["request_id"] if observation else context.get("request_id"),
            "usage_origin": usage.get("usage_origin"),
            "aggregate": usage.get("aggregate", False),
            "measurement_eligible": SQLiteStore.authoritative_usage(usage),
            "reported_model": usage.get("reported_model"),
        }
        for name in ("input_tokens", "cached_tokens", "output_tokens"):
            value = usage.get(name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                details[name] = value
        log.info("lethes.usage %s", canonical(details), extra={"lethes": details})
        if not observation or not SQLiteStore.authoritative_usage(usage):
            return
        signature = digest(
            [observation["request_id"], usage["input_tokens"], details.get("cached_tokens")]
        )
        if context.get("status_usage") == signature:
            return
        await self._emit_status(
            {
                **observation,
                "input_tokens": usage["input_tokens"],
                "count_source": "provider_usage",
                "cached_tokens": details.get("cached_tokens"),
            },
            emitter,
        )
        context["status_usage"] = signature

    async def stream(
        self,
        event: dict,
        __metadata__: dict | None = None,
        __id__: str | None = None,
        __event_emitter__: Emitter | None = None,
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
        reported = event.get("model") or data.get("model")
        if isinstance(reported, str):
            # Some streams name the model in an earlier chunk than their usage.
            context["response_model"] = reported
        usage = event.get("usage") or data.get("usage")
        if not usage and isinstance(message, dict):
            usage = message.get("usage")
        if isinstance(usage, dict) and context.get("request_id"):
            recorded = self._record_usage(
                context,
                context["request_id"],
                usage,
                context.get("response_model"),
            )
            await self._usage_status(context, recorded, __event_emitter__)
        return event

    def _record_usage(
        self,
        context: dict,
        request_id: str,
        usage: dict,
        model: str | None = None,
        origin: str = "provider_stream",
    ) -> tuple[dict, dict | None] | None:
        def record(manager: ContextManager) -> tuple[dict, dict | None] | None:
            previous = manager.store.usage(request_id)
            manager.record_usage(
                request_id,
                usage,
                model=model,
                origin=origin,
                aggregate=origin == "outlet"
                and (
                    context.get("provider_hops", 1) > 1
                    or manager.store.turn_attempts(request_id) > 1
                ),
            )
            saved = manager.store.usage(request_id)
            if not saved or saved == previous:
                return None
            return saved, manager.store.observation(request_id)

        path = context.get("state_path", self.valves.state_path)
        if path == self.valves.state_path:
            return record(self._get_manager())
        else:
            # A valve change must not redirect an in-flight response to a different database.
            store = SQLiteStore(path)
            try:
                return record(ContextManager(store=store))
            finally:
                store.close()

    async def outlet(
        self,
        body: dict,
        __metadata__: dict | None = None,
        __user__: dict | None = None,
        __id__: str | None = None,
        __event_emitter__: Emitter | None = None,
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
                reported = body.get("model")
                recorded = self._record_usage(
                    context,
                    request_id,
                    usage,
                    reported if isinstance(reported, str) else None,
                    origin="outlet",
                )
                await self._usage_status(context, recorded, __event_emitter__)
        return body
