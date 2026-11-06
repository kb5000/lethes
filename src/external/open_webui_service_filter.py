"""
title: lethes — Independent context service
version: 2.0.0a1
requirements: httpx>=0.27,pydantic>=2
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging

import httpx
from pydantic import BaseModel, Field, SecretStr

log = logging.getLogger("lethes.remote_filter")
HOST_FIELDS = {
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


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalized(message):
    result = {k: v for k, v in message.items() if k not in HOST_FIELDS}
    result.setdefault("content", None)
    return result


def is_prefix(prefix, messages):
    return len(prefix) <= len(messages) and all(
        canonical(normalized(a)) == canonical(normalized(b))
        for a, b in zip(prefix, messages, strict=False)
    )


def instruction_count(messages):
    count = 0
    for message in messages:
        if message.get("role") not in {"system", "developer"}:
            break
        count += 1
    return count


class Filter:
    class Valves(BaseModel):
        priority: int = 100
        service_url: str = "http://lethes:8000"
        service_api_key: SecretStr = SecretStr("")
        timeout_seconds: float = Field(default=60, gt=0)
        show_status: bool = True

    def __init__(self):
        self.valves = self.Valves()
        self.toggle = True

    @staticmethod
    def _key(function_id):
        return f"_lethes_service_v2_{function_id or 'default'}"

    @staticmethod
    def _metadata(body, metadata):
        if metadata is not None:
            return metadata
        if not isinstance(body.get("metadata"), dict):
            body["metadata"] = {}
        return body["metadata"]

    @staticmethod
    def _correlation(metadata, user, body):
        chat = metadata.get("chat_id") or body.get("chat_id")
        message = metadata.get("message_id") or body.get("id")
        if not chat or not message:
            return ""
        return digest(
            [(user or {}).get("id") or metadata.get("user_id"), chat, message, body.get("model")]
        )

    async def _post(self, path, payload):
        async with httpx.AsyncClient(
            base_url=self.valves.service_url.rstrip("/"),
            timeout=self.valves.timeout_seconds,
            headers={"Authorization": "Bearer " + self.valves.service_api_key.get_secret_value()},
        ) as client:
            response = await client.post(path, json=payload)
            response.raise_for_status()
            return response.json()

    async def _status(self, line, emitter):
        if self.valves.show_status and emitter:
            try:
                await emitter(
                    {"type": "status", "data": {"description": line, "done": True, "hidden": False}}
                )
            except Exception as exc:
                log.warning("lethes.status_failed error=%s", type(exc).__name__)

    async def inlet(self, body, __metadata__=None, __task__=None, __id__=None):
        metadata = self._metadata(body, __metadata__)
        metadata[self._key(__id__)] = {"skip": bool(__task__ or metadata.get("task"))}
        return body

    async def request(
        self,
        body,
        __metadata__=None,
        __user__=None,
        __model__=None,
        __event_emitter__=None,
        __chat_id__=None,
        __message_id__=None,
        __task__=None,
        __id__=None,
    ):
        metadata = self._metadata(body, __metadata__)
        context = metadata.setdefault(self._key(__id__), {})
        if __task__ or metadata.get("task") or context.get("skip"):
            return body
        chat = __chat_id__ or metadata.get("chat_id") or body.get("chat_id")
        user = (__user__ or {}).get("id") or metadata.get("user_id")
        if not user or not chat or not body.get("messages"):
            return body
        incoming = copy.deepcopy(body["messages"])
        wire, source = context.get("wire", []), context.get("source", [])
        if wire and is_prefix(wire, incoming):
            incoming = source + incoming[len(wire) :]
        elif source and not is_prefix(source, incoming):
            head, old_head = instruction_count(incoming), instruction_count(wire)
            if is_prefix(wire[old_head:], incoming[head:]):
                incoming = (
                    incoming[:head]
                    + source[instruction_count(source) :]
                    + incoming[head + len(wire) - old_head :]
                )
            elif any(canonical(normalized(m)) in context.get("overlays", []) for m in incoming):
                raise ValueError(
                    "Host rewrote compiled history; cannot safely restore original path"
                )
        if __message_id__ and not metadata.get("message_id"):
            metadata["message_id"] = __message_id__
        model_info = (__model__ or {}).get("info") or {}
        system = (model_info.get("params") or {}).get("system")
        token_context = {
            k: v
            for k, v in {
                "tools": body.get("tools"),
                "tool_choice": body.get("tool_choice"),
                "response_format": body.get("response_format"),
                "prefix_messages": [{"role": "system", "content": system}] if system else [],
            }.items()
            if v
        }
        correlation = self._correlation({**metadata, "chat_id": chat}, __user__, body)
        try:
            result = await self._post(
                "/v2/context/prepare",
                {
                    "user_id": str(user),
                    "chat_id": str(chat),
                    "model": str(body.get("model") or (__model__ or {}).get("id") or "default"),
                    "messages": incoming,
                    "request_context": token_context,
                    "correlation": correlation,
                    "reply_tokens": body.get("max_completion_tokens", body.get("max_tokens", 0))
                    or 0,
                    "environment": {
                        "model_info": model_info,
                        "provider": (__model__ or {}).get("owned_by"),
                        "tools": body.get("tools"),
                        "tool_choice": body.get("tool_choice"),
                        "response_format": body.get("response_format"),
                    },
                },
            )
        except (httpx.HTTPError, ValueError) as exc:
            log.error("lethes.prepare_failed error=%s", type(exc).__name__)
            raise RuntimeError(
                "lethes service could not prepare context; check service logs"
            ) from None
        source_payloads = {canonical(normalized(s)) for s in incoming}
        context.update(
            {
                "request_id": result["request_id"],
                "wire": copy.deepcopy(result["messages"]),
                "source": incoming,
                "correlation": correlation,
                "provider_hops": context.get("provider_hops", 0) + 1,
                "response_model": None,
                "usage_signature": None,
                "overlays": [
                    canonical(normalized(m))
                    for m in result["messages"]
                    if canonical(normalized(m)) not in source_payloads
                ],
            }
        )
        body["messages"] = result["messages"]
        await self._status(result["status"], __event_emitter__)
        return body

    async def _usage(self, context, usage, model, origin, emitter, correlation=""):
        try:
            result = await self._post(
                "/v2/usage",
                {
                    "request_id": context.get("request_id"),
                    "correlation": context.get("correlation") or correlation,
                    "usage": usage,
                    "model": model,
                    "origin": origin,
                    "aggregate": origin == "outlet" and context.get("provider_hops", 1) > 1,
                },
            )
            signature = digest([result["observation"]["request_id"], result["status"]])
            if result["authoritative"] and signature != context.get("usage_signature"):
                await self._status(result["status"], emitter)
                context["usage_signature"] = signature
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.warning("lethes.usage_failed error=%s", type(exc).__name__)

    async def stream(self, event, __metadata__=None, __id__=None, __event_emitter__=None):
        context = (__metadata__ or {}).get(self._key(__id__), {})
        data = event.get("data")
        if isinstance(data, str):
            try:
                data = json.loads(data.removeprefix("data:").strip())
            except ValueError:
                data = {}
        if not isinstance(data, dict):
            data = {}
        reported = event.get("model") or data.get("model")
        if isinstance(reported, str):
            context["response_model"] = reported
        message = event.get("message") or data.get("message") or {}
        usage = (
            event.get("usage")
            or data.get("usage")
            or (message.get("usage") if isinstance(message, dict) else None)
        )
        if isinstance(usage, dict) and context.get("request_id"):
            await self._usage(
                context, usage, context.get("response_model"), "provider_stream", __event_emitter__
            )
        return event

    async def outlet(
        self, body, __metadata__=None, __user__=None, __id__=None, __event_emitter__=None
    ):
        metadata = __metadata__ or {}
        context = metadata.get(self._key(__id__), {})
        usage = body.get("usage") or (
            body["messages"][-1].get("usage") if body.get("messages") else None
        )
        correlation = self._correlation(metadata, __user__, body)
        if isinstance(usage, dict) and (context.get("request_id") or correlation):
            await self._usage(
                context, usage, body.get("model"), "outlet", __event_emitter__, correlation
            )
        return body
