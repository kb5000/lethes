"""Independent context service and same-origin observer. Main-model calls stay at the host."""

from __future__ import annotations

import copy
import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from . import __version__
from .costs import CachePricing
from .engine import ContextConfig, ContextManager, ContextOverflow
from .history import canonical, digest
from .observability import context_observation, status_line
from .providers import OpenAIEmbeddings, OpenAIQueryExpander
from .retrieval import HybridRetriever
from .store import SQLiteStore
from .tokens import ByteTokenCounter, LiteLLMTokenCounter, TiktokenCounter

log = logging.getLogger("lethes.service")
BEARER = HTTPBearer(auto_error=False)


class Profile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    context: dict[str, Any] = Field(default_factory=dict)
    tokenizer: str = "bytes"
    tokenizer_model: str = ""
    pricing: dict[str, Any] | None = None
    retrieval: dict[str, Any] = Field(default_factory=dict)
    namespace: str = "service-v2"


class ServiceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    defaults: Profile = Field(default_factory=Profile)
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)


class RequestContext(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    prefix_messages: list[dict[str, Any]] = Field(default_factory=list)


class PrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1)
    chat_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    messages: list[dict[str, Any]] = Field(min_length=1)
    environment: dict[str, Any] = Field(default_factory=dict)
    request_context: RequestContext = Field(default_factory=RequestContext)
    reply_tokens: int = Field(default=0, ge=0, strict=True)
    correlation: str = ""


class UsageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str | None = None
    correlation: str = ""
    usage: dict[str, Any]
    model: str | None = None
    origin: Literal["provider", "provider_stream", "outlet"] = "provider"
    aggregate: bool = False

    @model_validator(mode="after")
    def identify(self):
        if not self.request_id and not self.correlation:
            raise ValueError("request_id or correlation is required")
        return self


def merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def manager_for(profile: Profile, store: SQLiteStore) -> ContextManager:
    retrieval = copy.deepcopy(profile.retrieval)

    def parameters(name: str) -> dict:
        params = retrieval.pop(name, {}) or {}
        params["api_key"] = os.environ.get(params.pop("api_key_env", ""), "")
        return params

    embedding, query = parameters("embedding"), parameters("query")
    return ContextManager(
        ContextConfig(**profile.context),
        store=store,
        retriever=HybridRetriever(
            embedder=OpenAIEmbeddings(**embedding) if embedding.get("model") else None,
            expander=OpenAIQueryExpander(**query) if query.get("model") else None,
            **retrieval,
        ),
        counter=ByteTokenCounter()
        if profile.tokenizer in {"bytes", "litellm"}
        else TiktokenCounter(profile.tokenizer),
        pricing=CachePricing(**profile.pricing) if profile.pricing is not None else None,
        namespace=digest(profile.model_dump()),
    )


def create_app(
    *,
    api_key: str | None = None,
    state_path: str | None = None,
    config: ServiceConfig | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        key = api_key if api_key is not None else os.environ.get("LETHES_API_KEY", "")
        if len(key) < 16:
            raise RuntimeError("Set LETHES_API_KEY to a secret of at least 16 characters")
        selected = config
        if selected is None:
            config_path = os.environ.get("LETHES_CONFIG_PATH")
            selected = (
                ServiceConfig.model_validate_json(Path(config_path).read_text())
                if config_path
                else ServiceConfig()
            )
        store = SQLiteStore(
            state_path or os.environ.get("LETHES_STATE_PATH", "./data/lethes.sqlite3")
        )
        try:
            profiles = {"": selected.defaults}
            profiles.update(
                {
                    name: Profile.model_validate(merge(selected.defaults.model_dump(), value))
                    for name, value in selected.models.items()
                }
            )
            managers = {name: manager_for(profile, store) for name, profile in profiles.items()}
            app.state.key, app.state.store = key, store
            app.state.profiles, app.state.managers = profiles, managers
            yield
        finally:
            store.close()

    app = FastAPI(title="lethes", version=__version__, lifespan=lifespan)

    async def authorize(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(BEARER)],
    ):
        if not credentials or not hmac.compare_digest(
            credentials.credentials.encode(), app.state.key.encode()
        ):
            raise HTTPException(401, "Invalid API key", headers={"WWW-Authenticate": "Bearer"})

    router = APIRouter(prefix="/v2", dependencies=[Depends(authorize)])

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "version": __version__}

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("/observer/")

    @router.post("/context/prepare")
    async def prepare(body: PrepareRequest):
        started = time.perf_counter()
        model_info = body.environment.get("model_info")
        if model_info is None:
            model_info = {}
        if not isinstance(model_info, dict):
            raise HTTPException(422, "environment.model_info must be an object")
        base_model = model_info.get("base_model_id")
        if base_model is not None and not isinstance(base_model, str):
            raise HTTPException(422, "base_model_id must be a string")
        name = (
            body.model
            if body.model in app.state.profiles
            else base_model
            if base_model in app.state.profiles
            else ""
        )
        profile, original = app.state.profiles[name], app.state.managers[name]
        manager = copy.copy(original)
        context = {k: v for k, v in body.request_context.model_dump().items() if v}
        try:
            if profile.tokenizer == "litellm":
                manager.counter = LiteLLMTokenCounter(
                    profile.tokenizer_model or base_model or body.model,
                    tools=context.get("tools"),
                    tool_choice=context.get("tool_choice"),
                    prefix_messages=context.get("prefix_messages"),
                    response_format=context.get("response_format"),
                )
                overhead = 0
            else:
                overhead = len(canonical(context.get("tools") or []).encode())
                if context.get("response_format"):
                    overhead += len(canonical(context["response_format"]).encode())
                if context.get("prefix_messages"):
                    overhead += manager.counter.count(context["prefix_messages"])
            extra_reply = max(0, body.reply_tokens - manager.config.reply_reserve)
            environment = {
                **body.environment,
                "lethes_model_binding": getattr(manager.counter, "sdk_model", None)
                or base_model
                or body.model,
            }
            result = await manager.prepare(
                body.messages,
                user_id=body.user_id,
                chat_id=body.chat_id,
                model=body.model,
                environment=environment,
                request_context=context,
                correlation=body.correlation,
                extra_tokens=overhead,
                extra_reply_tokens=extra_reply,
            )
        except (ContextOverflow, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None
        observation = context_observation(
            result,
            body.model,
            manager.config.input_limit - extra_reply,
            (time.perf_counter() - started) * 1000,
        )
        manager.store.record_observation(result.request_id, observation)
        log.info("lethes.context %s", canonical(observation), extra={"lethes": observation})
        return {
            "request_id": result.request_id,
            "messages": result.messages,
            "observation": observation,
            "status": status_line(observation),
        }

    @router.post("/usage")
    async def usage(body: UsageRequest):
        store = app.state.store
        request_id = body.request_id or store.correlated_request(body.correlation)
        observation = store.observation(request_id) if request_id else None
        if not observation:
            raise HTTPException(404, "Unknown observed request")
        try:
            ContextManager(store=store).record_usage(
                request_id,
                body.usage,
                model=body.model,
                origin=body.origin,
                aggregate=body.aggregate
                or (body.origin == "outlet" and store.turn_attempts(request_id) > 1),
            )
        except (TypeError, ValueError):
            raise HTTPException(422, "Invalid usage counts") from None
        item = store.observations(request_id=request_id)[0]
        if item["authoritative"]:
            observed = {
                **observation,
                "input_tokens": item["usage"]["input_tokens"],
                "cached_tokens": item["usage"].get("cached_tokens"),
                "count_source": "provider_usage",
            }
        else:
            observed = observation
        details = {
            "request_id": request_id,
            "usage": item["usage"],
            "authoritative": item["authoritative"],
        }
        log.info("lethes.usage %s", canonical(details), extra={"lethes": details})
        return {**item, "status": status_line(observed)}

    @router.get("/observations")
    async def observations(limit: Annotated[int, Query(ge=1, le=200)] = 100):
        return {"items": app.state.store.observations(limit)}

    @router.get("/observations/{request_id}")
    async def observation(request_id: str):
        items = app.state.store.observations(request_id=request_id)
        if not items:
            raise HTTPException(404, "Unknown observed request")
        return items[0]

    app.include_router(router)

    @app.get("/observer/", include_in_schema=False)
    async def observer_index():
        return await observer_asset("index.html")

    @app.get("/observer/{asset}", include_in_schema=False)
    async def observer_asset(asset: str):
        media = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}
        if asset not in media:
            raise HTTPException(404, "Unknown observer asset")
        return Response(
            (Path(__file__).parent / "observer" / asset).read_bytes(),
            media_type=media[asset],
            headers={"Cache-Control": "no-store"},
        )

    return app


app = create_app()
