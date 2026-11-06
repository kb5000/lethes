import copy
import importlib.util
import runpy
from contextlib import asynccontextmanager

import httpx
import pytest
from conftest import story

KEY = "local-test-service-key-1234"
AUTH = {"Authorization": "Bearer " + KEY}


@asynccontextmanager
async def running(tmp_path, config=None, key=KEY):
    module = pytest.importorskip("lethes.service")
    app = module.create_app(
        api_key=key,
        state_path=str(tmp_path / "service.sqlite3"),
        config=module.ServiceConfig.model_validate(config or {}),
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://service", headers=AUTH
        ) as client:
            yield client, app


def payload(messages=None, **kwargs):
    return {
        "user_id": "u",
        "chat_id": "c",
        "model": "m",
        "messages": messages or [{"role": "user", "content": "Private original scene."}],
        **kwargs,
    }


async def test_service_auth_and_offline_observer_assets(tmp_path):
    async with running(tmp_path) as (client, app):
        assert (await client.get("/healthz")).json()["status"] == "ok"
        for path in ["/observer/", "/observer/app.js", "/observer/style.css"]:
            response = await client.get(path)
            assert response.status_code == 200
            assert response.headers["cache-control"] == "no-store"
            assert "cdn.jsdelivr" not in response.text and KEY not in response.text
        assert (await client.get("/observer/unknown.js")).status_code == 404
        assert (
            await client.get("/v2/observations", headers={"Authorization": "Bearer wrong"})
        ).status_code == 401
        assert (await client.get("/v2/observations?limit=201")).status_code == 422
        assert (await client.get("/v2/observations/missing")).status_code == 404
        assert (
            await client.post("/v2/usage", json={"request_id": "missing", "usage": {}})
        ).status_code == 404


async def test_service_usage_replay_restart_and_private_observer_read_model(tmp_path):
    async with running(tmp_path) as (client, app):
        first = (await client.post("/v2/context/prepare", json=payload())).json()
        assert first["observation"]["count_source"] == "local_estimate"
        usage = await client.post(
            "/v2/usage",
            json={
                "request_id": first["request_id"],
                "model": "m",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 10},
                    "private_vendor": "not-for-observer",
                },
            },
        )
        assert usage.status_code == 200 and usage.json()["authoritative"]
        assert "本次实测" in usage.json()["status"]
        detail = (await client.get("/v2/observations/" + first["request_id"])).json()
        assert detail["observation"] == first["observation"]
        assert detail["usage"]["input_tokens"] == 20
        listed = await client.get("/v2/observations")
        assert "Private original scene" not in listed.text and "not-for-observer" not in listed.text
        assert "raw" not in detail["usage"]
    async with running(tmp_path) as (client, app):
        replay = (await client.post("/v2/context/prepare", json=payload())).json()
        assert replay["observation"]["input_tokens"] == 20
        assert replay["observation"]["count_source"] == "reported_usage"
        assert replay["observation"]["epoch"] == first["observation"]["epoch"]


async def test_service_model_overrides_reservations_and_branch_restore(tmp_path):
    config = {
        "defaults": {
            "context": {
                "context_window": 6500,
                "reply_reserve": 500,
                "safety_reserve": 200,
                "recent_tokens": 650,
                "minimum_recent_turns": 1,
            }
        },
        "models": {"small": {"context": {"context_window": 4000}}},
    }
    async with running(tmp_path, config) as (client, app):
        raw = story()
        first = (
            await client.post("/v2/context/prepare", json=payload(raw, reply_tokens=900))
        ).json()
        assert first["observation"]["input_limit"] == 5400
        assert first["observation"]["input_tokens"] <= 5400
        edited = copy.deepcopy(raw)
        edited[-1]["content"] = "A different scene."
        changed = (await client.post("/v2/context/prepare", json=payload(edited))).json()
        assert changed["observation"]["layout"]["head"] != first["observation"]["layout"]["head"]
        restored = (await client.post("/v2/context/prepare", json=payload(raw))).json()
        assert restored["messages"] == first["messages"]
        smaller = (
            await client.post("/v2/context/prepare", json={**payload(), "model": "small"})
        ).json()
        assert smaller["observation"]["input_limit"] == 3300


async def test_usage_correlation_aggregate_and_origin_priority(tmp_path):
    async with running(tmp_path) as (client, app):
        first = (await client.post("/v2/context/prepare", json=payload(correlation="turn"))).json()
        await client.post(
            "/v2/usage",
            json={
                "request_id": first["request_id"],
                "usage": {"prompt_tokens": 30},
                "origin": "provider_stream",
            },
        )
        outlet = (
            await client.post(
                "/v2/usage",
                json={"correlation": "turn", "usage": {"prompt_tokens": 999}, "origin": "outlet"},
            )
        ).json()
        assert outlet["usage"]["input_tokens"] == 30
        await client.post("/v2/context/prepare", json=payload(correlation="turn"))
        total = (
            await client.post(
                "/v2/usage",
                json={"correlation": "turn", "usage": {"prompt_tokens": 999}, "origin": "outlet"},
            )
        ).json()
        assert total["usage"]["aggregate"] and not total["authoritative"]
        assert "本次实测" not in total["status"]


@pytest.mark.parametrize(
    "override",
    [
        {"reply_tokens": True},
        {"messages": []},
        {"environment": {"model_info": []}},
        {"environment": {"model_info": {"base_model_id": []}}},
        {"request_context": {"prefix_messages": "bad"}},
        {"request_context": {"tools": "bad"}},
        {"request_context": {"unknown_envelope": "bad"}},
    ],
)
async def test_invalid_prepare_is_rejected(tmp_path, override):
    async with running(tmp_path) as (client, app):
        assert (
            await client.post("/v2/context/prepare", json={**payload(), **override})
        ).status_code == 422


async def test_capacity_and_invalid_usage_are_explicit(tmp_path):
    async with running(tmp_path) as (client, app):
        response = await client.post(
            "/v2/context/prepare", json=payload([{"role": "user", "content": "x" * 50000}])
        )
        assert response.status_code == 422
        first = (await client.post("/v2/context/prepare", json=payload())).json()
        invalid = await client.post(
            "/v2/usage", json={"request_id": first["request_id"], "usage": {"input_tokens": None}}
        )
        assert invalid.status_code == 422
        estimated = (
            await client.post(
                "/v2/usage",
                json={
                    "request_id": first["request_id"],
                    "usage": {"prompt_tokens": 1, "estimated": True},
                },
            )
        ).json()
        assert not estimated["authoritative"]


async def test_observer_and_status_exclude_malformed_cache_counts(tmp_path):
    async with running(tmp_path) as (client, app):
        first = (await client.post("/v2/context/prepare", json=payload())).json()
        for details in [{"cached_tokens": {"not": "a count"}}, ["malformed"]]:
            response = await client.post(
                "/v2/usage",
                json={
                    "request_id": first["request_id"],
                    "usage": {"prompt_tokens": 40, "prompt_tokens_details": details},
                },
            )
            assert response.status_code == 200
            assert response.json()["authoritative"]
            assert "cached_tokens" not in response.json()["usage"]
            assert "缓存读" not in response.json()["status"]


async def test_service_requires_key_and_valid_profile(tmp_path):
    with pytest.raises(RuntimeError, match="LETHES_API_KEY"):
        async with running(tmp_path, key="short"):
            pass
    with pytest.raises(ValueError):
        async with running(tmp_path, {"defaults": {"context": {"context_window": 1}}}):
            pass


async def test_model_mapping_changes_isolate_usage_but_counter_switch_preserves_it(
    tmp_path, monkeypatch
):
    service = pytest.importorskip("lethes.service")
    from lethes.tokens import ByteTokenCounter

    class ModelCounter(ByteTokenCounter):
        def __init__(self, model, **kwargs):
            self.sdk_model = self.model = model
            self.cache_identity = "fake:" + model

    monkeypatch.setattr(service, "LiteLLMTokenCounter", ModelCounter)
    async with running(tmp_path) as (client, app):
        first = (await client.post("/v2/context/prepare", json=payload())).json()
        await client.post(
            "/v2/usage", json={"request_id": first["request_id"], "usage": {"prompt_tokens": 30}}
        )
    async with running(
        tmp_path, {"defaults": {"tokenizer": "litellm", "tokenizer_model": "m"}}
    ) as (client, app):
        same = (await client.post("/v2/context/prepare", json=payload())).json()
        assert same["observation"]["input_tokens"] == 30
        assert same["observation"]["count_source"] == "reported_usage"
    async with running(
        tmp_path, {"defaults": {"tokenizer": "litellm", "tokenizer_model": "different"}}
    ) as (client, app):
        changed = (await client.post("/v2/context/prepare", json=payload())).json()
        assert changed["observation"]["count_source"] == "local_estimate"


@pytest.mark.skipif(importlib.util.find_spec("litellm") is None, reason="optional SDK")
async def test_service_litellm_envelope_and_deployment_alias(tmp_path):
    config = {"models": {"workspace": {"tokenizer": "litellm", "tokenizer_model": "gpt-4o"}}}
    async with running(tmp_path, config) as (client, app):
        first = (
            await client.post("/v2/context/prepare", json={**payload(), "model": "workspace"})
        ).json()
        request = {
            **payload(),
            "model": "workspace",
            "request_context": {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "roll", "parameters": {"type": "object"}},
                    }
                ]
            },
        }
        richer = (await client.post("/v2/context/prepare", json=request)).json()
        assert richer["observation"]["input_tokens"] > first["observation"]["input_tokens"]


async def test_remote_filter_end_to_end_tool_rehydration_and_usage(tmp_path):
    remote = runpy.run_path("src/external/open_webui_service_filter.py")
    adapter = remote["Filter"]()
    async with running(
        tmp_path,
        {
            "defaults": {
                "context": {
                    "context_window": 6500,
                    "reply_reserve": 500,
                    "safety_reserve": 200,
                    "recent_tokens": 650,
                    "minimum_recent_turns": 1,
                }
            }
        },
    ) as (client, app):
        sources = []

        async def post(path, data):
            if path.endswith("prepare"):
                sources.append(copy.deepcopy(data["messages"]))
            response = await client.post(path, json=data)
            response.raise_for_status()
            return response.json()

        adapter._post = post
        metadata = {"chat_id": "c", "message_id": "m", "user_id": "u"}
        body = {"model": "m", "messages": story()}
        events = []

        async def emit(event):
            events.append(event)

        original = copy.deepcopy(body["messages"])
        await adapter.inlet(body, __metadata__=metadata)
        await adapter.request(body, __metadata__=metadata, __event_emitter__=emit)
        first_wire = copy.deepcopy(body["messages"])
        extra = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "t", "type": "function", "function": {"name": "roll", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "t", "content": "20"},
        ]
        body["messages"] += extra
        await adapter.request(body, __metadata__=metadata)
        assert sources[-1] == original + extra
        assert body["messages"] == first_wire + extra
        context = metadata[adapter._key(None)]
        await adapter.stream({"model": "m"}, __metadata__=metadata)
        await adapter.stream(
            {"usage": {"prompt_tokens": 1000}}, __metadata__=metadata, __event_emitter__=emit
        )
        assert "本次实测" in events[-1]["data"]["description"]
        assert app.state.store.usage(context["request_id"])["input_tokens"] == 1000
        assert (await client.get("/v2/observations")).json()["items"]


async def test_remote_filter_fails_prepare_explicitly_but_usage_does_not_damage_output():
    adapter = runpy.run_path("src/external/open_webui_service_filter.py")["Filter"]()

    async def offline(path, data):
        raise httpx.ConnectError("private upstream detail")

    adapter._post = offline
    metadata = {"chat_id": "c", "message_id": "m", "user_id": "u"}
    body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    original = copy.deepcopy(body)
    with pytest.raises(RuntimeError, match="check service logs"):
        await adapter.request(body, __metadata__=metadata)
    assert body == original
    metadata[adapter._key(None)]["request_id"] = "id"
    event = {"usage": {"prompt_tokens": 100}}
    assert await adapter.stream(event, __metadata__=metadata) is event
    internal = copy.deepcopy(body)
    await adapter.inlet(internal, __metadata__=metadata, __task__="title_generation")
    assert await adapter.request(internal, __metadata__=metadata) == internal
