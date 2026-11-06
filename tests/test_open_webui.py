import asyncio
import copy
import json

import pytest
from conftest import ScriptedRetriever, story

from lethes import Span
from lethes.history import digest
from lethes.integrations.open_webui import OpenWebUIFilter


@pytest.fixture
def adapter(tmp_path):
    adapter = OpenWebUIFilter()
    adapter.valves.state_path = str(tmp_path / "owui.sqlite")
    adapter.valves.context_window = 6500
    adapter.valves.reply_reserve = 500
    adapter.valves.safety_reserve = 200
    adapter.valves.recent_tokens = 650
    adapter.valves.minimum_recent_turns = 1
    adapter._get_manager().retriever = ScriptedRetriever((Span(1, 3),))
    return adapter


def meta(chat="c", message="m"):
    return {"chat_id": chat, "message_id": message, "user_id": "u"}


async def test_inlet_leaves_full_history_and_request_compiles(adapter):
    metadata = meta()
    body = {"model": "model", "messages": story(), "tools": []}
    original = copy.deepcopy(body)
    assert await adapter.inlet(body, __metadata__=metadata) == original
    events = []

    async def emit(event):
        events.append(event)

    await adapter.request(body, __metadata__=metadata, __event_emitter__=emit)
    assert len(body["messages"]) < len(original["messages"])
    assert events[-1]["data"]["done"] is True
    first = copy.deepcopy(body["messages"])
    await adapter.request(body, __metadata__=metadata)
    assert body["messages"] == first
    json.dumps(metadata)  # Host lifecycle metadata must remain serializable.


async def test_reconstructed_tool_hop_and_already_compiled_hop(adapter):
    metadata = meta()
    original = story()
    body = {"model": "model", "messages": copy.deepcopy(original)}
    await adapter.request(body, __metadata__=metadata)
    compiled = copy.deepcopy(body["messages"])
    tools = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t", "type": "function", "function": {"name": "roll", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t", "content": "20"},
    ]
    body["messages"] = compiled + tools
    await adapter.request(body, __metadata__=metadata)
    assert body["messages"] == compiled + tools
    body["messages"] = original + tools
    await adapter.request(body, __metadata__=metadata)
    assert body["messages"] == compiled + tools


async def test_background_tasks_and_unscoped_api_do_not_share_state(adapter):
    body = {"model": "model", "messages": story()}
    original = copy.deepcopy(body["messages"])
    metadata = meta()
    await adapter.inlet(body, __metadata__=metadata, __task__="title_generation")
    await adapter.request(body, __metadata__=metadata)
    assert body["messages"] == original
    body = {"model": "model", "messages": story()}
    await adapter.request(body, __metadata__={})
    assert body["messages"] == original


async def test_no_shared_request_telemetry_and_pure_api_no_emitter(adapter):
    async def run(chat):
        metadata = meta(chat)
        body = {"model": "model", "messages": story()}
        await adapter.request(body, __metadata__=metadata)
        context = metadata[adapter._key(None)]
        event = {"usage": {"prompt_tokens": 1000, "prompt_tokens_details": {"cached_tokens": 700}}}
        assert await adapter.stream(event, __metadata__=metadata) is event
        await adapter.outlet(body, __metadata__=metadata)
        return context["request_id"]

    ids = await asyncio.gather(run("a"), run("b"))
    assert ids[0] != ids[1]
    assert all(adapter._get_manager().store.usage(i)["cached_tokens"] == 700 for i in ids)


async def test_completed_callback_with_fresh_metadata_can_correlate(adapter):
    metadata = meta()
    body = {"model": "model", "messages": story()}
    await adapter.request(body, __metadata__=metadata)
    request_id = metadata[adapter._key(None)]["request_id"]
    completed = {
        "model": "model",
        "chat_id": "c",
        "id": "m",
        "messages": [{"role": "assistant", "content": "story", "usage": {"prompt_tokens": 3000}}],
    }
    await adapter.outlet(completed, __metadata__=meta())
    assert adapter._get_manager().store.usage(request_id)["input_tokens"] == 3000


async def test_stream_model_from_earlier_chunk_checks_usage_and_resets_per_hop(adapter):
    metadata = meta()
    body = {"model": "requested", "messages": story()}
    await adapter.request(body, __metadata__=metadata)
    context = metadata[adapter._key(None)]
    first_id = context["request_id"]
    await adapter.stream({"model": "different-route", "choices": []}, __metadata__=metadata)
    await adapter.stream({"usage": {"prompt_tokens": 100}}, __metadata__=metadata)
    usage = adapter._get_manager().store.usage(first_id)
    assert usage["reported_model"] == "different-route"
    assert usage["measurement_eligible"] is False
    await adapter.request(body, __metadata__=metadata)
    await adapter.stream({"usage": {"prompt_tokens": 101}}, __metadata__=metadata)
    next_usage = adapter._get_manager().store.usage(context["request_id"])
    assert "reported_model" not in next_usage
    assert context["response_model"] is None


async def test_valves_changes_invalidate_configuration(adapter):
    first = adapter._get_manager()
    adapter.valves.recent_tokens += 1
    second = adapter._get_manager()
    assert first is not second


async def test_model_system_prompt_and_tool_schema_change_domain(adapter):
    metadata = meta()
    body = {"model": "workspace", "messages": story()}
    await adapter.request(
        body,
        __metadata__=metadata,
        __model__={"info": {"base_model_id": "base", "params": {"system": "one"}}},
    )
    original_epoch = adapter._get_manager().store.correlated_request(
        adapter._correlation(metadata, None, body)
    )
    body["messages"] = story()
    await adapter.inlet(body, __metadata__=metadata)
    await adapter.request(
        body,
        __metadata__=metadata,
        __model__={"info": {"base_model_id": "base", "params": {"system": "two"}}},
    )
    assert metadata[adapter._key(None)]["request_id"] != original_epoch


async def test_stream_strings_done_and_partial_usage_are_not_dropped(adapter):
    metadata = meta()
    await adapter.request({"model": "m", "messages": story()}, __metadata__=metadata)
    request_id = metadata[adapter._key(None)]["request_id"]
    for event in [{"data": "[DONE]"}, {"data": "not-json"}, {"data": []}]:
        assert await adapter.stream(event, __metadata__=metadata) is event
    await adapter.stream(
        {"message": {"usage": {"input_tokens": 100, "cache_read_input_tokens": 800}}},
        __metadata__=metadata,
    )
    await adapter.stream({"data": 'data: {"usage": {"output_tokens": 20}}'}, __metadata__=metadata)
    usage = adapter._get_manager().store.usage(request_id)
    assert usage["cached_tokens"] == 800
    assert usage["output_tokens"] == 20


async def test_usage_after_state_path_change_goes_to_original_store(adapter, tmp_path):
    metadata = meta()
    await adapter.request({"model": "m", "messages": story()}, __metadata__=metadata)
    original = adapter._get_manager()
    request_id = metadata[adapter._key(None)]["request_id"]
    adapter.valves.state_path = str(tmp_path / "different.sqlite")
    await adapter.stream({"usage": {"prompt_tokens": 100}}, __metadata__=metadata)
    assert original.store.usage(request_id)["input_tokens"] == 100


async def test_merged_system_tool_hop_rehydrates_all_original_history(adapter):
    metadata = meta()
    original = story()
    body = {"model": "m", "messages": copy.deepcopy(original)}
    await adapter.request(body, __metadata__=metadata)
    compiled = copy.deepcopy(body["messages"])
    tool = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "t", "type": "function", "function": {"name": "roll", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "t", "content": "20"},
    ]
    body["messages"] = [{"role": "system", "content": "Merged system prompt"}] + compiled[1:] + tool
    await adapter.request(body, __metadata__=metadata)
    source = metadata[adapter._key(None)]["source"]
    assert source[1:] == original[1:] + tool
    assert body["messages"][-2:] == tool


async def test_request_specific_litellm_models_and_tools_are_not_shared(adapter, monkeypatch):
    seen = []
    from lethes.tokens import ByteTokenCounter

    class FakeCounter(ByteTokenCounter):
        def __init__(self, model, **kwargs):
            self.model = model
            self.sdk_model = model
            self.cache_identity = "sdk:" + model
            seen.append((model, kwargs))

    monkeypatch.setattr("lethes.integrations.open_webui.LiteLLMTokenCounter", FakeCounter)
    adapter.valves.tokenizer = "litellm"
    base = adapter._get_manager()

    async def request(model, chat):
        body = {
            "model": "workspace-" + chat,
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": chat}}],
        }
        metadata = meta(chat)
        await adapter.request(
            body,
            __metadata__=metadata,
            __model__={"info": {"base_model_id": model, "params": {"system": "System " + chat}}},
        )
        return metadata

    await asyncio.gather(request("gpt-4o", "a"), request("other-model", "b"))
    assert {m for m, _ in seen} == {"gpt-4o", "other-model"}
    assert {k["tools"][0]["function"]["name"] for _, k in seen} == {"a", "b"}
    assert isinstance(base.counter, ByteTokenCounter) and not isinstance(base.counter, FakeCounter)


async def test_aggregated_outlet_after_tool_hops_is_not_a_prompt_measurement(adapter):
    metadata = meta()
    body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    await adapter.request(body, __metadata__=metadata)
    # A repeated provider hop may have the same compiled body, but a turn total
    # still cannot be attached to just the last per-call request ID.
    await adapter.request(body, __metadata__=metadata)
    request_id = metadata[adapter._key(None)]["request_id"]
    await adapter.outlet({"model": "m", "usage": {"prompt_tokens": 999}}, __metadata__=metadata)
    assert adapter._get_manager().store.usage(request_id)["aggregate"] is True
    replay_meta = meta("c", "another-message")
    replay_body = {"model": "m", "messages": [{"role": "user", "content": "hello"}]}
    await adapter.request(replay_body, __metadata__=replay_meta)
    assert replay_body["messages"] == [{"role": "user", "content": "hello"}]


async def test_compact_status_persists_dispatch_budget_and_updates_to_actual_usage(adapter, caplog):
    caplog.set_level("INFO", logger="lethes.integrations.open_webui")
    metadata = meta()
    events = []

    async def emit(event):
        events.append(event)

    body = {"model": "m", "messages": story(), "max_tokens": 900}
    await adapter.request(body, __metadata__=metadata, __event_emitter__=emit)
    compiled = copy.deepcopy(body["messages"])
    request_id = metadata[adapter._key(None)]["request_id"]
    store = adapter._get_manager().store
    observation = store.observation(request_id)
    assert observation["input_limit"] == adapter._get_manager().config.input_limit - 400
    assert observation["elapsed_ms"] >= 0
    assert observation["count_source"] == "local_estimate"
    assert "≈" in events[0]["data"]["description"]
    assert str(observation["input_limit"]) in events[0]["data"]["description"].replace(",", "")
    event = {
        "model": "m",
        "usage": {"prompt_tokens": 1234, "prompt_tokens_details": {"cached_tokens": 800}},
    }
    assert await adapter.stream(event, __metadata__=metadata, __event_emitter__=emit) is event
    assert len(events) == 2
    actual = events[-1]["data"]["description"]
    assert "1,234" in actual and "本次实测" in actual and "缓存读 800" in actual
    assert "≈" not in actual and "\n" not in actual
    # Output-only chunks and duplicate reports do not append another UI status.
    await adapter.stream(event, __metadata__=metadata, __event_emitter__=emit)
    await adapter.stream(
        {"usage": {"completion_tokens": 25}}, __metadata__=metadata, __event_emitter__=emit
    )
    assert len(events) == 2
    assert store.observation(request_id) == observation  # Preserve the original dispatch decision.
    assert store.usage(request_id)["input_tokens"] == 1234
    assert body["messages"] == compiled
    assert not any(m["content"].startswith("lethes ·") for m in body["messages"])
    assert story()[1]["content"] not in caplog.text
    assert any(r.lethes.get("input_tokens") == 1234 for r in caplog.records if hasattr(r, "lethes"))


async def test_fresh_completed_metadata_can_render_persisted_observation_after_restart(adapter):
    metadata = meta()
    body = {"model": "m", "messages": story()}
    await adapter.request(body, __metadata__=metadata)
    request_id = metadata[adapter._key(None)]["request_id"]
    restarted = OpenWebUIFilter()
    restarted.valves = adapter.valves.model_copy()
    events = []

    async def emit(event):
        events.append(event)

    completed = {"model": "m", "chat_id": "c", "id": "m", "usage": {"prompt_tokens": 1200}}
    await restarted.outlet(completed, __metadata__=meta(), __event_emitter__=emit)
    assert "1,200" in events[-1]["data"]["description"]
    assert restarted._get_manager().store.observation(request_id)["reason"] == "initial"
    restarted._get_manager().store.delete_scope(digest(["u", "c"]))
    assert restarted._get_manager().store.observation(request_id) is None


@pytest.mark.parametrize(
    "usage,model",
    [
        ({"prompt_tokens": 1000, "estimated": True}, "m"),
        ({"prompt_tokens": 1000}, "different-route"),
        ({"completion_tokens": 1000}, "m"),
        ({"prompt_tokens": True}, "m"),
    ],
)
async def test_unreliable_usage_never_appears_as_a_real_input_count(adapter, usage, model):
    metadata = meta()
    events = []

    async def emit(event):
        events.append(event)

    await adapter.request(
        {"model": "m", "messages": story()}, __metadata__=metadata, __event_emitter__=emit
    )
    await adapter.stream(
        {"model": model, "usage": usage}, __metadata__=metadata, __event_emitter__=emit
    )
    assert len(events) == 1
    assert "估算" in events[0]["data"]["description"]


async def test_aggregate_usage_is_logged_without_replacing_single_request_status(adapter, caplog):
    caplog.set_level("INFO", logger="lethes.integrations.open_webui")
    metadata = meta()
    events = []

    async def emit(event):
        events.append(event)

    body = {"model": "m", "messages": story()}
    await adapter.request(body, __metadata__=metadata, __event_emitter__=emit)
    await adapter.request(body, __metadata__=metadata, __event_emitter__=emit)
    await adapter.outlet(
        {"model": "m", "usage": {"prompt_tokens": 9999}},
        __metadata__=metadata,
        __event_emitter__=emit,
    )
    assert len(events) == 2
    assert any(r.lethes.get("aggregate") is True for r in caplog.records if hasattr(r, "lethes"))


async def test_disconnected_status_emitter_does_not_fail_model_request(adapter):
    async def disconnected(event):
        raise ConnectionError("browser disconnected")

    body = {"model": "m", "messages": story()}
    metadata = meta()
    assert (
        await adapter.request(body, __metadata__=metadata, __event_emitter__=disconnected) is body
    )
    request_id = metadata[adapter._key(None)]["request_id"]
    assert adapter._get_manager().store.observation(request_id)


async def test_disabling_status_preserves_manager_prefix_and_audit(adapter):
    metadata = meta()
    body = {"model": "m", "messages": story()}
    await adapter.request(body, __metadata__=metadata)
    first_wire = copy.deepcopy(body["messages"])
    manager = adapter._get_manager()
    adapter.valves.show_status = False
    events = []

    async def emit(event):
        events.append(event)

    await adapter.request(body, __metadata__=metadata, __event_emitter__=emit)
    assert adapter._get_manager() is manager
    assert body["messages"] == first_wire and not events
    assert manager.store.observation(metadata[adapter._key(None)]["request_id"])


def test_local_embedding_settings_bind_retriever(adapter):
    adapter.valves.embedding_model = "local-small"
    adapter.valves.embedding_api_base = "http://embedding:80/v1"
    adapter.valves.embedding_query_prefix = "query: "
    adapter.valves.embedding_chunk_chars = 384
    adapter.valves.embedding_chunk_overlap = 64
    adapter.valves.semantic_floor = 0.1
    retriever = adapter._get_manager().retriever
    assert retriever.embedding_query_prefix == "query: "
    assert retriever.embedding_chunk_chars == 384
    assert retriever.embedding_chunk_overlap == 64
    assert retriever.semantic_floor == 0.1
