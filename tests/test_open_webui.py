import asyncio
import copy
import json

import pytest
from conftest import ScriptedRetriever, story

from lethes import Span
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
