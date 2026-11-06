import copy
import sys
from types import SimpleNamespace

import pytest
from conftest import prepare, story

from lethes import ContextConfig, ContextManager, LiteLLMTokenCounter, SQLiteStore
from lethes.history import canonical
from lethes.tokens import Measurement, UsageAwareCounter


@pytest.fixture
def sdk(monkeypatch):
    calls = []

    def token_counter(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        total = 3 + sum(len(str(m.get("content") or "")) + 4 for m in kwargs.get("messages", []))
        total += 20 * len(kwargs.get("tools") or [])
        if "text" in kwargs:
            total = len(kwargs["text"])
        if kwargs.get("messages"):
            kwargs["messages"][0]["content"] = "SDK mutated its own copy"
        return total

    module = SimpleNamespace(
        model_cost={"gpt-4o": {}, "other-model": {}}, token_counter=token_counter
    )
    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setattr("lethes.tokens.version", lambda name: "test-version")
    return calls


def test_litellm_counts_model_messages_tools_and_detaches_input(sdk):
    raw = [{"role": "user", "content": [{"type": "text", "text": "中文"}]}]
    original = copy.deepcopy(raw)
    counter = LiteLLMTokenCounter(
        "gpt-4o",
        tools=[{"type": "function", "function": {"name": "roll"}}],
        tool_choice="auto",
        prefix_messages=[{"role": "system", "content": "Narrate"}],
        response_format={"type": "json_object"},
    )
    first = counter.count(raw)
    assert first > 20 and counter.count(raw) == first
    assert raw == original
    assert sdk[0]["tools"][0]["function"]["name"] == "roll"
    assert sdk[0]["tool_choice"] == "auto"
    assert sdk[0]["use_default_image_token_count"] is True
    assert sdk[0]["messages"][0]["content"] == "Narrate"
    assert len(sdk) == 2


def test_unknown_model_does_not_silently_become_gpt(sdk):
    with pytest.raises(ValueError, match="Unknown LiteLLM model"):
        LiteLLMTokenCounter("unmapped-workspace-alias")
    assert LiteLLMTokenCounter("openai/gpt-4o").sdk_model == "gpt-4o"
    assert (
        LiteLLMTokenCounter("gpt-4o").cache_identity
        != LiteLLMTokenCounter("other-model").cache_identity
    )


async def test_exact_reported_prompt_count_wins_and_survives_restart(tmp_path):
    path = tmp_path / "counts.sqlite"
    raw = [{"role": "user", "content": "Some input."}]
    manager = ContextManager(store=SQLiteStore(path))
    first = await prepare(manager, raw, model="model-a")
    manager.record_usage(
        first.request_id,
        {
            "prompt_tokens": 40,
            "completion_tokens": 9000,
            "prompt_tokens_details": {"cached_tokens": 20},
        },
        model="model-a",
    )
    manager.store.close()
    resumed = ContextManager(store=SQLiteStore(path))
    replay = await prepare(resumed, raw, model="model-a")
    assert replay.input_tokens == 40
    assert replay.count_source == "reported_usage"
    assert replay.count_reference == first.request_id
    assert replay.messages == first.messages and replay.snapshot.epoch == first.snapshot.epoch
    assert resumed.store.usage(first.request_id)["cached_tokens"] == 20


async def test_new_messages_use_real_prefix_plus_estimated_delta():
    manager = ContextManager()
    raw = [{"role": "user", "content": "First"}]
    first = await prepare(manager, raw, model="m")
    manager.record_usage(first.request_id, {"prompt_tokens": 20})
    extended = raw + [
        {"role": "assistant", "content": "Reply"},
        {"role": "user", "content": "Next"},
    ]
    result = await prepare(manager, extended, model="m")
    delta = manager.counter.count(extended) - manager.counter.count(raw)
    assert result.input_tokens == 20 + delta
    assert result.count_source == "reported_prefix_plus_estimate"
    assert result.count_reference == first.request_id


async def test_model_provider_and_configuration_isolate_measurements():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="m", environment={"provider": "one"})
    manager.record_usage(first.request_id, {"prompt_tokens": 10})
    for kwargs in [
        dict(model="different", environment={"provider": "one"}),
        dict(model="m", environment={"provider": "two"}),
        dict(model="m", environment={"provider": "one", "tools": [{}]}),
    ]:
        result = await prepare(manager, raw, **kwargs)
        assert result.count_source == "local_estimate"
        assert result.input_tokens != 10


async def test_edited_prompt_never_claims_old_count_is_measured():
    manager = ContextManager()
    raw = [{"role": "user", "content": "old scene"}]
    first = await prepare(manager, raw, model="m")
    manager.record_usage(first.request_id, {"prompt_tokens": 10})
    edited = [{"role": "user", "content": "edited scene"}]
    changed = await prepare(manager, edited, model="m")
    assert changed.count_source != "reported_usage"
    assert changed.input_tokens == manager.counter.count(edited)
    restored = await prepare(manager, raw, model="m")
    assert restored.input_tokens == 10 and restored.count_source == "reported_usage"


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": -1},
        {"prompt_tokens": True},
        {"prompt_tokens": 0},
        {"prompt_tokens": 2, "source": "local_estimate"},
        {"prompt_tokens": 2, "litellm_count_source": "token_count_model"},
        {"prompt_tokens": 2, "estimated": True},
        {"completion_tokens": 5},
    ],
)
async def test_invalid_missing_and_estimated_usage_cannot_override_counter(usage):
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="m")
    manager.record_usage(first.request_id, usage)
    replay = await prepare(manager, raw, model="m")
    assert replay.count_source == "local_estimate"
    assert replay.input_tokens == manager.counter.count(raw)


async def test_reported_model_mismatch_is_preserved_but_not_used_for_budget():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="expected")
    manager.record_usage(first.request_id, {"prompt_tokens": 10}, model="actually-different")
    manager.record_usage(first.request_id, {"completion_tokens": 5})
    assert manager.store.usage(first.request_id)["input_tokens"] == 10
    replay = await prepare(manager, raw, model="expected")
    assert replay.count_source == "local_estimate"


async def test_reply_reserve_is_not_added_to_prompt_count():
    manager = ContextManager(
        ContextConfig(context_window=3000, reply_reserve=100, safety_reserve=100)
    )
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="m", extra_reply_tokens=500, extra_tokens=50)
    assert first.input_tokens == manager.counter.count(raw) + 50
    manager.record_usage(first.request_id, {"prompt_tokens": 120})
    replay = await prepare(manager, raw, model="m", extra_reply_tokens=600, extra_tokens=50)
    assert replay.input_tokens == 120 and replay.count_source == "reported_usage"


async def test_actual_overflow_rebuilds_and_archives_same_history_layout():
    manager = ContextManager(
        ContextConfig(
            context_window=12000,
            reply_reserve=500,
            safety_reserve=200,
            recent_tokens=500,
            minimum_recent_turns=1,
        )
    )
    raw = story(16)
    first = await prepare(manager, raw, model="m")
    assert first.snapshot.tail_start == 1
    actual = manager.config.input_limit + 1000
    manager.record_usage(first.request_id, {"prompt_tokens": actual})
    rebuilt = await prepare(manager, raw, model="m")
    assert rebuilt.reason == "count_capacity"
    assert rebuilt.input_tokens <= manager.config.input_limit
    assert rebuilt.snapshot.epoch != first.snapshot.epoch
    assert rebuilt.messages[-1] == raw[-1]
    assert manager.store.usage(first.request_id)["input_tokens"] == actual
    again = await prepare(manager, raw, model="m")
    assert again.messages == rebuilt.messages and again.snapshot.epoch == rebuilt.snapshot.epoch


def test_prefix_calibration_does_not_apply_negative_offset_to_unrelated_content():
    class Counter:
        def count(self, messages):
            return 3 + sum(len(m["content"]) for m in messages)

    old = [{"role": "user", "content": "x" * 100}]
    sample = Measurement(canonical(old), 20, "r")
    calibrated = UsageAwareCounter(
        Counter(), [sample], lambda wire: sample if wire == sample.wire else None
    )
    unrelated = calibrated.describe([{"role": "user", "content": "short"}])
    assert unrelated.tokens == 8 and unrelated.source == "local_estimate"


async def test_same_model_usage_survives_switch_to_litellm_counter(sdk):
    store = SQLiteStore()
    raw = [{"role": "user", "content": "Same actual request."}]
    old = ContextManager(store=store)
    first = await prepare(old, raw, model="gpt-4o")
    old.record_usage(first.request_id, {"prompt_tokens": 42}, model="gpt-4o")
    updated = ContextManager(store=store, counter=LiteLLMTokenCounter("gpt-4o"))
    result = await prepare(updated, raw, model="gpt-4o")
    assert result.input_tokens == 42
    assert result.count_source == "reported_usage"
    assert result.count_reference == first.request_id


async def test_new_reported_model_invalidates_prior_alias_measurements():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="alias")
    manager.record_usage(first.request_id, {"prompt_tokens": 10}, model="alias")
    second = await prepare(manager, raw, model="alias")
    manager.record_usage(second.request_id, {"prompt_tokens": 15}, model="different-route")
    third = await prepare(manager, raw, model="alias")
    assert third.count_source == "local_estimate"
    manager.record_usage(third.request_id, {"prompt_tokens": 12}, model="alias")
    restored = await prepare(manager, raw, model="alias")
    assert restored.input_tokens == 12 and restored.count_source == "reported_usage"


async def test_anthropic_full_input_includes_cache_reads_and_writes():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="claude")
    manager.record_usage(
        first.request_id,
        {
            "input_tokens": 10,
            "cache_read_input_tokens": 20,
            "cache_creation_input_tokens": 5,
            "output_tokens": 100,
        },
    )
    result = await prepare(manager, raw, model="claude")
    assert result.input_tokens == 35
    assert result.count_source == "reported_usage"


async def test_outlet_cannot_overwrite_per_request_provider_usage_with_turn_total():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="m")
    manager.record_usage(
        first.request_id, {"prompt_tokens": 10, "completion_tokens": 5}, origin="provider_stream"
    )
    manager.record_usage(
        first.request_id,
        {"prompt_tokens": 999, "completion_tokens": 500},
        origin="outlet",
        aggregate=True,
    )
    result = await prepare(manager, raw, model="m")
    assert result.input_tokens == 10
    assert manager.store.usage(first.request_id)["output_tokens"] == 5


async def test_aggregate_outlet_is_saved_but_not_used_as_one_request_measurement():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first = await prepare(manager, raw, model="m")
    manager.record_usage(first.request_id, {"prompt_tokens": 999}, origin="outlet", aggregate=True)
    result = await prepare(manager, raw, model="m")
    assert result.count_source == "local_estimate"
    assert manager.store.usage(first.request_id)["input_tokens"] == 999


async def test_inferred_counter_models_do_not_share_usage_under_default_model(sdk):
    store = SQLiteStore()
    raw = [{"role": "user", "content": "hello"}]
    first_manager = ContextManager(store=store, counter=LiteLLMTokenCounter("gpt-4o"))
    first = await prepare(first_manager, raw)
    first_manager.record_usage(first.request_id, {"prompt_tokens": 42}, model="gpt-4o")
    other_manager = ContextManager(store=store, counter=LiteLLMTokenCounter("other-model"))
    other = await prepare(other_manager, raw)
    assert other.count_source == "local_estimate"
    assert other.input_tokens != 42


def test_disabled_sdk_counter_is_not_accepted_as_zero_budget(sdk, monkeypatch):
    monkeypatch.setattr(sys.modules["litellm"], "token_counter", lambda **kwargs: 0)
    with pytest.raises(ValueError, match="invalid/disabled"):
        LiteLLMTokenCounter("gpt-4o").count([{"role": "user", "content": "hello"}])


async def test_tool_property_order_is_part_of_measured_request_identity():
    manager = ContextManager()
    raw = [{"role": "user", "content": "hello"}]
    first_context = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "parameters": {
                        "type": "object",
                        "properties": {"first": {"type": "string"}, "second": {"type": "string"}},
                    }
                },
            }
        ]
    }
    second_context = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "parameters": {
                        "type": "object",
                        "properties": {"second": {"type": "string"}, "first": {"type": "string"}},
                    }
                },
            }
        ]
    }
    first = await prepare(manager, raw, model="m", request_context=first_context)
    manager.record_usage(first.request_id, {"prompt_tokens": 10})
    second = await prepare(manager, raw, model="m", request_context=second_context)
    assert second.count_source != "reported_usage"
    assert second.input_tokens != 10
