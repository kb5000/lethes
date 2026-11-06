import asyncio
import copy
import json

import pytest
from conftest import ScriptedRetriever, prepare, story

from lethes import ContextConfig, ContextManager, ContextOverflow, Span, SQLiteStore
from lethes.engine import assemble
from lethes.history import History, digest


async def test_short_chat_passes_through_unchanged(manager, retriever):
    raw = [{"role": "system", "content": "roleplay"}, {"role": "user", "content": "Hello"}]
    result = await prepare(manager, raw)
    assert result.messages == raw
    assert not result.snapshot.reminders
    assert raw[-1]["content"] == "Hello"


async def test_reminder_is_exact_original_not_extracted_facts(manager):
    raw = story()
    raw[1]["content"] = "这把剑只有在下雨时才能出鞘。\n“除非献出记忆”，她补充道。"
    result = await prepare(manager, raw)
    reminder = result.snapshot.reminders[0]
    excerpts = json.loads(json.loads(reminder.payload)["content"].split("\n", 1)[1])
    assert excerpts[0]["message"] == raw[1]
    assert excerpts[0]["path_revision"] == History.from_messages(raw).heads[1]
    assert result.estimated_tokens <= manager.config.input_limit
    assert result.messages[-1] == raw[-1]


async def test_reminders_remain_in_place_and_prefix_grows(manager, retriever):
    raw = story()
    first = await prepare(manager, raw)
    retriever.spans = (Span(1, 3), Span(5, 7))
    extended = raw + [
        {"role": "assistant", "content": "She hesitates."},
        {"role": "user", "content": "She remembers the promise."},
    ]
    second = await prepare(manager, extended)
    assert second.reason == "append"
    assert second.messages[: len(first.messages)] == first.messages
    assert second.snapshot.reminders[0] == first.snapshot.reminders[0]
    assert len(second.snapshot.reminders) == 2
    assert second.potential_cached_tokens > 0
    assert second.snapshot.epoch == first.snapshot.epoch


async def test_repeated_input_is_idempotent_even_if_retrieval_changes(manager, retriever):
    raw = story()
    first = await prepare(manager, raw)
    calls = retriever.calls
    retriever.spans = (Span(9, 10),)
    replay = await prepare(manager, raw)
    assert replay.messages == first.messages
    assert replay.snapshot == first.snapshot
    assert replay.reason == "replay"
    assert first.request_id != replay.request_id
    assert calls == retriever.calls


async def test_mutating_result_does_not_mutate_saved_snapshot(manager):
    raw = story()
    first = await prepare(manager, raw)
    expected = copy.deepcopy(first.messages)
    first.messages[-1]["content"] = "tampered"
    assert (await prepare(manager, raw)).messages == expected


async def test_old_branch_restores_identical_snapshot(manager):
    raw = story()
    original = await prepare(manager, raw)
    branch = copy.deepcopy(raw)
    branch[10]["content"] = "The castle has never existed."
    changed = await prepare(manager, branch)
    assert changed.snapshot.epoch != original.snapshot.epoch
    restored = await prepare(manager, raw)
    assert restored.messages == original.messages
    assert restored.snapshot.epoch == original.snapshot.epoch


async def test_fork_before_rebuild_cannot_reuse_future_excerpts(manager, retriever):
    raw = story(80)
    raw[130]["content"] = "FUTURE_SECRET_DRAGON"
    retriever.spans = (Span(130, 131),)
    future = await prepare(manager, raw)
    assert "FUTURE_SECRET_DRAGON" in str(future.messages)
    earlier = raw[:25] + [{"role": "user", "content": "A different path begins."}]
    branch = await prepare(manager, earlier)
    assert "FUTURE_SECRET_DRAGON" not in str(branch.messages)
    assert branch.snapshot.epoch != future.snapshot.epoch


async def test_edit_output_and_same_id_input_are_versions(manager):
    raw = story()
    raw[-1]["id"] = "fixed-id"
    first = await prepare(manager, raw)
    edited = copy.deepcopy(raw)
    edited[-1]["content"] = "She throws the sword away."
    assert (await prepare(manager, edited)).snapshot.head != first.snapshot.head
    response = raw + [
        {"role": "assistant", "content": "She draws it."},
        {"role": "user", "content": "Continue"},
    ]
    reply = await prepare(manager, response)
    response[-2]["content"] = "She cannot draw it."
    modified = await prepare(manager, response)
    assert reply.snapshot.head != modified.snapshot.head
    assert "She draws it." not in str(modified.messages)


async def test_state_survives_restart_and_isolates_users(tmp_path, manager, retriever):
    path = tmp_path / "state.sqlite"
    first_manager = ContextManager(manager.config, store=SQLiteStore(path), retriever=retriever)
    original = await prepare(first_manager, story())
    first_manager.store.close()
    restarted = ContextManager(manager.config, store=SQLiteStore(path), retriever=retriever)
    assert (await prepare(restarted, story())).snapshot == original.snapshot
    other = await restarted.prepare(story(), user_id="other", chat_id="chat")
    assert other.reason == "initial"
    assert other.snapshot.epoch != original.snapshot.epoch


async def test_environment_and_model_changes_separate_cache_domains(manager):
    raw = story()
    a = await prepare(manager, raw, model="a", environment={"tools": []})
    b = await prepare(manager, raw, model="b", environment={"tools": []})
    c = await prepare(manager, raw, model="a", environment={"tools": ["different"]})
    assert len({a.snapshot.epoch, b.snapshot.epoch, c.snapshot.epoch}) == 3
    assert b.potential_cached_tokens == c.potential_cached_tokens == 0


async def test_concurrent_same_path_is_first_writer_wins(manager):
    results = await asyncio.gather(*(prepare(manager, story()) for _ in range(8)))
    assert len({r.snapshot.serialize() for r in results}) == 1
    assert len({r.request_id for r in results}) == 8


async def test_capacity_rebuild_retains_current_turn(manager, retriever):
    raw = story()
    first = await prepare(manager, raw)
    size = manager.config.input_limit - first.estimated_tokens + 200
    raw += [{"role": "assistant", "content": "x" * size}, {"role": "user", "content": "continue"}]
    second = await prepare(manager, raw)
    assert second.reason == "capacity"
    assert second.snapshot.epoch != first.snapshot.epoch
    assert second.messages[-1]["content"] == "continue"
    assert second.estimated_tokens <= manager.config.input_limit


async def test_oversized_protected_turn_raises_without_saving(manager):
    raw = [{"role": "user", "content": "x" * 20000}]
    with pytest.raises(ContextOverflow):
        await prepare(manager, raw)
    scope = digest(["u", "chat"])
    manager.store.delete_scope(scope)


async def test_over_budget_candidates_report_omissions(manager, retriever):
    raw = story()
    raw[1]["content"] = "long " * 3000
    result = await prepare(manager, raw)
    assert result.omitted
    assert result.estimated_tokens <= manager.config.input_limit
    assert result.messages[-1] == raw[-1]


async def test_retriever_cannot_return_foreign_or_future_span(manager, retriever):
    class Bad:
        async def search(self, history, before):
            from lethes import Retrieval

            return Retrieval((Span(0, len(history) + 10),))

    manager.retriever = Bad()
    with pytest.raises(ValueError, match="outside visible"):
        await prepare(manager, story())


async def test_tool_exchange_is_not_split(manager):
    raw = story()
    raw += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "roll", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": "20"},
    ]
    result = await prepare(manager, raw)
    assert result.messages[-2:] == raw[-2:]
    assert (
        assemble(History.from_messages(raw), result.snapshot.tail_start, result.snapshot.reminders)
        == result.messages
    )


async def test_tool_followup_preserves_previous_request(manager, retriever):
    raw = story()
    first = await prepare(manager, raw)
    retriever.spans = (Span(5, 7),)
    tool = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c", "type": "function", "function": {"name": "roll", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c", "content": "1"},
    ]
    followup = await prepare(manager, raw + tool)
    assert followup.messages == first.messages + tool


async def test_hundreds_of_turns_with_periodic_edits_and_rebuilds():
    manager = ContextManager(
        ContextConfig(
            context_window=7000,
            reply_reserve=500,
            safety_reserve=200,
            recent_tokens=700,
            minimum_recent_turns=1,
        ),
        retriever=ScriptedRetriever((Span(1, 3),)),
    )
    raw = story(12)
    epochs = set()
    for i in range(220):
        raw += [
            {"role": "assistant", "content": f"The traveler sees milestone {i}."},
            {"role": "user", "content": "Keep walking, and remember the beginning."},
        ]
        if i and i % 55 == 0:
            raw[-8]["content"] = f"Edited milestone {i}"
        result = await prepare(manager, raw)
        epochs.add(result.snapshot.epoch)
        assert result.estimated_tokens <= manager.config.input_limit
        assert result.messages[-1] == raw[-1]
    assert len(epochs) > 3


async def test_replay_preserves_omission_diagnostics(manager):
    raw = story()
    raw[1]["content"] = "x" * 20000
    first = await prepare(manager, raw)
    second = await prepare(manager, raw)
    assert first.omitted and first.omitted == second.omitted


async def test_recent_target_larger_than_model_capacity_is_clamped():
    manager = ContextManager(
        ContextConfig(
            context_window=3000,
            reply_reserve=500,
            safety_reserve=100,
            recent_tokens=8192,
            minimum_recent_turns=2,
        ),
        retriever=ScriptedRetriever(),
    )
    result = await prepare(manager, story())
    assert result.estimated_tokens <= 2400


async def test_neighbor_expansion_can_fall_back_to_exact_hit(manager):
    from lethes import Retrieval

    class Focused:
        async def search(self, history, before):
            return Retrieval((Span(1, 4),), focuses=(2,))

    manager.retriever = Focused()
    raw = story()
    raw[1]["content"] = "x" * 20000
    raw[2]["content"] = "A small, crucial original passage."
    result = await prepare(manager, raw)
    assert "A small, crucial original passage." in str(result.messages)
    assert result.omitted
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
    followup = await prepare(manager, raw + tools)
    assert followup.messages == result.messages + tools


async def test_full_stable_prefix_rebuilds_to_make_room_for_current_required_recall(
    manager, retriever
):
    from lethes import Retrieval

    raw = story()
    first = await prepare(manager, raw)
    size = manager.config.input_limit - first.estimated_tokens - 300
    raw += [
        {"role": "assistant", "content": "p" * size},
        {"role": "user", "content": "A new topic needs an earlier passage."},
    ]

    class Required:
        async def search(self, history, before):
            span = Span(5, 7)
            return Retrieval((span,), required=(span,))

    manager.retriever = Required()
    second = await prepare(manager, raw)
    assert second.reason == "recall_capacity"
    assert second.snapshot.epoch != first.snapshot.epoch
    covered = {i for r in second.snapshot.reminders for s in r.spans for i in s.indices()}
    assert {5, 6} <= covered


async def test_resident_hit_does_not_refill_query_budget_with_weaker_candidates(manager, retriever):
    raw = story()
    raw[1]["content"] = "a" * 1800
    raw[3]["content"] = "b" * 1600
    retriever.spans = (Span(1, 2), Span(3, 4))
    first = await prepare(manager, raw)
    assert {i for r in first.snapshot.reminders for s in r.spans for i in s.indices()} == {1}
    for _ in range(5):
        raw += [
            {"role": "assistant", "content": "She nods."},
            {"role": "user", "content": "Continue the same scene."},
        ]
        current = await prepare(manager, raw)
        assert current.reason == "append"
        assert current.snapshot.reminders == first.snapshot.reminders
        assert current.messages[: len(first.messages)] == first.messages
        assert current.omitted == (Span(3, 4),)


async def test_overlapping_recall_and_recent_original_are_never_repeated(manager, retriever):
    raw = story()
    retriever.spans = (Span(1, 4), Span(2, 5), Span(3, 6))
    first = await prepare(manager, raw)
    raw += [{"role": "assistant", "content": "She nods."}, {"role": "user", "content": "Continue."}]
    retriever.spans = (Span(2, 5), Span(1, 4), Span(3, 7))
    current = await prepare(manager, raw)
    quoted = [i for r in current.snapshot.reminders for s in r.spans for i in s.indices()]
    assert len(quoted) == len(set(quoted))
    assert not set(quoted).intersection(range(current.snapshot.tail_start, len(raw)))
    assert current.messages[: len(first.messages)] == first.messages


async def test_recall_of_a_message_still_in_continuous_tail_is_not_requoted(manager, retriever):
    raw = story()
    retriever.spans = ()
    first = await prepare(manager, raw)
    recalled = first.snapshot.tail_start
    retriever.spans = (Span(recalled, recalled + 1),)
    raw += [
        {"role": "assistant", "content": "An extended answer. " * 15},
        {"role": "user", "content": "She keeps walking."},
    ]
    current = await prepare(manager, raw)
    assert current.snapshot.reminders == ()
    assert current.messages[: len(first.messages)] == first.messages
    assert current.omitted == ()


async def test_edit_of_previously_quoted_source_is_not_suppressed_as_already_seen(
    manager, retriever
):
    raw = story()
    raw[1]["id"] = "unchanged-id"
    raw[1]["content"] = "The gate only opens during rain."
    first = await prepare(manager, raw)
    edited = copy.deepcopy(raw)
    edited[1]["content"] = "The gate only opens at night."
    second = await prepare(manager, edited)
    assert "The gate only opens at night." in str(second.messages)
    assert "The gate only opens during rain." not in str(second.messages)
    assert second.snapshot.epoch != first.snapshot.epoch


async def test_tool_hop_does_not_run_retrieval_again(manager, retriever):
    raw = story()
    first = await prepare(manager, raw)
    calls = retriever.calls
    raw += [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "roll", "type": "function", "function": {"name": "roll", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "roll", "content": "20"},
    ]
    current = await prepare(manager, raw)
    assert retriever.calls == calls
    assert current.messages[: len(first.messages)] == first.messages


@pytest.mark.parametrize(
    "scores,required,expected",
    [
        ((1.0, 0.89), (), 1),
        ((1.0, 0.8), (), 0),
        ((0.01, 1.0), (Span(1, 2),), 0),
    ],
)
def test_resident_preference_is_bounded_and_cannot_override_required_match(
    scores, required, expected
):
    from lethes import Retrieval
    from lethes.engine import stable_recall_order

    search = Retrieval((Span(1, 2), Span(3, 4)), focuses=(1, 3), scores=scores, required=required)
    assert stable_recall_order(search, {3})[0] == expected


@pytest.mark.parametrize("scores", [(float("nan"),), (-1.0,), (1.0, 2.0)])
async def test_invalid_retrieval_scores_are_rejected(manager, scores):
    from lethes import Retrieval

    class Invalid:
        async def search(self, history, before):
            return Retrieval((Span(1, 2),), scores=scores)

    manager.retriever = Invalid()
    with pytest.raises(ValueError, match="scores"):
        await prepare(manager, story())


async def test_compatible_saved_layout_is_replayed_without_cold_reset():
    from dataclasses import asdict

    from lethes.history import canonical
    from lethes.state import Snapshot

    raw = [
        {"role": "system", "content": "Continue the story."},
        {"role": "user", "content": "She opens the door."},
    ]
    manager = ContextManager()
    history = History.from_messages(raw)
    # The earlier alpha's persisted domain contract and snapshot format remain valid.
    old_domain = digest(
        ["default", "default", {}, asdict(manager.config), None, manager.counter.cache_identity, 0]
    )
    legacy = Snapshot(
        history.head,
        len(history),
        "existing-cycle",
        1,
        (),
        tuple(canonical(m) for m in raw),
        manager.counter.count(raw),
        512,
    )
    manager.store.save(digest(["u", "chat"]), old_domain, legacy)
    result = await prepare(manager, raw)
    assert result.reason == "replay"
    assert result.snapshot.epoch == "existing-cycle"
    assert result.messages == raw
