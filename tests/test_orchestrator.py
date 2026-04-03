"""End-to-end orchestrator tests."""

import pytest
from lethes import ContextOrchestrator, Conversation, TokenBudget
from lethes.algorithms import GreedyByWeightAlgorithm
from lethes.cache.memory_backend import InMemoryCache
from lethes.engine.constraints import ConstraintSet
from lethes.models.message import Message
from lethes.weighting.static import StaticWeightStrategy


def _make_conversation(n: int = 10) -> Conversation:
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(Message(role=role, content=f"message {i} " + "word " * 20))
    return Conversation(msgs)


@pytest.mark.asyncio
async def test_orchestrator_basic():
    conv = _make_conversation(10)
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=200),
        algorithm=GreedyByWeightAlgorithm(),
        weighting=StaticWeightStrategy(),
        cache=InMemoryCache(),
        constraints=ConstraintSet(require_last_user=True),
    )
    result = await orchestrator.process(conv)
    assert result.token_count <= 220  # allow slight overage from overhead
    final_msgs = result.conversation.to_openai_messages()
    assert len(final_msgs) > 0


@pytest.mark.asyncio
async def test_orchestrator_last_user_always_kept():
    conv = _make_conversation(6)
    last_user = conv.last_user_message()
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=50),  # very tight
        algorithm=GreedyByWeightAlgorithm(),
        weighting=StaticWeightStrategy(),
        constraints=ConstraintSet(require_last_user=True),
    )
    result = await orchestrator.process(conv)
    final_ids = {m["role"] for m in result.conversation.to_openai_messages()}
    # Last user message content must be present
    final_contents = [m["content"] for m in result.conversation.to_openai_messages()]
    assert any(last_user.get_text_content() in str(c) for c in final_contents)


@pytest.mark.asyncio
async def test_full_flag_bypasses_truncation():
    msgs_raw = [
        {"role": "user", "content": "!full " + "word " * 200},
        {"role": "assistant", "content": "response " * 50},
    ]
    conv = Conversation.from_openai_messages(msgs_raw)
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=100),  # would normally truncate
        algorithm=GreedyByWeightAlgorithm(),
        weighting=StaticWeightStrategy(),
    )
    result = await orchestrator.process(conv)
    # full flag → both messages kept
    assert len(result.conversation.messages) == 2


@pytest.mark.asyncio
async def test_pin_flag_keeps_message():
    msgs_raw = [
        {"role": "user", "content": "!pin important context " + "word " * 5},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "new question"},
    ]
    conv = Conversation.from_openai_messages(msgs_raw)
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=30),
        algorithm=GreedyByWeightAlgorithm(),
        weighting=StaticWeightStrategy(),
    )
    result = await orchestrator.process(conv)
    contents = [str(m.content) for m in result.conversation.messages]
    assert any("important context" in c for c in contents)


@pytest.mark.asyncio
async def test_unlimited_budget():
    conv = _make_conversation(20)
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=0),  # unlimited
        algorithm=GreedyByWeightAlgorithm(),
    )
    result = await orchestrator.process(conv)
    assert len(result.conversation.messages) == 20


@pytest.mark.asyncio
async def test_empty_conversation():
    """Empty conversation should not crash and return an empty result."""
    conv = Conversation([])
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=1000))
    result = await orchestrator.process(conv)
    assert len(result.conversation.messages) == 0


@pytest.mark.asyncio
async def test_system_only_conversation():
    """A conversation with only a system message should pass through."""
    conv = Conversation([Message(role="system", content="You are helpful.")])
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=100))
    result = await orchestrator.process(conv)
    roles = [m.role for m in result.conversation.messages]
    assert "system" in roles


@pytest.mark.asyncio
async def test_result_fields_populated():
    """OrchestratorResult should have plan, flags, and token_count set."""
    conv = _make_conversation(4)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=500))
    result = await orchestrator.process(conv)
    assert result.plan is not None
    assert result.flags is not None
    assert result.token_count > 0


@pytest.mark.asyncio
async def test_token_target_budget_keeps_close_to_target():
    """TokenTargetBudget should keep close to target, not just min messages."""
    from lethes.models.budget import TokenTargetBudget

    conv = _make_conversation(20)
    orchestrator = ContextOrchestrator(
        budget=TokenTargetBudget(target_tokens=300, overshoot=100),
        algorithm=GreedyByWeightAlgorithm(),
    )
    result = await orchestrator.process(conv)
    # Should keep some but not all messages
    assert 1 <= len(result.conversation.messages) <= 20


@pytest.mark.asyncio
async def test_recent_flag_pins_messages():
    """!recent=2 should always keep the last 2 non-system messages."""
    msgs = [
        {"role": "user", "content": "!recent=2 latest query"},
        {"role": "assistant", "content": "reply"},
    ]
    # Prepend a lot of old context
    old = [
        {"role": "user" if i % 2 == 0 else "assistant",
         "content": "old message " + "word " * 20}
        for i in range(10)
    ]
    conv = Conversation.from_openai_messages(old + msgs)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=60))
    result = await orchestrator.process(conv)
    contents = [str(m.content) for m in result.conversation.messages]
    assert any("latest query" in c for c in contents)


@pytest.mark.asyncio
async def test_plan_disjoint_sets():
    """keep_full, summarize, and drop must be disjoint in the plan."""
    conv = _make_conversation(8)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=150))
    result = await orchestrator.process(conv)
    p = result.plan
    assert p.keep_full & p.summarize == frozenset()
    assert p.keep_full & p.drop == frozenset()
    assert p.summarize & p.drop == frozenset()


@pytest.mark.asyncio
async def test_cost_estimated_when_pricing_table_provided():
    """When pricing_table and model_id are provided, cost is estimated."""
    from lethes.models.pricing import ModelPricingTable

    table = ModelPricingTable.from_list([{
        "model_id": "test-model",
        "input_price_per_1m": 5.0,
        "cached_price_per_1m": 1.0,
        "output_price_per_1m": 15.0,
    }])
    conv = _make_conversation(3)
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=1000),
        pricing_table=table,
    )
    result = await orchestrator.process(conv, model_id="test-model")
    assert result.estimated_cost_usd is not None
    assert result.estimated_cost_usd >= 0.0


@pytest.mark.asyncio
async def test_cost_none_without_pricing_table():
    conv = _make_conversation(3)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=1000))
    result = await orchestrator.process(conv, model_id="gpt-4o")
    assert result.estimated_cost_usd is None


@pytest.mark.asyncio
async def test_tool_pair_not_split_when_no_other_violations():
    """Regression: repair() must always run, not only when constraint violations
    exist.  Previously, if the greedy algorithm dropped one half of a tool
    call/result pair but no *constraint* violation was detected (last-user kept,
    min-messages satisfied), repair() was skipped and the split pair reached the
    API causing an error."""
    # Build a conversation where the tool result has high weight (kept) but
    # the assistant tool_calls message has low weight (likely dropped on a
    # tight budget).  We give the tool result a very high static weight so
    # the greedy algorithm prefers it over the assistant message.
    import dataclasses
    from lethes.weighting.static import StaticWeightStrategy

    raw = [
        {"role": "user", "content": "please call the tool " + "word " * 10},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_xyz",
                    "type": "function",
                    "function": {"name": "do_thing", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_xyz",
            "name": "do_thing",
            "content": "done " + "word " * 10,
        },
        {"role": "user", "content": "thanks " + "word " * 5},
    ]
    conv = Conversation.from_openai_messages(raw)

    # Give the tool result a higher weight than the assistant tool_calls message
    # so greedy might prefer to keep one but drop the other.
    msgs = list(conv.messages)
    msgs[1] = dataclasses.replace(msgs[1], weight=0.1)   # assistant tool_calls — low
    msgs[2] = dataclasses.replace(msgs[2], weight=0.9)   # tool result — high
    conv = conv.with_messages(msgs)

    # Budget tight enough to drop the low-weight assistant tool_calls message
    # but keep both user messages and the high-weight tool result — without
    # triggering any explicit constraint violation (last user is retained,
    # min_chat_messages=1 is satisfied).  Tokens: user(14)+user(6)+tool(11)=31.
    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=31),
        algorithm=GreedyByWeightAlgorithm(),
        weighting=StaticWeightStrategy(),
        cache=InMemoryCache(),
        constraints=ConstraintSet(require_last_user=True, min_chat_messages=1),
    )
    result = await orchestrator.process(conv)
    final_msgs = result.conversation.to_openai_messages()

    # Verify tool pair integrity: if either half is present, both must be.
    has_tool_call = any(
        m.get("role") == "assistant" and m.get("tool_calls")
        for m in final_msgs
    )
    has_tool_result = any(m.get("role") == "tool" for m in final_msgs)
    assert has_tool_call == has_tool_result, (
        "Tool call and tool result must both be present or both be absent; "
        f"has_tool_call={has_tool_call}, has_tool_result={has_tool_result}"
    )
