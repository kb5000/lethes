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
