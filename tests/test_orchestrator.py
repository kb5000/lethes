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
