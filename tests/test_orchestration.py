"""Tests for the improved orchestration logic."""

from __future__ import annotations

import dataclasses

import pytest

from lethes.algorithms.greedy import GreedyByWeightAlgorithm
from lethes.algorithms.recency import RecencyBiasedAlgorithm
from lethes.engine.constraints import ConstraintSet
from lethes.engine.orchestrator import ContextOrchestrator, _build_weighting_context
from lethes.flags.schema import WellKnownFlag
from lethes.models.budget import TokenBudget, TokenTargetBudget
from lethes.models.conversation import Conversation
from lethes.models.message import Message
from lethes.utils.tokens import TokenCounter
from lethes.weighting.smart import SmartWeightingStrategy


# ── SmartWeightingStrategy ────────────────────────────────────────────────────

def _make_conv(*contents, roles=None):
    if roles is None:
        roles = ["user", "assistant"] * 10
    msgs = [
        Message(role=roles[i % len(roles)], content=c, sequence_index=i)
        for i, c in enumerate(contents)
    ]
    return Conversation(msgs)


@pytest.mark.asyncio
async def test_smart_keyword_relevance():
    """High-keyword-match messages score higher than low-match ones (pair_coherence=0 to isolate)."""
    # Disable pair_coherence so scores reflect only keyword relevance
    conv = _make_conv("the weather in Paris today", "football results", "Paris weather forecast")
    strategy = SmartWeightingStrategy(pair_coherence=0.0, tool_penalty=1.0)
    scores = await strategy.score(list(conv.messages), "what is the weather in Paris", conv)
    msgs = list(conv.messages)
    # msg 0 and msg 2 (both contain "weather" + "Paris") should score higher than msg 1 (football)
    assert scores[msgs[0].id] > scores[msgs[1].id]
    assert scores[msgs[2].id] > scores[msgs[1].id]


@pytest.mark.asyncio
async def test_smart_tool_penalty_applied():
    """Tool call messages score lower than equivalent regular messages."""
    conv = Conversation.from_openai_messages([
        {"role": "user", "content": "weather in Paris"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "22°C"},
        {"role": "assistant", "content": "It is 22°C in Paris."},
    ])
    strategy = SmartWeightingStrategy(tool_penalty=0.3)
    scores = await strategy.score(list(conv.messages), "weather in Paris", conv)
    msgs = list(conv.messages)
    # assistant tool_calls message should be penalised vs the final assistant reply
    assert scores[msgs[1].id] < scores[msgs[3].id]
    # tool result also penalised
    assert scores[msgs[2].id] < scores[msgs[0].id]


@pytest.mark.asyncio
async def test_smart_pair_coherence_lifts_assistant_reply():
    """A highly relevant user message should lift its assistant reply's score."""
    # user[0] is very relevant (exact keyword match), assistant[1] is its reply
    # user[2] is irrelevant, assistant[3] is its reply
    conv = _make_conv(
        "explain Python decorators",  # user — very relevant
        "A decorator is a callable...",  # assistant reply — should be lifted
        "what time is it",  # user — irrelevant
        "It is 3pm",  # assistant — low score
    )
    strategy = SmartWeightingStrategy(pair_coherence=0.9, tool_penalty=1.0)
    query = "explain Python decorators in detail"
    scores = await strategy.score(list(conv.messages), query, conv)
    msgs = list(conv.messages)
    # assistant[1] (reply to relevant user) should score higher than assistant[3]
    assert scores[msgs[1].id] > scores[msgs[3].id]


@pytest.mark.asyncio
async def test_smart_context_overrides_tool_penalty():
    """context['tool_penalty'] passed by orchestrator overrides default."""
    conv = Conversation.from_openai_messages([
        {"role": "user", "content": "call a tool"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "fn", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
    ])
    strategy = SmartWeightingStrategy(tool_penalty=0.1)  # default very low
    msgs = list(conv.messages)
    query = "call a tool"
    # Override to 1.0 via context — tool message should now score the same as user
    scores_with_override = await strategy.score(
        msgs, query, conv, context={"tool_penalty": 1.0}
    )
    scores_default = await strategy.score(msgs, query, conv)
    # With override=1.0, tool message scores higher than with default=0.1
    tool_id = msgs[2].id
    assert scores_with_override[tool_id] > scores_default[tool_id]


@pytest.mark.asyncio
async def test_smart_no_query_returns_neutral_with_role_penalty():
    """Empty query returns neutral scores for regular messages, penalised for tool msgs."""
    conv = Conversation.from_openai_messages([
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "fn", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ])
    strategy = SmartWeightingStrategy(tool_penalty=0.5)
    msgs = list(conv.messages)
    scores = await strategy.score(msgs, "", conv)
    # Regular user message: no penalty
    assert scores[msgs[0].id] == 1.0
    # Tool messages: penalised
    assert scores[msgs[1].id] == pytest.approx(0.5)
    assert scores[msgs[2].id] == pytest.approx(0.5)


# ── TokenTargetBudget ─────────────────────────────────────────────────────────

def test_token_target_budget_headroom():
    b = TokenTargetBudget(target_tokens=1000, overshoot=100)
    assert b.headroom_tokens(0) == 1100
    assert b.headroom_tokens(800) == 300
    assert b.headroom_tokens(1100) == 0


def test_token_target_budget_exceeded():
    b = TokenTargetBudget(target_tokens=1000, overshoot=100)
    assert not b.is_exceeded(1100)
    assert b.is_exceeded(1101)


def test_token_target_budget_greedy_fills_to_target():
    """Greedy algorithm fills up to the target, keeping highest-weight messages."""
    msgs = [
        Message(role="user", content="x " * 30, weight=1.0, sequence_index=i)
        for i in range(10)
    ]
    conv = Conversation(msgs)
    budget = TokenTargetBudget(target_tokens=100, overshoot=50)
    algo = GreedyByWeightAlgorithm()
    result = algo.select(conv, budget, ConstraintSet(), TokenCounter())
    # Should keep some messages and drop others based on budget
    assert len(result.keep_full) > 0
    assert len(result.keep_full) < 10


# ── Flag: !recent=N ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flag_recent_pins_last_n_messages():
    """!recent=2 should pin the last 2 non-system messages."""
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "msg 1"},
        {"role": "assistant", "content": "reply 1"},
        {"role": "user", "content": "msg 2"},
        {"role": "assistant", "content": "reply 2"},
        {"role": "user", "content": "!recent=2 current question"},
    ]
    conv = Conversation.from_openai_messages(msgs)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=50))
    result = await orchestrator.process(conv)
    final_msgs = result.conversation.to_openai_messages()
    contents = [m.get("content") or "" for m in final_msgs]
    # The last 2 non-system messages before current should be pinned (reply 2 + current)
    assert "current question" in " ".join(c for c in contents if c)


@pytest.mark.asyncio
async def test_flag_recent_zero_pins_nothing_extra():
    """!recent=0 should not pin any messages (no-op)."""
    msgs = [
        {"role": "user", "content": "old message " * 20},
        {"role": "assistant", "content": "old reply " * 20},
        {"role": "user", "content": "!recent=0 new question"},
    ]
    conv = Conversation.from_openai_messages(msgs)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=20))
    result = await orchestrator.process(conv)
    # Just verify it runs without error
    assert result.conversation is not None


# ── Flag: !keep_tag=label ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flag_keep_tag_pins_tagged_messages():
    """Messages tagged with !+tag=important are pinned when !keep_tag=important is used."""
    msgs = [
        {"role": "user", "content": "!+tag=important critical context"},
        {"role": "assistant", "content": "understood"},
        {"role": "user", "content": "unrelated topic " * 20},
        {"role": "assistant", "content": "sure " * 20},
        {"role": "user", "content": "!keep_tag=important new question"},
    ]
    conv = Conversation.from_openai_messages(msgs)
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=40))
    result = await orchestrator.process(conv)
    final_msgs = result.conversation.to_openai_messages()
    contents = " ".join(str(m.get("content") or "") for m in final_msgs)
    # The tagged message should be kept
    assert "critical context" in contents


# ── Flag: !target=N ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_flag_target_sets_token_target_budget():
    """!target=N should override the default budget with TokenTargetBudget."""
    msgs = [
        {"role": "user", "content": "question one " * 5},
        {"role": "assistant", "content": "answer one " * 5},
        {"role": "user", "content": "question two " * 5},
        {"role": "assistant", "content": "answer two " * 5},
        {"role": "user", "content": "!target=10 final question"},
    ]
    conv = Conversation.from_openai_messages(msgs)
    # Default budget is unlimited (0); target flag should override to ~10 tokens
    orchestrator = ContextOrchestrator(budget=TokenBudget(max_tokens=0))
    result = await orchestrator.process(conv)
    # With a tight target, something should be dropped
    assert result.token_count <= 10 + 150 + 50  # target + overshoot + system overhead


# ── Flag: !tool_penalty=F ─────────────────────────────────────────────────────

def test_build_weighting_context_extracts_tool_penalty():
    flags = {
        str(WellKnownFlag.TOOL_PENALTY): "0.2",
        str(WellKnownFlag.PAIR_COHERENCE): "0.9",
    }
    ctx = _build_weighting_context(flags)
    assert ctx["tool_penalty"] == pytest.approx(0.2)
    assert ctx["pair_coherence"] == pytest.approx(0.9)


def test_build_weighting_context_empty_flags():
    ctx = _build_weighting_context({})
    assert ctx == {}


def test_build_weighting_context_invalid_value_skipped():
    flags = {str(WellKnownFlag.TOOL_PENALTY): "not_a_number"}
    ctx = _build_weighting_context(flags)
    assert "tool_penalty" not in ctx


# ── RecencyBiasedAlgorithm default factor ─────────────────────────────────────

def test_recency_biased_newer_messages_have_higher_effective_weight():
    """Newer messages should rank higher under recency bias."""
    msgs = [
        Message(role="user", content="old " * 5, weight=1.0, sequence_index=i)
        for i in range(5)
    ]
    # All have equal base weight — recency alone should keep the newest
    conv = Conversation(msgs)
    algo = RecencyBiasedAlgorithm(recency_factor=10.0)  # strong bias
    budget = TokenBudget(max_tokens=30)
    result = algo.select(conv, budget, ConstraintSet(), TokenCounter())
    kept = set(result.keep_full)
    # The last message (index 4) should always be kept under strong recency
    assert msgs[-1].id in kept
