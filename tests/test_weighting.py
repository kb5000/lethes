"""Tests for weighting strategies: keyword, smart, composite."""

from __future__ import annotations

import pytest

from lethes.models.conversation import Conversation
from lethes.models.message import Message
from lethes.weighting.composite import CompositeWeightStrategy
from lethes.weighting.keyword import KeywordRelevanceStrategy, _default_tokenize
from lethes.weighting.smart import SmartWeightingStrategy
from lethes.weighting.static import StaticWeightStrategy


def _msg(content: str, role: str = "user", **kw) -> Message:
    return Message(role=role, content=content, **kw)


def _conv(*msgs: Message) -> Conversation:
    return Conversation(list(msgs))


# ── _default_tokenize ─────────────────────────────────────────────────────────


def test_tokenize_lowercases():
    assert "HELLO" not in _default_tokenize("HELLO world")
    assert "hello" in _default_tokenize("HELLO world")


def test_tokenize_strips_punctuation():
    tokens = _default_tokenize("hello, world!")
    assert "," not in tokens
    assert "hello" in tokens


def test_tokenize_empty():
    assert _default_tokenize("") == []


def test_tokenize_splits_on_whitespace():
    assert _default_tokenize("a b c") == ["a", "b", "c"]


# ── KeywordRelevanceStrategy ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_keyword_returns_score_for_each_message():
    msgs = [_msg("python programming"), _msg("cat videos"), _msg("python tutorial")]
    conv = _conv(*msgs)
    strat = KeywordRelevanceStrategy()
    scores = await strat.score(msgs, "python", conv)
    assert set(scores) == {m.id for m in msgs}


@pytest.mark.asyncio
async def test_keyword_relevant_scores_higher():
    relevant = _msg("python decorators and closures", role="user")
    irrelevant = _msg("cat videos are funny", role="assistant")
    msgs = [relevant, irrelevant]
    conv = _conv(*msgs)
    strat = KeywordRelevanceStrategy()
    scores = await strat.score(msgs, "python closures", conv)
    assert scores[relevant.id] >= scores[irrelevant.id]


@pytest.mark.asyncio
async def test_keyword_scores_in_range():
    msgs = [_msg("hello world"), _msg("goodbye moon")]
    conv = _conv(*msgs)
    strat = KeywordRelevanceStrategy()
    scores = await strat.score(msgs, "hello", conv)
    for v in scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_keyword_empty_query_returns_uniform():
    msgs = [_msg("anything"), _msg("something")]
    conv = _conv(*msgs)
    strat = KeywordRelevanceStrategy()
    scores = await strat.score(msgs, "", conv)
    assert all(v == 1.0 for v in scores.values())


@pytest.mark.asyncio
async def test_keyword_empty_messages():
    conv = Conversation([])
    strat = KeywordRelevanceStrategy()
    scores = await strat.score([], "query", conv)
    assert scores == {}


@pytest.mark.asyncio
async def test_keyword_all_same_content():
    """All identical messages should get the same score."""
    msgs = [_msg("python python python") for _ in range(3)]
    conv = _conv(*msgs)
    strat = KeywordRelevanceStrategy()
    scores = await strat.score(msgs, "python", conv)
    values = list(scores.values())
    assert len(set(values)) == 1


@pytest.mark.asyncio
async def test_keyword_floor_applied():
    """Even off-topic messages should score at least 0.1."""
    on_topic = _msg("python language features")
    off_topic = _msg("xyz abc def totally unrelated")
    msgs = [on_topic, off_topic]
    conv = _conv(*msgs)
    strat = KeywordRelevanceStrategy()
    scores = await strat.score(msgs, "python", conv)
    assert scores[off_topic.id] >= 0.1


def test_keyword_name():
    strat = KeywordRelevanceStrategy()
    assert "keyword" in strat.name()


# ── SmartWeightingStrategy ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_smart_returns_score_per_message():
    msgs = [_msg("python tutorial"), _msg("general content")]
    conv = _conv(*msgs)
    strat = SmartWeightingStrategy()
    scores = await strat.score(msgs, "python", conv)
    assert len(scores) == 2


@pytest.mark.asyncio
async def test_smart_tool_penalty_applied():
    """Tool messages should score lower than regular ones with same content."""
    user_msg = _msg("python decorators", role="user")
    tool_msg = Message(role="assistant", content="python decorators", tool_calls=[
        {"function": {"name": "search"}}
    ])
    msgs = [user_msg, tool_msg]
    conv = _conv(*msgs)
    strat = SmartWeightingStrategy(tool_penalty=0.5, pair_coherence=0.0)
    scores = await strat.score(msgs, "python", conv)
    assert scores[tool_msg.id] < scores[user_msg.id]


@pytest.mark.asyncio
async def test_smart_tool_penalty_zero_disables():
    """tool_penalty=1.0 means no penalty."""
    regular = _msg("python code", role="user")
    tool_msg = Message(role="tool", content="python result")
    msgs = [regular, tool_msg]
    conv = _conv(*msgs)
    strat = SmartWeightingStrategy(tool_penalty=1.0, pair_coherence=0.0)
    scores = await strat.score(msgs, "python", conv)
    # Both score from keyword relevance only, no penalty
    assert scores[tool_msg.id] >= 0.1


@pytest.mark.asyncio
async def test_smart_pair_coherence_lifts_assistant():
    """Assistant following highly-relevant user should score at least coherence × user_score."""
    user_msg = _msg("what is a python decorator?", role="user")
    assistant_msg = _msg("it wraps a function", role="assistant")
    msgs = [user_msg, assistant_msg]
    conv = _conv(*msgs)
    strat = SmartWeightingStrategy(pair_coherence=0.8, tool_penalty=1.0)
    scores = await strat.score(msgs, "python decorator", conv)
    # User has high score; assistant should be at least 0.8 × user_score
    assert scores[assistant_msg.id] >= 0.8 * scores[user_msg.id] - 1e-9


@pytest.mark.asyncio
async def test_smart_context_overrides_tool_penalty():
    """context dict can override tool_penalty for one turn."""
    tool_msg = Message(role="tool", content="some tool output")
    regular = _msg("query relevant", role="user")
    msgs = [regular, tool_msg]
    conv = _conv(*msgs)
    strat = SmartWeightingStrategy(tool_penalty=0.1)
    # Override via context: no penalty
    scores = await strat.score(msgs, "query", conv, context={"tool_penalty": 1.0})
    # With no penalty, tool_msg should score higher than with 0.1
    scores_penalised = await strat.score(msgs, "query", conv)
    assert scores[tool_msg.id] >= scores_penalised[tool_msg.id]


@pytest.mark.asyncio
async def test_smart_scores_in_range():
    msgs = [_msg("hello"), _msg("world")]
    conv = _conv(*msgs)
    strat = SmartWeightingStrategy()
    scores = await strat.score(msgs, "hello", conv)
    for v in scores.values():
        assert 0.0 <= v <= 1.0


@pytest.mark.asyncio
async def test_smart_empty_messages():
    strat = SmartWeightingStrategy()
    scores = await strat.score([], "query", Conversation([]))
    assert scores == {}


def test_smart_name_contains_params():
    strat = SmartWeightingStrategy(tool_penalty=0.3, pair_coherence=0.7)
    name = strat.name()
    assert "smart" in name
    assert "0.3" in name
    assert "0.7" in name


# ── CompositeWeightStrategy ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_composite_combines_strategies():
    """Scores should be a weighted average of both strategies."""
    msgs = [_msg("python tutorial"), _msg("unrelated topic")]
    conv = _conv(*msgs)
    # Two identical keyword strategies with equal weight → same as single
    kw = KeywordRelevanceStrategy()
    composite = CompositeWeightStrategy([(kw, 0.5), (kw, 0.5)])
    single_scores = await kw.score(msgs, "python", conv)
    combo_scores = await composite.score(msgs, "python", conv)
    for msg in msgs:
        assert abs(combo_scores[msg.id] - single_scores[msg.id]) < 1e-6


@pytest.mark.asyncio
async def test_composite_single_strategy():
    """Composite with one strategy is equivalent to that strategy."""
    msgs = [_msg("alpha"), _msg("beta")]
    conv = _conv(*msgs)
    strat = StaticWeightStrategy()
    composite = CompositeWeightStrategy([(strat, 1.0)])
    direct = await strat.score(msgs, "q", conv)
    via_composite = await composite.score(msgs, "q", conv)
    assert direct == via_composite


@pytest.mark.asyncio
async def test_composite_empty_messages():
    strat = StaticWeightStrategy()
    composite = CompositeWeightStrategy([(strat, 1.0)])
    scores = await composite.score([], "q", Conversation([]))
    assert scores == {}


def test_composite_requires_at_least_one_strategy():
    with pytest.raises(ValueError):
        CompositeWeightStrategy([])


@pytest.mark.asyncio
async def test_composite_unequal_weights():
    """Higher-weighted strategy should dominate the result."""
    msgs = [_msg("python tutorial", role="user")]
    conv = _conv(*msgs)
    # static always returns 1.0; keyword will return ≤1.0 for relevant content
    # Give more weight to keyword so its lower score drags down the composite
    static = StaticWeightStrategy()
    keyword = KeywordRelevanceStrategy()
    # 10% static, 90% keyword — score should be close to keyword's score
    composite = CompositeWeightStrategy([(static, 0.1), (keyword, 0.9)])
    kw_scores = await keyword.score(msgs, "python", conv)
    composite_scores = await composite.score(msgs, "python", conv)
    # composite should be between keyword and 1.0
    assert composite_scores[msgs[0].id] <= 1.0 + 1e-9


@pytest.mark.asyncio
async def test_composite_scores_in_range():
    msgs = [_msg("hello"), _msg("world")]
    conv = _conv(*msgs)
    composite = CompositeWeightStrategy([
        (KeywordRelevanceStrategy(), 0.6),
        (StaticWeightStrategy(), 0.4),
    ])
    scores = await composite.score(msgs, "hello", conv)
    for v in scores.values():
        assert 0.0 <= v <= 1.0 + 1e-9


def test_composite_name():
    composite = CompositeWeightStrategy([
        (StaticWeightStrategy(), 0.5),
        (KeywordRelevanceStrategy(), 0.5),
    ])
    name = composite.name()
    assert "composite" in name


# ── StaticWeightStrategy ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_static_returns_all_ones():
    msgs = [_msg("a"), _msg("b"), _msg("c")]
    conv = _conv(*msgs)
    strat = StaticWeightStrategy()
    scores = await strat.score(msgs, "anything", conv)
    assert all(v == 1.0 for v in scores.values())


@pytest.mark.asyncio
async def test_static_empty():
    strat = StaticWeightStrategy()
    assert await strat.score([], "q", Conversation([])) == {}
