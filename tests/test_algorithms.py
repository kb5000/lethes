"""Tests for selection algorithms."""

import pytest
from lethes.algorithms.dependency import DependencyAwareAlgorithm
from lethes.algorithms.greedy import GreedyByWeightAlgorithm
from lethes.algorithms.prefix_cache import PrefixCacheOptimizedAlgorithm
from lethes.algorithms.recency import RecencyBiasedAlgorithm
from lethes.cache.memory_backend import InMemoryCache
from lethes.cache.prefix_tracker import PrefixSequenceTracker
from lethes.engine.constraints import ConstraintSet
from lethes.models.budget import TokenBudget
from lethes.models.conversation import Conversation
from lethes.models.message import Message
from lethes.utils.tokens import TokenCounter


def _make_conv(*contents: str, roles=None) -> Conversation:
    if roles is None:
        roles = ["user", "assistant"] * (len(contents) // 2 + 1)
    msgs = [
        Message(role=roles[i % len(roles)], content=c, sequence_index=i)
        for i, c in enumerate(contents)
    ]
    return Conversation(msgs)


def _counter() -> TokenCounter:
    return TokenCounter()


def _constraints() -> ConstraintSet:
    return ConstraintSet()


def test_greedy_keeps_all_within_budget():
    conv = _make_conv("short", "also short")
    budget = TokenBudget(max_tokens=1000)
    algo = GreedyByWeightAlgorithm()
    result = algo.select(conv, budget, _constraints(), _counter())
    assert len(result.keep_full) == 2
    assert len(result.drop) == 0


def test_greedy_drops_low_weight_on_overflow():
    # One very long message, one short one with higher weight.
    # Without a pre-computed summary, the overflow message goes to summarize
    # (so TurnSummarizer can generate one) rather than straight to drop.
    long_msg = Message(role="user", content="word " * 500, weight=0.1, sequence_index=0)
    short_msg = Message(role="assistant", content="hi", weight=10.0, sequence_index=1)
    conv = Conversation([long_msg, short_msg])
    budget = TokenBudget(max_tokens=50)
    algo = GreedyByWeightAlgorithm()
    result = algo.select(conv, budget, _constraints(), _counter())
    assert short_msg.id in result.keep_full
    assert long_msg.id in result.summarize
    assert long_msg.id not in result.drop


def test_greedy_drops_on_overflow_when_summarize_disabled():
    long_msg = Message(role="user", content="word " * 500, weight=0.1, sequence_index=0)
    short_msg = Message(role="assistant", content="hi", weight=10.0, sequence_index=1)
    conv = Conversation([long_msg, short_msg])
    budget = TokenBudget(max_tokens=50)
    algo = GreedyByWeightAlgorithm(prefer_summarize=False)
    result = algo.select(conv, budget, _constraints(), _counter())
    assert short_msg.id in result.keep_full
    assert long_msg.id in result.drop


def test_pinned_message_never_dropped():
    long_msg = Message(role="user", content="word " * 500, pinned=True, sequence_index=0)
    conv = Conversation([long_msg])
    budget = TokenBudget(max_tokens=10)
    algo = GreedyByWeightAlgorithm()
    result = algo.select(conv, budget, _constraints(), _counter())
    assert long_msg.id in result.keep_full
    assert long_msg.id not in result.drop


def test_recency_bias_keeps_latest():
    msgs = [
        Message(role="user", content="word " * 100, weight=1.0, sequence_index=i)
        for i in range(5)
    ]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=150)
    algo = RecencyBiasedAlgorithm(recency_factor=5.0)
    result = algo.select(conv, budget, _constraints(), _counter())
    # Latest message (index 4) should be kept
    assert msgs[-1].id in result.keep_full


def test_dependency_aware_promotes_dependency():
    dep_msg = Message(role="user", content="dep context", sequence_index=0)
    main_msg = Message(
        role="assistant",
        content="answer",
        dependencies=[dep_msg.id],
        weight=10.0,
        sequence_index=1,
    )
    conv = Conversation([dep_msg, main_msg])
    budget = TokenBudget(max_tokens=20)  # Only enough for main_msg
    inner = GreedyByWeightAlgorithm()
    algo = DependencyAwareAlgorithm(inner)
    result = algo.select(conv, budget, _constraints(), _counter())
    # Both should be kept (dep promoted)
    assert dep_msg.id not in result.drop


# ── Greedy edge cases ─────────────────────────────────────────────────────────


def test_greedy_empty_conversation():
    conv = Conversation([])
    budget = TokenBudget(max_tokens=1000)
    result = GreedyByWeightAlgorithm().select(conv, budget, _constraints(), _counter())
    assert result.keep_full == []
    assert result.drop == []


def test_greedy_system_messages_never_in_result():
    """System messages are excluded from algo input; they never appear in drop."""
    sys_msg = Message(role="system", content="you are helpful", sequence_index=0)
    user_msg = Message(role="user", content="hi", sequence_index=1)
    conv = Conversation([sys_msg, user_msg])
    budget = TokenBudget(max_tokens=5)
    result = GreedyByWeightAlgorithm().select(conv, budget, _constraints(), _counter())
    assert sys_msg.id not in result.drop
    assert sys_msg.id not in result.summarize


def test_greedy_prefer_summarize_false():
    """When prefer_summarize=False, overflow goes directly to drop."""
    msgs = [
        Message(role="user", content="word " * 200, weight=1.0, sequence_index=0),
        Message(role="assistant", content="word " * 200, weight=1.0, sequence_index=1),
        Message(role="user", content="short", weight=2.0, sequence_index=2),
    ]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=50)
    result = GreedyByWeightAlgorithm(prefer_summarize=False).select(
        conv, budget, _constraints(), _counter()
    )
    # Nothing goes to summarize
    assert len(result.summarize) == 0


def test_greedy_all_messages_fit():
    msgs = [
        Message(role="user", content="hi", sequence_index=i)
        for i in range(5)
    ]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=10000)
    result = GreedyByWeightAlgorithm().select(conv, budget, _constraints(), _counter())
    assert set(result.keep_full) == {m.id for m in msgs}
    assert result.drop == []


# ── Recency edge cases ────────────────────────────────────────────────────────


def test_recency_single_message():
    """Single-message conversation should not crash (division by zero guard)."""
    msgs = [Message(role="user", content="hello", weight=1.0, sequence_index=0)]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=1000)
    algo = RecencyBiasedAlgorithm(recency_factor=2.0)
    result = algo.select(conv, budget, _constraints(), _counter())
    assert msgs[0].id in result.keep_full


def test_recency_factor_zero_equals_greedy():
    """recency_factor=0 should behave identically to plain greedy."""
    msgs = [
        Message(role="user", content="word " * 100, weight=1.0, sequence_index=i)
        for i in range(4)
    ]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=300)
    greedy_result = GreedyByWeightAlgorithm().select(conv, budget, _constraints(), _counter())
    recency_result = RecencyBiasedAlgorithm(recency_factor=0.0).select(
        conv, budget, _constraints(), _counter()
    )
    assert set(greedy_result.keep_full) == set(recency_result.keep_full)


def test_recency_name():
    algo = RecencyBiasedAlgorithm(recency_factor=1.5)
    assert "recency_biased" in algo.name()
    assert "1.5" in algo.name()


# ── DependencyAwareAlgorithm edge cases ───────────────────────────────────────


def test_dependency_aware_no_dependencies():
    """When there are no dependencies, result should match the inner algo."""
    msgs = [
        Message(role="user", content="word " * 100, weight=1.0, sequence_index=0),
        Message(role="assistant", content="short reply", weight=5.0, sequence_index=1),
    ]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=30)
    inner = GreedyByWeightAlgorithm()
    algo = DependencyAwareAlgorithm(inner)
    inner_result = inner.select(conv, budget, _constraints(), _counter())
    aware_result = algo.select(conv, budget, _constraints(), _counter())
    assert set(inner_result.keep_full) == set(aware_result.keep_full)


def test_dependency_aware_name():
    algo = DependencyAwareAlgorithm(GreedyByWeightAlgorithm())
    assert "dependency_aware" in algo.name()
    assert "greedy" in algo.name()


def test_dependency_aware_chain():
    """Dependency chains: A → B → C should all be kept when C is kept."""
    msg_a = Message(role="user", content="context a", sequence_index=0)
    msg_b = Message(
        role="assistant", content="context b", dependencies=[msg_a.id], sequence_index=1
    )
    msg_c = Message(
        role="user", content="query c", dependencies=[msg_b.id], weight=10.0, sequence_index=2
    )
    conv = Conversation([msg_a, msg_b, msg_c])
    budget = TokenBudget(max_tokens=25)  # barely fits msg_c
    algo = DependencyAwareAlgorithm(GreedyByWeightAlgorithm())
    result = algo.select(conv, budget, _constraints(), _counter())
    assert msg_a.id not in result.drop
    assert msg_b.id not in result.drop
    assert msg_c.id in result.keep_full


# ── PrefixCacheOptimizedAlgorithm ─────────────────────────────────────────────


def _make_tracker_and_session() -> tuple[PrefixSequenceTracker, str]:
    cache = InMemoryCache()
    tracker = PrefixSequenceTracker(cache)
    return tracker, "test_session"


def test_prefix_cache_fallback_to_greedy_when_no_prefix():
    """No prior sequence → falls back to greedy (same result)."""
    tracker, sid = _make_tracker_and_session()
    msgs = [
        Message(role="user", content="word " * 100, weight=1.0, sequence_index=0),
        Message(role="assistant", content="short", weight=5.0, sequence_index=1),
    ]
    conv = Conversation(msgs)
    budget = TokenBudget(max_tokens=30)
    algo = PrefixCacheOptimizedAlgorithm(tracker, sid)
    greedy = GreedyByWeightAlgorithm()
    prefix_result = algo.select(conv, budget, _constraints(), _counter())
    greedy_result = greedy.select(conv, budget, _constraints(), _counter())
    assert set(prefix_result.keep_full) == set(greedy_result.keep_full)


@pytest.mark.asyncio
async def test_prefix_cache_pins_known_prefix():
    """Messages that form the previous prefix should be pinned (kept)."""
    tracker, sid = _make_tracker_and_session()

    # Record a prior sequence
    msg1 = Message(role="user", content="first message", sequence_index=0)
    msg2 = Message(role="assistant", content="first response", sequence_index=1)
    await tracker.prepare(sid)
    await tracker.record(sid, [msg1.id, msg2.id])

    # Current conversation = same two messages + a new one (tight budget)
    msg3 = Message(role="user", content="word " * 200, weight=0.01, sequence_index=2)
    conv = Conversation([msg1, msg2, msg3])
    budget = TokenBudget(max_tokens=50)

    algo = PrefixCacheOptimizedAlgorithm(tracker, sid)
    result = algo.select(conv, budget, _constraints(), _counter())

    # msg1 and msg2 form the known prefix — they should be kept
    assert msg1.id in result.keep_full
    assert msg2.id in result.keep_full


def test_prefix_cache_name():
    tracker, sid = _make_tracker_and_session()
    algo = PrefixCacheOptimizedAlgorithm(tracker, sid)
    assert algo.name() == "prefix_cache_optimized"
