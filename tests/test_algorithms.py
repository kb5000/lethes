"""Tests for selection algorithms."""

import pytest
from lethes.algorithms.greedy import GreedyByWeightAlgorithm
from lethes.algorithms.recency import RecencyBiasedAlgorithm
from lethes.algorithms.dependency import DependencyAwareAlgorithm
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
    # One very long message, one short one with higher weight
    long_msg = Message(role="user", content="word " * 500, weight=0.1, sequence_index=0)
    short_msg = Message(role="assistant", content="hi", weight=10.0, sequence_index=1)
    conv = Conversation([long_msg, short_msg])
    budget = TokenBudget(max_tokens=50)
    algo = GreedyByWeightAlgorithm()
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
