"""Tests for models: budgets, ContextPlan, CostEstimator, pricing."""

from __future__ import annotations

import pytest

from lethes.algorithms.base import SelectionResult
from lethes.engine.constraints import ConstraintChecker, ConstraintSet, ConstraintViolation
from lethes.engine.cost_estimator import CostEstimator
from lethes.engine.planner import ContextPlan
from lethes.models.budget import (
    Budget,
    CompositeBudget,
    CostBudget,
    TokenBudget,
    TokenTargetBudget,
)
from lethes.models.conversation import Conversation
from lethes.models.message import Message
from lethes.models.pricing import ModelPricingEntry, ModelPricingTable
from lethes.utils.tokens import TokenCounter


# ── Budget protocol conformance ────────────────────────────────────────────────


def test_token_budget_is_budget_protocol():
    assert isinstance(TokenBudget(max_tokens=1000), Budget)


def test_token_target_budget_is_budget_protocol():
    assert isinstance(TokenTargetBudget(target_tokens=1000), Budget)


def test_cost_budget_is_budget_protocol():
    assert isinstance(CostBudget(max_cost_usd=1.0), Budget)


def test_composite_budget_is_budget_protocol():
    assert isinstance(CompositeBudget(token_budget=TokenBudget(max_tokens=1000)), Budget)


# ── TokenBudget ────────────────────────────────────────────────────────────────


def test_token_budget_not_exceeded():
    budget = TokenBudget(max_tokens=1000)
    assert not budget.is_exceeded(500)
    assert not budget.is_exceeded(1000)


def test_token_budget_exceeded():
    budget = TokenBudget(max_tokens=1000)
    assert budget.is_exceeded(1001)


def test_token_budget_unlimited_never_exceeded():
    budget = TokenBudget(max_tokens=0)
    assert not budget.is_exceeded(999_999)


def test_token_budget_headroom():
    budget = TokenBudget(max_tokens=1000)
    assert budget.headroom_tokens(600) == 400
    assert budget.headroom_tokens(1000) == 0
    assert budget.headroom_tokens(1200) == 0  # clamped at 0


def test_token_budget_unlimited_headroom():
    budget = TokenBudget(max_tokens=0)
    assert budget.headroom_tokens(500) == -1


# ── TokenTargetBudget ──────────────────────────────────────────────────────────


def test_token_target_within_range():
    budget = TokenTargetBudget(target_tokens=1000, overshoot=150)
    assert not budget.is_exceeded(900)
    assert not budget.is_exceeded(1000)
    assert not budget.is_exceeded(1150)


def test_token_target_exceeded_past_overshoot():
    budget = TokenTargetBudget(target_tokens=1000, overshoot=150)
    assert budget.is_exceeded(1151)


def test_token_target_headroom():
    budget = TokenTargetBudget(target_tokens=1000, overshoot=150)
    assert budget.headroom_tokens(800) == 350   # 1000+150 - 800
    assert budget.headroom_tokens(1200) == 0    # clamped


def test_token_target_default_overshoot():
    budget = TokenTargetBudget(target_tokens=5000)
    # Default overshoot is 150
    assert not budget.is_exceeded(5150)
    assert budget.is_exceeded(5151)


# ── CostBudget ─────────────────────────────────────────────────────────────────


def test_cost_budget_not_exceeded():
    budget = CostBudget(max_cost_usd=1.0)
    assert not budget.is_exceeded(0, cost_usd=0.5)


def test_cost_budget_exceeded():
    budget = CostBudget(max_cost_usd=1.0)
    assert budget.is_exceeded(0, cost_usd=1.01)


def test_cost_budget_unlimited_never_exceeded():
    budget = CostBudget(max_cost_usd=0.0)
    assert not budget.is_exceeded(0, cost_usd=999.0)


def test_cost_budget_headroom_is_unlimited():
    budget = CostBudget(max_cost_usd=1.0)
    assert budget.headroom_tokens(500) == -1


# ── CompositeBudget ────────────────────────────────────────────────────────────


def test_composite_budget_both_ok():
    budget = CompositeBudget(
        token_budget=TokenBudget(max_tokens=1000),
        cost_budget=CostBudget(max_cost_usd=1.0),
    )
    assert not budget.is_exceeded(500, cost_usd=0.5)


def test_composite_budget_tokens_exceeded():
    budget = CompositeBudget(
        token_budget=TokenBudget(max_tokens=1000),
        cost_budget=CostBudget(max_cost_usd=10.0),
    )
    assert budget.is_exceeded(1001, cost_usd=0.01)


def test_composite_budget_cost_exceeded():
    budget = CompositeBudget(
        token_budget=TokenBudget(max_tokens=100_000),
        cost_budget=CostBudget(max_cost_usd=0.01),
    )
    assert budget.is_exceeded(100, cost_usd=0.05)


def test_composite_budget_no_cost_budget():
    budget = CompositeBudget(token_budget=TokenBudget(max_tokens=500))
    assert not budget.is_exceeded(400, cost_usd=9999.0)
    assert budget.is_exceeded(501, cost_usd=0.0)


def test_composite_budget_unlimited():
    budget = CompositeBudget.unlimited()
    assert not budget.is_exceeded(999_999, cost_usd=999.0)


# ── ModelPricingEntry & ModelPricingTable ──────────────────────────────────────


def _table() -> ModelPricingTable:
    return ModelPricingTable.from_list([
        {
            "model_id": "gpt-4o",
            "input_price_per_1m": 5.0,
            "cached_price_per_1m": 1.25,
            "output_price_per_1m": 15.0,
        },
        {
            "model_id": "gpt-4o-mini*",
            "input_price_per_1m": 0.15,
            "cached_price_per_1m": 0.075,
            "output_price_per_1m": 0.60,
        },
    ])


def test_pricing_exact_match():
    table = _table()
    entry = table.get("gpt-4o")
    assert entry is not None
    assert entry.input_price_per_1m == 5.0


def test_pricing_glob_match():
    table = _table()
    entry = table.get("gpt-4o-mini-2024")
    assert entry is not None
    assert entry.input_price_per_1m == 0.15


def test_pricing_no_match_returns_none():
    table = _table()
    assert table.get("claude-3-sonnet") is None


def test_pricing_estimate_cost_basic():
    table = _table()
    # 1M tokens * $5/M = $5
    cost = table.estimate_cost("gpt-4o", input_tokens=1_000_000)
    assert abs(cost - 5.0) < 1e-6


def test_pricing_estimate_cost_with_cache():
    table = _table()
    # 500k cached (1.25/M) + 500k uncached (5/M)
    cost = table.estimate_cost("gpt-4o", input_tokens=1_000_000, cached_tokens=500_000)
    expected = (500_000 * 1.25 + 500_000 * 5.0) / 1_000_000
    assert abs(cost - expected) < 1e-6


def test_pricing_estimate_cost_unknown_model():
    table = _table()
    assert table.estimate_cost("unknown-model", input_tokens=1_000_000) == 0.0


def test_pricing_estimate_cost_output_tokens():
    table = _table()
    cost = table.estimate_cost("gpt-4o", input_tokens=0, output_tokens=1_000_000)
    assert abs(cost - 15.0) < 1e-6


def test_pricing_exact_match_takes_precedence_over_glob():
    """Exact match should be returned even if a glob also matches."""
    table = ModelPricingTable.from_list([
        {"model_id": "gpt-4o*", "input_price_per_1m": 99.0,
         "cached_price_per_1m": 0, "output_price_per_1m": 0},
        {"model_id": "gpt-4o", "input_price_per_1m": 5.0,
         "cached_price_per_1m": 0, "output_price_per_1m": 0},
    ])
    entry = table.get("gpt-4o")
    assert entry.input_price_per_1m == 5.0


# ── ContextPlan ────────────────────────────────────────────────────────────────


def _plan(keep=(), summarize=(), drop=()) -> ContextPlan:
    return ContextPlan(
        keep_full=frozenset(keep),
        summarize=frozenset(summarize),
        drop=frozenset(drop),
        algorithm_name="test",
        weighting_strategy_name="static",
        pre_plan_tokens=100,
        post_plan_tokens=80,
    )


def test_plan_total_kept():
    plan = _plan(keep=["a", "b"], summarize=["c"])
    assert plan.total_kept == 3


def test_plan_total_dropped():
    plan = _plan(drop=["x", "y"])
    assert plan.total_dropped == 2


def test_plan_summarize_groups_returns_list():
    plan = _plan(summarize=["a", "b"])
    groups = plan.summarize_groups()
    assert isinstance(groups, list)
    all_ids = {mid for g in groups for mid in g}
    assert all_ids == {"a", "b"}


def test_plan_from_selection_result():
    result = SelectionResult(
        keep_full=["a", "b"],
        summarize=["c"],
        drop=["d"],
        estimated_tokens=50,
    )
    plan = ContextPlan.from_selection_result(
        result, algorithm_name="greedy", weighting_strategy_name="smart",
        pre_plan_tokens=200,
    )
    assert plan.keep_full == frozenset(["a", "b"])
    assert plan.summarize == frozenset(["c"])
    assert plan.drop == frozenset(["d"])
    assert plan.post_plan_tokens == 50
    assert plan.algorithm_name == "greedy"


# ── ConstraintViolation & ConstraintChecker ────────────────────────────────────


def test_constraint_violation_fields():
    v = ConstraintViolation(rule="require_last_user", message_id="m1",
                             description="Last user dropped")
    assert v.rule == "require_last_user"
    assert v.message_id == "m1"


def test_constraint_checker_no_violations():
    checker = ConstraintChecker()
    constraints = ConstraintSet()
    user = Message(role="user", content="hello", sequence_index=0)
    conv = Conversation([user])
    result = SelectionResult(keep_full=[user.id], summarize=[], drop=[])
    violations = checker.validate(result, conv, constraints)
    assert violations == []


def test_constraint_checker_last_user_dropped_violation():
    checker = ConstraintChecker()
    constraints = ConstraintSet(require_last_user=True)
    user = Message(role="user", content="query", sequence_index=0)
    conv = Conversation([user])
    result = SelectionResult(keep_full=[], summarize=[], drop=[user.id])
    violations = checker.validate(result, conv, constraints)
    assert any(v.rule == "require_last_user" for v in violations)


def test_constraint_checker_last_user_summarized_violation():
    checker = ConstraintChecker()
    constraints = ConstraintSet(require_last_user=True)
    user = Message(role="user", content="query", sequence_index=0)
    conv = Conversation([user])
    result = SelectionResult(keep_full=[], summarize=[user.id], drop=[])
    violations = checker.validate(result, conv, constraints)
    assert any(v.rule == "require_last_user" for v in violations)


def test_constraint_checker_min_chat_messages_violation():
    checker = ConstraintChecker()
    constraints = ConstraintSet(min_chat_messages=2)
    user = Message(role="user", content="q", sequence_index=0)
    conv = Conversation([user])
    result = SelectionResult(keep_full=[user.id], summarize=[], drop=[])
    violations = checker.validate(result, conv, constraints)
    assert any(v.rule == "min_chat_messages" for v in violations)


def test_constraint_checker_repair_last_user():
    checker = ConstraintChecker()
    constraints = ConstraintSet(require_last_user=True)
    user = Message(role="user", content="query", sequence_index=0)
    conv = Conversation([user])
    result = SelectionResult(keep_full=[], summarize=[], drop=[user.id])
    repaired = checker.repair(result, conv, constraints)
    assert user.id in repaired.keep_full
    assert user.id not in repaired.drop


def test_constraint_checker_repair_min_chat():
    checker = ConstraintChecker()
    constraints = ConstraintSet(min_chat_messages=2, require_last_user=False)
    m1 = Message(role="user", content="a", sequence_index=0)
    m2 = Message(role="assistant", content="b", sequence_index=1)
    conv = Conversation([m1, m2])
    # Only m2 kept; m1 dropped
    result = SelectionResult(keep_full=[m2.id], summarize=[], drop=[m1.id])
    repaired = checker.repair(result, conv, constraints)
    assert len(repaired.keep_full) + len(repaired.summarize) >= 2


# ── CostEstimator ─────────────────────────────────────────────────────────────


def test_cost_estimator_basic():
    table = _table()
    counter = TokenCounter()
    estimator = CostEstimator(table, counter)
    msgs = [Message(role="user", content="hello world")]
    conv = Conversation(msgs)
    cost = estimator.estimate(conv, "gpt-4o")
    assert cost >= 0.0


def test_cost_estimator_token_count():
    table = _table()
    counter = TokenCounter()
    estimator = CostEstimator(table, counter)
    msgs = [Message(role="user", content="hello world")]
    conv = Conversation(msgs)
    count = estimator.token_count(conv)
    assert count > 0


def test_cost_estimator_unknown_model_returns_zero():
    table = _table()
    counter = TokenCounter()
    estimator = CostEstimator(table, counter)
    msgs = [Message(role="user", content="hello")]
    conv = Conversation(msgs)
    cost = estimator.estimate(conv, "unknown-xyz")
    assert cost == 0.0
