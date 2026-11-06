import math

import pytest

from lethes import CachePricing, RenewalPolicy
from lethes.costs import parse_usage


def test_cache_eligibility_rounding_and_actual_price():
    pricing = CachePricing(10, 1, minimum_cache_tokens=1024, cache_block_tokens=128)
    assert pricing.eligible(1023) == 0
    assert pricing.eligible(1151) == 1024
    assert pricing.cost(2000, 1151) == pytest.approx((976 * 10 + 1024) / 1e6)


def test_cycle_rate_matches_independent_enumeration():
    pricing = CachePricing(10, 1, 0, 1)
    policy = RenewalPolicy(pricing)
    decision = policy.decide(
        old_tokens=5000, new_tokens=1000, old_shared=4900, new_shared=0, growth=100, capacity=6000
    )
    cycles = []
    for n in range(1, 52):
        total = 1000 * 10
        for step in range(1, n):
            total += 100 * 10 + (1000 + (step - 1) * 100)
        cycles.append((total / n / 1e6, n))
    average, rounds = min(cycles)
    assert decision.average_cost == pytest.approx(average)
    assert decision.cycle_rounds == rounds


def test_larger_cached_prefix_can_be_cheaper_than_a_cold_shorter_one():
    decision = RenewalPolicy(CachePricing(10, 0.01, 0, 1)).decide(
        old_tokens=2000, new_tokens=1000, old_shared=1900, new_shared=0, growth=100, capacity=20000
    )
    assert not decision.rebuild


def test_expensive_bloated_prefix_is_replaced():
    decision = RenewalPolicy(CachePricing(10, 5, 0, 1)).decide(
        old_tokens=18000,
        new_tokens=1000,
        old_shared=17900,
        new_shared=0,
        growth=100,
        capacity=20000,
    )
    assert decision.rebuild
    assert decision.keep_excess > decision.rebuild_excess


def test_rebuild_overhead_can_reverse_decision():
    policy = RenewalPolicy(CachePricing(10, 5, 0, 1))
    params = dict(
        old_tokens=18000,
        new_tokens=1000,
        old_shared=17900,
        new_shared=0,
        growth=100,
        capacity=20000,
    )
    assert policy.decide(**params).rebuild
    assert not policy.decide(**params, rebuild_overhead=100).rebuild


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(uncached_per_million=-1, cached_per_million=0),
        dict(uncached_per_million=math.nan, cached_per_million=0),
    ],
)
def test_invalid_prices(kwargs):
    with pytest.raises(ValueError):
        CachePricing(**kwargs)


def test_missing_usage_is_not_fabricated():
    assert parse_usage({}) == {"raw": {}}
    assert "cached_tokens" not in parse_usage({"prompt_tokens": 100})


def test_openai_and_anthropic_usage():
    openai = parse_usage(
        {
            "prompt_tokens": 1000,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 800},
        }
    )
    assert openai["input_tokens"] == 1000 and openai["cached_tokens"] == 800
    anthropic = parse_usage(
        {
            "input_tokens": 100,
            "cache_read_input_tokens": 800,
            "cache_creation_input_tokens": 200,
            "output_tokens": 20,
        }
    )
    assert anthropic["input_tokens"] == 1100 and anthropic["cached_tokens"] == 800


async def test_manager_applies_economic_rebuild_to_bloated_prefix():
    from conftest import ScriptedRetriever, prepare

    from lethes import ContextConfig, ContextManager

    manager = ContextManager(
        ContextConfig(
            context_window=12000,
            reply_reserve=500,
            safety_reserve=200,
            recent_tokens=500,
            minimum_recent_turns=1,
        ),
        pricing=CachePricing(10, 5, 0, 1),
        retriever=ScriptedRetriever(),
    )
    raw = [{"role": "user", "content": "old " * 1700}]
    first = await prepare(manager, raw)
    raw += [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "New scene."}]
    second = await prepare(manager, raw)
    assert second.reason == "economic"
    assert second.snapshot.epoch != first.snapshot.epoch
    assert second.economics["keep_excess"] > second.economics["rebuild_excess"]


async def test_economically_compacted_short_history_still_queries_omitted_originals():
    from conftest import ScriptedRetriever, prepare

    from lethes import ContextConfig, ContextManager, Span

    manager = ContextManager(
        ContextConfig(
            context_window=12000,
            reply_reserve=500,
            safety_reserve=200,
            recent_tokens=500,
            minimum_recent_turns=1,
        ),
        pricing=CachePricing(10, 5, 0, 1),
        retriever=ScriptedRetriever(),
    )
    original = "The original setting. " * 320
    raw = [{"role": "user", "content": original}]
    await prepare(manager, raw)
    raw += [{"role": "assistant", "content": "Done."}, {"role": "user", "content": "New scene."}]
    compacted = await prepare(manager, raw)
    assert compacted.reason == "economic"
    assert original not in str(compacted.messages)
    manager.retriever = ScriptedRetriever((Span(0, 1),))
    raw += [
        {"role": "assistant", "content": "She nods."},
        {"role": "user", "content": "Recall the original setting."},
    ]
    recalled = await prepare(manager, raw)
    assert original in str(recalled.messages)
