"""Dispatch observations shared by embedded and HTTP integrations."""

from dataclasses import asdict

from .state import ContextResult


def context_observation(
    result: ContextResult, model: str | None, input_limit: int, elapsed_ms: float
) -> dict:
    return {
        "request_id": result.request_id,
        "model": model,
        "epoch": result.snapshot.epoch,
        "reason": result.reason,
        "input_tokens": result.input_tokens,
        "input_limit": input_limit,
        "count_source": result.count_source,
        "count_reference": result.count_reference,
        "potential_cached_tokens": result.potential_cached_tokens,
        "recalled_messages": len({i for s in result.recalled for i in s.indices()}),
        "omitted_messages": len({i for s in result.omitted for i in s.indices()}),
        "recalled": [asdict(s) for s in result.recalled],
        "omitted": [asdict(s) for s in result.omitted],
        "warnings": list(result.warnings),
        "economics": result.economics,
        "elapsed_ms": round(elapsed_ms, 2),
        "layout": {
            "head": result.snapshot.head,
            "history_length": result.snapshot.history_length,
            "tail_start": result.snapshot.tail_start,
            "reminders": [
                {"anchor": r.anchor, "spans": [asdict(s) for s in r.spans]}
                for r in result.snapshot.reminders
            ],
        },
    }


def status_line(observation: dict) -> str:
    sources = {
        "reported_usage": "历史实测",
        "reported_prefix_plus_estimate": "实测+估算",
        "usage_calibrated_estimate": "校准估算",
        "local_estimate": "估算",
        "provider_usage": "本次实测",
    }
    reasons = {
        "initial": "初建",
        "append": "续接",
        "replay": "复用",
        "capacity": "容量重建",
        "recall_capacity": "召回重建",
        "economic": "经济重建",
        "count_capacity": "计数修复",
    }
    source = observation["count_source"]
    approximate = "" if source in {"reported_usage", "provider_usage"} else "≈"
    line = (
        f"lethes · 输入 {approximate}{observation['input_tokens']:,}"
        f"/{observation['input_limit']:,} token（{sources.get(source, '估算')}）"
        f" · {reasons.get(observation['reason'], '就绪')}"
        f" · 召回 {observation['recalled_messages']}"
    )
    if observation.get("cached_tokens") is not None:
        line += f" · 缓存读 {observation['cached_tokens']:,}"
    if observation["omitted_messages"]:
        line += f" · 未纳入 {observation['omitted_messages']}"
    if observation["warnings"]:
        line += f" · 检索降级 {len(observation['warnings'])}"
    return line
