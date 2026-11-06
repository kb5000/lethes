"""Offline replay of complete chat paths; no main-model calls or invented usage."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

from lethes import CachePricing, ContextConfig, ContextManager, SQLiteStore


def synthetic(turns: int) -> list[dict]:
    messages = [{"role": "system", "content": "Continue the roleplay in Chinese."}]
    messages += [
        {"role": "user", "content": "银月剑出鞘需要献出一段记忆。"},
        {"role": "assistant", "content": "她记住了这个代价，并把剑收进剑鞘。"},
    ]
    for i in range(turns):
        messages.extend(
            [
                {"role": "user", "content": f"第{i}段旅程：她走过森林，询问当地的故事。"},
                {
                    "role": "assistant",
                    "content": "风吹过树梢。旅店老板为她指路，她继续向城堡前行。",
                },
            ]
        )
    messages.append({"role": "user", "content": "她握住银月剑，准备将它拔出。"})
    return messages


def histories(data: object) -> list[list[dict]]:
    if isinstance(data, dict) and "histories" in data:
        return data["histories"]
    messages = data["messages"] if isinstance(data, dict) else data
    if not isinstance(messages, list):
        raise ValueError("Expected messages or history snapshots")
    return [messages[: i + 1] for i, message in enumerate(messages) if message["role"] == "user"]


async def replay(args: argparse.Namespace) -> None:
    data = json.loads(args.input.read_text()) if args.input else synthetic(args.synthetic_turns)
    pricing = None
    if args.input_price is not None and args.cached_price is not None:
        pricing = CachePricing(args.input_price, args.cached_price)
    store = SQLiteStore(args.state)
    manager = ContextManager(
        ContextConfig(
            context_window=args.context_window,
            reply_reserve=512,
            safety_reserve=256,
            recent_tokens=1200,
            minimum_recent_turns=2,
        ),
        store=store,
        pricing=pricing,
    )
    rows = []
    try:
        for turn, history in enumerate(histories(data)):
            result = await manager.prepare(history, user_id="replay", chat_id="replay")
            rows.append(
                {
                    "turn": turn,
                    "history_messages": len(history),
                    "estimated_input_tokens": result.estimated_tokens,
                    "potential_cached_tokens": result.potential_cached_tokens,
                    "epoch": result.snapshot.epoch,
                    "reason": result.reason,
                    "reminders": len(result.snapshot.reminders),
                    "omitted_ranges": len(result.omitted),
                }
            )
        if not rows:
            raise ValueError("No user-turn snapshots to replay")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(
            json.dumps(
                {
                    "requests": len(rows),
                    "epochs": len({r["epoch"] for r in rows}),
                    "maximum_estimated_input_tokens": max(
                        r["estimated_input_tokens"] for r in rows
                    ),
                    "actual_cache_usage": None,
                    "output": str(args.output),
                },
                ensure_ascii=False,
            )
        )
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--synthetic-turns", type=int, default=300)
    parser.add_argument("--context-window", type=int, default=8000)
    parser.add_argument("--state", default=":memory:")
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cached-price", type=float)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lethes-replay.csv"))
    asyncio.run(replay(parser.parse_args()))
