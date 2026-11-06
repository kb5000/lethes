"""Replay real Open WebUI exports, including all retained branch paths."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import time
from pathlib import Path

from lethes import CachePricing, ContextConfig, ContextManager, TiktokenCounter
from lethes.history import History
from lethes.importers import open_webui_paths


def validate(result, raw, limit):
    history = History.from_messages(raw)
    assert result.estimated_tokens <= limit
    assert result.messages[-1] == raw[-1]
    quoted: set[int] = set()
    for reminder in result.snapshot.reminders:
        excerpts = json.loads(json.loads(reminder.payload)["content"].split("\n", 1)[1])
        for excerpt in excerpts:
            i = excerpt["message_index"]
            assert i not in quoted and i < result.snapshot.tail_start
            quoted.add(i)
            assert i < len(raw)
            assert excerpt["message"] == raw[i]
            assert excerpt["path_revision"] == history.heads[i]


async def run(args):
    data = args.input.read_bytes()
    paths = open_webui_paths(json.loads(data))
    pricing = (
        None
        if args.input_price is None or args.cached_price is None
        else CachePricing(args.input_price, args.cached_price)
    )
    manager = ContextManager(
        ContextConfig(
            context_window=args.context_window,
            reply_reserve=2048,
            safety_reserve=1024,
            recent_tokens=args.recent_tokens,
        ),
        pricing=pricing,
        counter=TiktokenCounter(args.tokenizer) if args.tokenizer != "bytes" else None,
    )
    started = time.perf_counter()
    rows = []
    probes = []
    for path in paths:
        previous = None
        previous_history = None
        for i, message in enumerate(path.messages):
            if message["role"] != "user":
                continue
            raw = path.messages[: i + 1]
            result = await manager.prepare(raw, user_id="replay", chat_id=path.chat_id)
            validate(result, raw, manager.config.input_limit)
            if previous and result.snapshot.epoch == previous.snapshot.epoch:
                assert result.messages[: len(previous.messages)] == previous.messages
            rows.append(
                {
                    "path": hashlib.sha256(path.leaf_id.encode()).hexdigest()[:12],
                    "active": path.active,
                    "messages": len(raw),
                    "reason": result.reason,
                    "epoch": result.snapshot.epoch,
                    "estimated_tokens": result.estimated_tokens,
                    "potential_cached_tokens": result.potential_cached_tokens,
                    "omitted_ranges": len(result.omitted),
                }
            )
            previous, previous_history = result, raw
        if path.active and previous_history:
            baseline = await manager.prepare(
                previous_history, user_id="replay", chat_id=path.chat_id
            )
            for role in ("user", "assistant"):
                edited = [dict(m) for m in previous_history]
                index = next(j for j in range(len(edited) - 1, -1, -1) if edited[j]["role"] == role)
                edited[index]["content"] += "\n（回放测试：编辑此消息以形成分叉。）"
                changed = await manager.prepare(edited, user_id="replay", chat_id=path.chat_id)
                validate(changed, edited, manager.config.input_limit)
                restored = await manager.prepare(
                    previous_history, user_id="replay", chat_id=path.chat_id
                )
                assert restored.snapshot == baseline.snapshot
            if args.probes:
                for p in json.loads(args.probes.read_text()):
                    raw = path.messages + [{"role": "user", "content": p["query"]}]
                    result = await manager.prepare(raw, user_id="replay", chat_id=path.chat_id)
                    validate(result, raw, manager.config.input_limit)
                    targets = {j for j, m in enumerate(raw[:-1]) if p["expected"] in m["content"]}
                    if not targets:
                        raise ValueError("Probe expected substring is absent")
                    covered = set(range(result.snapshot.tail_start, len(raw))) | {
                        j for r in result.snapshot.reminders for s in r.spans for j in s.indices()
                    }
                    probes.append({"hit": bool(targets & covered), "targets": sorted(targets)})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "interactive.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "paths": len(paths),
        "path_lengths": [len(p.messages) for p in paths],
        "user_requests": len(rows),
        "epochs": len({r["epoch"] for r in rows}),
        "max_estimated_tokens": max(r["estimated_tokens"] for r in rows),
        "input_limit": manager.config.input_limit,
        "original_passages_valid": True,
        "duplicate_source_citations": 0,
        "sum_estimated_input": sum(r["estimated_tokens"] for r in rows),
        "sum_potential_cached": sum(r["potential_cached_tokens"] for r in rows),
        "sum_potential_uncached": sum(
            r["estimated_tokens"] - r["potential_cached_tokens"] for r in rows
        ),
        "stable_prefixes_valid": True,
        "input_output_edit_restore_valid": True,
        "actual_cache_usage": None,
        "tokenizer": args.tokenizer,
        "probe_results": probes,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    manager.store.close()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--recent-tokens", type=int, default=8192)
    parser.add_argument("--tokenizer", default="bytes")
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cached-price", type=float)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lethes-interactive-report"))
    asyncio.run(run(parser.parse_args()))
