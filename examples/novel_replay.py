"""Replay a local novel without uploading it or committing its contents.

Artificial 'continue' turns exercise the mechanism only; no model is asked to
continue the novel. Optional human-written probes evaluate source passage recall.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import time
from pathlib import Path

from lethes import (
    CachePricing,
    ContextConfig,
    ContextManager,
    LiteLLMTokenCounter,
    SQLiteStore,
    TiktokenCounter,
)
from lethes.history import History, normalize


def chunks(source: str, maximum: int) -> list[str]:
    if maximum <= 0:
        raise ValueError("Chunk size must be positive")
    result: list[str] = []
    pending = ""
    for line in source.splitlines(keepends=True):
        while line:
            available = maximum - len(pending)
            if len(line) <= available:
                pending += line
                break
            if pending:
                result.append(pending)
                pending = ""
            elif len(line) > maximum:
                result.append(line[:maximum])
                line = line[maximum:]
    if pending:
        result.append(pending)
    assert "".join(result) == source
    return result


def dialogue(passages: list[str]) -> list[dict]:
    result = [{"role": "system", "content": "Use the supplied original narrative as context."}]
    for passage in passages:
        result.extend(
            [{"role": "user", "content": "继续"}, {"role": "assistant", "content": passage}]
        )
    return result


def validate(result, raw: list[dict], limit: int) -> None:
    history = History.from_messages(raw)
    assert result.estimated_tokens <= limit
    assert result.messages[-1] == normalize(raw[-1])
    quoted: set[int] = set()
    for reminder in result.snapshot.reminders:
        data = json.loads(json.loads(reminder.payload)["content"].split("\n", 1)[1])
        for record in data:
            index = record["message_index"]
            assert index not in quoted and index < result.snapshot.tail_start
            quoted.add(index)
            assert index < len(raw)
            assert record["path_revision"] == history.heads[index]
            assert record["message"] == normalize(raw[index])


async def run(args: argparse.Namespace) -> None:
    source = args.text.read_text(encoding=args.encoding)
    tokenizer = getattr(args, "tokenizer", "bytes")
    tokenizer_model = getattr(args, "tokenizer_model", "")
    passages = chunks(source, args.chunk_chars)
    original = dialogue(passages)
    store = SQLiteStore(args.state)
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
            recent_tokens=8192,
            minimum_recent_turns=2,
        ),
        store=store,
        pricing=pricing,
        counter=(
            LiteLLMTokenCounter(tokenizer_model)
            if tokenizer == "litellm"
            else TiktokenCounter(tokenizer)
            if tokenizer != "bytes"
            else None
        ),
    )
    started = time.perf_counter()
    rows = []
    probe_rows = []
    try:
        for n in range(1, len(passages) + 1, args.stride):
            raw = original[: 1 + 2 * n] + [{"role": "user", "content": "继续"}]
            result = await manager.prepare(raw, user_id="local", chat_id="novel")
            validate(result, raw, manager.config.input_limit)
            rows.append(
                {
                    "passages": n,
                    "input_messages": len(raw),
                    "estimated_tokens": result.estimated_tokens,
                    "potential_cached_tokens": result.potential_cached_tokens,
                    "epoch": result.snapshot.epoch,
                    "reason": result.reason,
                    "reminders": len(result.snapshot.reminders),
                    "omitted_ranges": len(result.omitted),
                }
            )
        # Rewind/edit/restore is a separate path, not an append-only special case.
        rewind_n = min(15, len(passages))
        rewind = original[: 1 + 2 * rewind_n] + [{"role": "user", "content": "继续"}]
        before = await manager.prepare(rewind, user_id="local", chat_id="novel")
        edited = [dict(m) for m in rewind]
        edited[-2]["content"] = "（测试：替换此处旁白，形成另一条历史路径。）"
        branch = await manager.prepare(edited, user_id="local", chat_id="novel")
        validate(branch, edited, manager.config.input_limit)
        restored = await manager.prepare(rewind, user_id="local", chat_id="novel")
        assert before.snapshot == restored.snapshot
        if args.probes:
            for i, probe in enumerate(json.loads(args.probes.read_text())):
                raw = original + [{"role": "user", "content": probe["query"]}]
                result = await manager.prepare(raw, user_id="local", chat_id=f"probe-{i}")
                validate(result, raw, manager.config.input_limit)
                targets = [
                    j
                    for j, message in enumerate(raw[:-1])
                    if probe["expected"] in str(message.get("content", ""))
                ]
                if not targets:
                    raise ValueError(f"Probe {i} anchor is absent or crosses a chunk boundary")
                covered = set(range(result.snapshot.tail_start, len(raw))) | {
                    j for r in result.snapshot.reminders for s in r.spans for j in s.indices()
                }
                probe_rows.append(
                    {
                        "probe": i,
                        "hit": bool(set(targets) & covered),
                        "target_messages": targets,
                        "recalled_ranges": [[s.start, s.end] for s in result.recalled],
                        "estimated_tokens": result.estimated_tokens,
                    }
                )
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / "novel.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = {
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "characters": len(source),
            "passages": len(passages),
            "requests": len(rows),
            "epochs": len({r["epoch"] for r in rows}),
            "max_estimated_tokens": max(r["estimated_tokens"] for r in rows),
            "input_limit": manager.config.input_limit,
            "branch_restore": True,
            "raw_passages_validated": True,
            "duplicate_source_citations": 0,
            "sum_estimated_input": sum(r["estimated_tokens"] for r in rows),
            "sum_potential_cached": sum(r["potential_cached_tokens"] for r in rows),
            "sum_potential_uncached": sum(
                r["estimated_tokens"] - r["potential_cached_tokens"] for r in rows
            ),
            "actual_cache_usage": None,
            "tokenizer": getattr(args, "tokenizer", "bytes"),
            "tokenizer_model": getattr(args, "tokenizer_model", None),
            "probe_results": probe_rows,
            "elapsed_seconds": round(time.perf_counter() - started, 2),
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        store.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--encoding", default="utf-8-sig")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--context-window", type=int, default=32768)
    parser.add_argument("--tokenizer", default="bytes")
    parser.add_argument("--tokenizer-model", default="")
    parser.add_argument("--state", default=":memory:")
    parser.add_argument("--input-price", type=float)
    parser.add_argument("--cached-price", type=float)
    parser.add_argument("--probes", type=Path)
    parser.add_argument("--output", type=Path, default=Path("/tmp/lethes-novel-report"))
    args = parser.parse_args()
    if args.stride <= 0 or not args.text.stat().st_size:
        parser.error("Input must be nonempty and stride positive")
    asyncio.run(run(args))
