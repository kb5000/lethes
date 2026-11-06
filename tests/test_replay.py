import argparse
import json
import runpy

import pytest


@pytest.mark.parametrize("maximum", [1, 7, 64, 900])
def test_novel_chunking_preserves_every_original_character(maximum):
    chunker = runpy.run_path("examples/novel_replay.py")["chunks"]
    source = "第一章\n\n她说：‘继续。’  \n" + "很长的段落" * 300 + "\n尾声"
    pieces = chunker(source, maximum)
    assert "".join(pieces) == source
    assert all(0 < len(p) <= maximum for p in pieces)


async def test_local_novel_replay_report_without_main_model_or_source_publication(tmp_path):
    module = runpy.run_path("examples/novel_replay.py")
    source = "第一章\n银剑的代价是记忆。\n" + "旅人继续走向城堡。\n" * 20
    path = tmp_path / "novel.txt"
    path.write_text(source)
    args = argparse.Namespace(
        text=path,
        encoding="utf-8",
        chunk_chars=64,
        stride=1,
        state=":memory:",
        context_window=12000,
        input_price=None,
        cached_price=None,
        probes=None,
        output=tmp_path / "report",
    )
    await module["run"](args)
    report = json.loads((args.output / "report.json").read_text())
    assert report["raw_passages_validated"]
    assert report["branch_restore"]
    assert report["actual_cache_usage"] is None
    assert source not in (args.output / "report.json").read_text()
