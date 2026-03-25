"""Tests for the 5-label LLMContextAnalyzer."""

from __future__ import annotations

import json

import pytest

from lethes.models.conversation import Conversation
from lethes.models.message import Message
from lethes.weighting.llm_analyzer import (
    DEFAULT_LABEL,
    LABEL_WEIGHTS,
    LABELS,
    LLMContextAnalyzer,
)


def _analyzer(**kwargs) -> LLMContextAnalyzer:
    return LLMContextAnalyzer(
        api_base="https://api.example.com/v1",
        api_key="test",
        **kwargs,
    )


# ── Label constants ───────────────────────────────────────────────────────────

def test_labels_are_five():
    assert len(LABELS) == 5


def test_all_labels_have_weights():
    for label in LABELS:
        assert label in LABEL_WEIGHTS


def test_weights_are_ordered():
    """K > H > M > S > D"""
    k, h, m, s, d = (LABEL_WEIGHTS[l] for l in ("K", "H", "M", "S", "D"))
    assert k > h > m > s > d


def test_default_label_is_middle():
    assert DEFAULT_LABEL == "M"
    assert LABEL_WEIGHTS[DEFAULT_LABEL] == 0.5


# ── _parse_labels ─────────────────────────────────────────────────────────────

def test_parse_labels_json_object():
    a = _analyzer()
    result = a._parse_labels('{"labels": ["K", "H", "M", "S", "D"]}', expected_length=5)
    assert result == ["K", "H", "M", "S", "D"]


def test_parse_labels_json_array():
    a = _analyzer()
    result = a._parse_labels('["K", "M", "D"]', expected_length=3)
    assert result == ["K", "M", "D"]


def test_parse_labels_with_markdown_fences():
    a = _analyzer()
    raw = '```json\n{"labels": ["K", "H", "M"]}\n```'
    result = a._parse_labels(raw, expected_length=3)
    assert result == ["K", "H", "M"]


def test_parse_labels_lowercase_normalised():
    a = _analyzer()
    result = a._parse_labels('{"labels": ["k", "h", "m"]}', expected_length=3)
    assert result == ["K", "H", "M"]


def test_parse_labels_regex_fallback():
    """When JSON is broken, fall back to extracting K/H/M/S/D from text."""
    a = _analyzer()
    # LLM returned explanatory text instead of JSON
    raw = "Message 1 is K (keep), message 2 is S (skip), message 3 is M (maybe)."
    result = a._parse_labels(raw, expected_length=3)
    assert result == ["K", "S", "M"]


def test_parse_labels_returns_none_on_empty():
    a = _analyzer()
    result = a._parse_labels("I don't know.", expected_length=3)
    assert result is None


def test_parse_labels_truncates_if_too_long():
    a = _analyzer()
    result = a._parse_labels('{"labels": ["K", "H", "M", "S", "D"]}', expected_length=3)
    assert result == ["K", "H", "M"]


def test_parse_labels_pads_if_too_short():
    a = _analyzer()
    result = a._parse_labels('{"labels": ["K", "H"]}', expected_length=4)
    assert result == ["K", "H", DEFAULT_LABEL, DEFAULT_LABEL]


def test_parse_labels_unknown_label_replaced_with_default():
    a = _analyzer()
    result = a._parse_labels('{"labels": ["K", "X", "Z"]}', expected_length=3)
    assert result == ["K", DEFAULT_LABEL, DEFAULT_LABEL]


# ── _build_result ─────────────────────────────────────────────────────────────

def _make_msgs(n: int) -> list[Message]:
    roles = ["user", "assistant"] * 10
    return [
        Message(role=roles[i % len(roles)], content=f"msg {i}", sequence_index=i)
        for i in range(n)
    ]


def test_build_result_maps_labels_to_weights():
    a = _analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = a._build_result(msgs, msgs, ["K", "S", "D"], set(), set())
    assert scores[msgs[0].id] == LABEL_WEIGHTS["K"]
    assert scores[msgs[1].id] == LABEL_WEIGHTS["S"]
    assert scores[msgs[2].id] == LABEL_WEIGHTS["D"]


def test_build_result_system_messages_always_one():
    a = _analyzer()
    sys_msg = Message(role="system", content="You are helpful.", sequence_index=0)
    usr_msg = Message(role="user", content="hello", sequence_index=1)
    conv = Conversation([sys_msg, usr_msg])
    system_ids = {sys_msg.id}
    scores = a._build_result(
        [sys_msg, usr_msg],
        [usr_msg],           # window excludes system
        ["D"],               # LLM would label user as Drop
        set(),
        system_ids,
    )
    assert scores[sys_msg.id] == 1.0
    assert scores[usr_msg.id] == LABEL_WEIGHTS["D"]


def test_build_result_out_of_window_get_default():
    a = _analyzer()
    msgs = _make_msgs(5)
    conv = Conversation(msgs)
    window = msgs[3:]          # only last 2 in window
    out_of_window = {m.id for m in msgs[:3]}
    scores = a._build_result(msgs, window, ["K", "H"], out_of_window, set())
    # out-of-window messages get default weight
    for i in range(3):
        assert scores[msgs[i].id] == LABEL_WEIGHTS[DEFAULT_LABEL]
    assert scores[msgs[3].id] == LABEL_WEIGHTS["K"]
    assert scores[msgs[4].id] == LABEL_WEIGHTS["H"]


def test_build_result_none_labels_all_get_default():
    a = _analyzer()
    msgs = _make_msgs(3)
    scores = a._build_result(msgs, msgs, None, set(), set())
    for m in msgs:
        assert scores[m.id] == LABEL_WEIGHTS[DEFAULT_LABEL]


# ── _format_snippet ───────────────────────────────────────────────────────────

def test_format_snippet_truncates():
    a = _analyzer(content_truncate_chars=10)
    msg = Message(role="user", content="hello world this is a long message")
    snippet = a._format_snippet(msg)
    assert len(snippet) <= 13  # 10 chars + "…"
    assert snippet.endswith("…")


def test_format_snippet_tool_calls_shows_function_names():
    a = _analyzer()
    msg = Message(
        role="assistant",
        content=None,
        tool_calls=[
            {"id": "c1", "type": "function", "function": {"name": "get_weather"}},
            {"id": "c2", "type": "function", "function": {"name": "search_web"}},
        ],
    )
    snippet = a._format_snippet(msg)
    assert "get_weather" in snippet
    assert "search_web" in snippet


def test_format_snippet_tool_result_prefixed():
    a = _analyzer()
    msg = Message(role="tool", content="22°C, sunny", tool_call_id="c1")
    snippet = a._format_snippet(msg)
    assert snippet.startswith("[tool result]")
    assert "22°C" in snippet


# ── Integration: score() with mocked LLM ─────────────────────────────────────

class _FakeCache:
    def __init__(self): self._store = {}
    async def get(self, key): return self._store.get(key)
    async def set(self, key, value, ttl=None): self._store[key] = value
    async def delete(self, key): self._store.pop(key, None)
    async def exists(self, key): return key in self._store


@pytest.mark.asyncio
async def test_score_uses_cache_on_second_call(respx_mock):
    """Second call with same messages should hit cache, not LLM."""
    import respx
    import httpx

    a = _analyzer(cache=_FakeCache())
    msgs = _make_msgs(3)
    conv = Conversation(msgs)

    # Mock LLM response
    route = respx_mock.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"labels": ["K", "H", "M"]}'}}
                ]
            },
        )
    )

    # First call — hits LLM
    scores1 = await a.score(msgs, "test query", conv)
    assert route.called
    assert route.call_count == 1

    # Second call with same input — should hit cache
    scores2 = await a.score(msgs, "test query", conv)
    assert route.call_count == 1  # still 1, not 2

    assert scores1 == scores2


@pytest.mark.asyncio
async def test_score_falls_back_to_default_on_llm_failure(respx_mock):
    """LLM failure → all non-system messages get default weight."""
    import httpx

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )

    a = _analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "query", conv)

    default_w = LABEL_WEIGHTS[DEFAULT_LABEL]
    for m in msgs:
        assert scores[m.id] == default_w


@pytest.mark.asyncio
async def test_score_label_k_means_highest_weight(respx_mock):
    """K label maps to the highest weight (1.0)."""
    import httpx

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"labels": ["D", "D", "K"]}'}}
                ]
            },
        )
    )

    a = _analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "query", conv)

    assert scores[msgs[2].id] == 1.0        # K
    assert scores[msgs[0].id] == LABEL_WEIGHTS["D"]
    assert scores[msgs[1].id] == LABEL_WEIGHTS["D"]
