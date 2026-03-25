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
    _extract_keywords,
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


# ── _extract_keywords ─────────────────────────────────────────────────────────

def test_extract_keywords_basic():
    text = "Python decorators are a powerful feature of Python programming"
    kws = _extract_keywords(text, top_n=3)
    assert "python" in kws
    assert "decorators" in kws


def test_extract_keywords_filters_stop_words():
    text = "the and for are but not you all can"
    kws = _extract_keywords(text, top_n=5)
    assert kws == []  # all stop words


def test_extract_keywords_top_n_respected():
    text = "alpha beta gamma delta epsilon zeta eta"
    kws = _extract_keywords(text, top_n=3)
    assert len(kws) <= 3


# ── _cluster_messages ─────────────────────────────────────────────────────────

def _make_topic_msgs(groups: list[tuple[str, list[str]]]) -> list[Message]:
    """Build messages grouped by topic. Each group is (role, [contents])."""
    msgs = []
    idx = 0
    for role, contents in groups:
        for content in contents:
            msgs.append(Message(role=role, content=content, sequence_index=idx))
            idx += 1
    return msgs


def test_cluster_single_topic():
    """All on-topic messages form one cluster."""
    a = _analyzer()
    msgs = [
        Message(role="user", content="Paris weather forecast today", sequence_index=i)
        for i in range(4)
    ]
    clusters = a._cluster_messages(msgs)
    assert len(clusters) >= 1
    # All indices present across all clusters
    all_indices = [i for c in clusters for i in c.indices]
    assert sorted(all_indices) == list(range(4))


def test_cluster_assigns_keywords():
    a = _analyzer()
    msgs = [
        Message(role="user", content="Python decorators usage", sequence_index=0),
        Message(role="assistant", content="Python decorators wrap functions", sequence_index=1),
    ]
    clusters = a._cluster_messages(msgs)
    assert len(clusters) >= 1
    assert clusters[0].keywords  # has some keywords


def test_cluster_ids_are_sequential():
    a = _analyzer()
    msgs = _make_msgs(6)
    clusters = a._cluster_messages(msgs)
    for i, c in enumerate(clusters):
        assert c.topic_id == f"topic_{i}"


def test_cluster_empty_window():
    a = _analyzer()
    assert a._cluster_messages([]) == []


def test_cluster_single_message():
    a = _analyzer()
    msgs = [Message(role="user", content="hello", sequence_index=0)]
    clusters = a._cluster_messages(msgs)
    assert len(clusters) == 1
    assert clusters[0].indices == [0]


def test_cluster_covers_all_indices():
    """Every index 0..n-1 must appear in exactly one cluster."""
    a = _analyzer()
    msgs = _make_msgs(10)
    clusters = a._cluster_messages(msgs)
    all_indices = sorted(i for c in clusters for i in c.indices)
    assert all_indices == list(range(10))
    # No duplicates
    assert len(all_indices) == len(set(all_indices))


# ── _build_overview ───────────────────────────────────────────────────────────

def test_build_overview_contains_topic_ids():
    a = _analyzer()
    msgs = _make_msgs(4)
    clusters = a._cluster_messages(msgs)
    overview = a._build_overview(msgs, "test query", clusters, set())
    for c in clusters:
        assert c.topic_id in overview


def test_build_overview_contains_current_question():
    a = _analyzer()
    msgs = _make_msgs(3)
    clusters = a._cluster_messages(msgs)
    overview = a._build_overview(msgs, "what is the answer?", clusters, set())
    assert "what is the answer?" in overview


def test_build_overview_auto_expanded_messages_shown():
    a = _analyzer()
    msgs = _make_msgs(4)
    clusters = a._cluster_messages(msgs)
    # Mark index 1 as auto-expanded
    overview = a._build_overview(msgs, "query", clusters, {1})
    # Index 1 content should appear in expanded section
    assert "[2]" in overview  # 1-based


def test_build_overview_label_count_in_footer():
    a = _analyzer()
    msgs = _make_msgs(5)
    clusters = a._cluster_messages(msgs)
    overview = a._build_overview(msgs, "q", clusters, set())
    assert "5 labels" in overview


# ── Integration: entry logic with mocked LLM ─────────────────────────────────


def _entry_analyzer(**kwargs) -> LLMContextAnalyzer:
    return LLMContextAnalyzer(
        api_base="https://api.example.com/v1",
        api_key="test",
        use_entry_logic=True,
        max_expansions=2,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_entry_logic_single_shot(respx_mock):
    """LLM responds directly with labels (no tool calls)."""
    import httpx

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"labels": ["K", "H", "M"]}', "tool_calls": None}}
                ]
            },
        )
    )

    a = _entry_analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "test query", conv)

    assert scores[msgs[0].id] == LABEL_WEIGHTS["K"]
    assert scores[msgs[1].id] == LABEL_WEIGHTS["H"]
    assert scores[msgs[2].id] == LABEL_WEIGHTS["M"]


@pytest.mark.asyncio
async def test_entry_logic_expand_then_classify(respx_mock):
    """LLM calls expand_topic once, then classifies on second call."""
    import httpx

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: LLM asks to expand topic_0
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "type": "function",
                                        "function": {
                                            "name": "expand_topic",
                                            "arguments": '{"topic_id": "topic_0"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        else:
            # Second call: classify after seeing expansion
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"labels": ["K", "H", "D"]}', "tool_calls": None}}
                    ]
                },
            )

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    a = _entry_analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "query", conv)

    assert call_count == 2
    assert scores[msgs[0].id] == LABEL_WEIGHTS["K"]
    assert scores[msgs[1].id] == LABEL_WEIGHTS["H"]
    assert scores[msgs[2].id] == LABEL_WEIGHTS["D"]


@pytest.mark.asyncio
async def test_entry_logic_unknown_topic_id_handled(respx_mock):
    """expand_topic with unknown topic_id returns error message, loop continues."""
    import httpx

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "tc1",
                                        "type": "function",
                                        "function": {
                                            "name": "expand_topic",
                                            "arguments": '{"topic_id": "nonexistent"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
            )
        else:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"labels": ["M", "M", "K"]}', "tool_calls": None}}
                    ]
                },
            )

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    a = _entry_analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "query", conv)

    assert call_count == 2
    assert scores[msgs[2].id] == LABEL_WEIGHTS["K"]


@pytest.mark.asyncio
async def test_entry_logic_respects_max_expansions(respx_mock):
    """After max_expansions tool calls, the loop sends a final no-tool request."""
    import httpx

    call_count = 0

    def side_effect(request):
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content)
        # If no tools offered, return labels; otherwise keep calling tool
        if "tools" not in body:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": '{"labels": ["K", "M", "D"]}', "tool_calls": None}}
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"tc{call_count}",
                                    "type": "function",
                                    "function": {
                                        "name": "expand_topic",
                                        "arguments": '{"topic_id": "topic_0"}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(side_effect=side_effect)

    # max_expansions=2 is already set in _entry_analyzer default
    a = _entry_analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "query", conv)

    # max_expansions=2 means iterations 0,1 offer tools; iteration 2 does not
    assert call_count == 3
    assert scores[msgs[0].id] == LABEL_WEIGHTS["K"]


@pytest.mark.asyncio
async def test_entry_logic_falls_back_on_failure(respx_mock):
    """LLM failure during entry logic → default weights."""
    import httpx

    respx_mock.post("https://api.example.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="Server error")
    )

    a = _entry_analyzer()
    msgs = _make_msgs(3)
    conv = Conversation(msgs)
    scores = await a.score(msgs, "query", conv)

    default_w = LABEL_WEIGHTS[DEFAULT_LABEL]
    for m in msgs:
        assert scores[m.id] == default_w
