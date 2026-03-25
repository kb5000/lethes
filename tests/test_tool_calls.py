"""Tests for tool call and multimodal message support."""

from __future__ import annotations

import pytest

from lethes.algorithms.greedy import GreedyByWeightAlgorithm
from lethes.engine.constraints import ConstraintChecker, ConstraintSet
from lethes.models.budget import TokenBudget
from lethes.models.conversation import Conversation, _link_tool_dependencies
from lethes.models.message import Message
from lethes.utils.content import get_text_content
from lethes.utils.tokens import TokenCounter


# ── Content extraction ────────────────────────────────────────────────────────

def test_get_text_content_none():
    assert get_text_content(None) == ""


def test_get_text_content_string():
    assert get_text_content("hello") == "hello"


def test_get_text_content_multimodal():
    content = [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
    ]
    assert get_text_content(content) == "describe this"


def test_get_text_content_multimodal_multiple_text_blocks():
    content = [
        {"type": "text", "text": "first"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        {"type": "text", "text": "second"},
    ]
    assert get_text_content(content) == "first\nsecond"


# ── Message.from_dict / to_dict round-trip ────────────────────────────────────

def test_from_dict_plain():
    msg = Message.from_dict({"role": "user", "content": "hi"})
    assert msg.role == "user"
    assert msg.content == "hi"
    assert msg.tool_calls is None
    assert msg.tool_call_id is None
    assert msg.name is None


def test_from_dict_tool_call_assistant():
    """Assistant message with tool_calls and null content."""
    raw = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ],
    }
    msg = Message.from_dict(raw)
    assert msg.role == "assistant"
    assert msg.content is None
    assert msg.tool_calls is not None
    assert msg.tool_calls[0]["id"] == "call_abc"
    assert msg.tool_call_id is None


def test_from_dict_tool_result():
    raw = {
        "role": "tool",
        "tool_call_id": "call_abc",
        "name": "get_weather",
        "content": "22°C, cloudy",
    }
    msg = Message.from_dict(raw)
    assert msg.role == "tool"
    assert msg.tool_call_id == "call_abc"
    assert msg.name == "get_weather"
    assert msg.content == "22°C, cloudy"
    assert msg.tool_calls is None


def test_to_dict_tool_call_assistant():
    msg = Message(
        role="assistant",
        content=None,
        tool_calls=[{"id": "call_abc", "type": "function", "function": {"name": "fn", "arguments": "{}"}}],
    )
    d = msg.to_dict()
    assert d["role"] == "assistant"
    assert d["content"] is None
    assert "tool_calls" in d
    assert d["tool_calls"][0]["id"] == "call_abc"
    assert "tool_call_id" not in d
    assert "name" not in d


def test_to_dict_tool_result():
    msg = Message(role="tool", content="ok", tool_call_id="call_abc", name="fn")
    d = msg.to_dict()
    assert d["role"] == "tool"
    assert d["content"] == "ok"
    assert d["tool_call_id"] == "call_abc"
    assert d["name"] == "fn"
    assert "tool_calls" not in d


def test_to_dict_plain_no_extra_keys():
    msg = Message(role="user", content="hello")
    d = msg.to_dict()
    assert set(d.keys()) == {"role", "content"}


def test_round_trip_multimodal():
    raw = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png", "detail": "high"}},
        ],
    }
    msg = Message.from_dict(raw)
    assert msg.to_dict() == raw


# ── get_text_content on Message ────────────────────────────────────────────────

def test_message_get_text_content_none():
    msg = Message(role="assistant", content=None, tool_calls=[])
    assert msg.get_text_content() == ""


def test_message_get_text_content_multimodal():
    msg = Message(
        role="user",
        content=[
            {"type": "text", "text": "hello"},
            {"type": "image_url", "image_url": {"url": "https://x.com/img.jpg"}},
        ],
    )
    assert msg.get_text_content() == "hello"


# ── Token counting ────────────────────────────────────────────────────────────

def test_token_count_includes_tool_calls():
    counter = TokenCounter()
    msg_with = Message(
        role="assistant",
        content=None,
        tool_calls=[{"id": "c1", "type": "function", "function": {"name": "fn", "arguments": '{"x":1}'}}],
    )
    msg_without = Message(role="assistant", content=None)
    assert counter.count(msg_with) > counter.count(msg_without)


def test_count_dict_includes_tool_calls():
    counter = TokenCounter()
    d_with = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "fn", "arguments": "{}"}}],
    }
    d_without = {"role": "assistant", "content": None}
    assert counter.count_dict(d_with) > counter.count_dict(d_without)


# ── Conversation.from_openai_messages ─────────────────────────────────────────

TOOL_CALL_CONVERSATION = [
    {"role": "user", "content": "What's the weather in Paris?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_xyz",
        "name": "get_weather",
        "content": "22°C, partly cloudy",
    },
    {"role": "assistant", "content": "It's 22°C and partly cloudy in Paris."},
]


def test_from_openai_messages_preserves_tool_fields():
    conv = Conversation.from_openai_messages(TOOL_CALL_CONVERSATION)
    msgs = list(conv.messages)
    assert msgs[1].tool_calls is not None
    assert msgs[2].tool_call_id == "call_xyz"
    assert msgs[2].name == "get_weather"


def test_from_openai_messages_links_dependencies():
    conv = Conversation.from_openai_messages(TOOL_CALL_CONVERSATION)
    msgs = list(conv.messages)
    assistant_tool_call = msgs[1]
    tool_result = msgs[2]

    # tool result depends on assistant
    assert assistant_tool_call.id in tool_result.dependencies
    # assistant depends on tool result
    assert tool_result.id in assistant_tool_call.dependencies


def test_to_openai_messages_round_trip():
    conv = Conversation.from_openai_messages(TOOL_CALL_CONVERSATION)
    output = conv.to_openai_messages()
    assert output[1]["content"] is None
    assert output[1]["tool_calls"][0]["id"] == "call_xyz"
    assert output[2]["tool_call_id"] == "call_xyz"
    assert output[2]["name"] == "get_weather"
    assert "tool_calls" not in output[2]
    assert "tool_call_id" not in output[1]


# ── Constraint repair keeps tool pairs together ───────────────────────────────

def test_constraint_repair_promotes_tool_call_dependency():
    """When a tool result is kept but the assistant tool_calls msg is dropped,
    the repair step must promote the assistant message."""
    conv = Conversation.from_openai_messages(TOOL_CALL_CONVERSATION)
    msgs = list(conv.messages)
    # msgs[1] = assistant tool_calls, msgs[2] = tool result
    asst_id = msgs[1].id
    tool_id = msgs[2].id

    from lethes.algorithms.base import SelectionResult

    # Simulate: algorithm dropped the assistant tool_calls msg but kept the tool result
    result = SelectionResult(
        keep_full=[msgs[0].id, tool_id, msgs[3].id],
        summarize=[],
        drop=[asst_id],
        estimated_tokens=100,
    )
    checker = ConstraintChecker()
    repaired = checker.repair(result, conv, ConstraintSet())

    assert asst_id in repaired.keep_full, "assistant tool_calls msg must be promoted"
    assert tool_id in repaired.keep_full


def test_constraint_repair_promotes_tool_result_dependency():
    """When the assistant tool_calls msg is kept but the tool result is dropped,
    the tool result must be promoted."""
    conv = Conversation.from_openai_messages(TOOL_CALL_CONVERSATION)
    msgs = list(conv.messages)
    asst_id = msgs[1].id
    tool_id = msgs[2].id

    from lethes.algorithms.base import SelectionResult

    result = SelectionResult(
        keep_full=[msgs[0].id, asst_id, msgs[3].id],
        summarize=[],
        drop=[tool_id],
        estimated_tokens=100,
    )
    checker = ConstraintChecker()
    repaired = checker.repair(result, conv, ConstraintSet())

    assert tool_id in repaired.keep_full, "tool result must be promoted"
    assert asst_id in repaired.keep_full


# ── Greedy algorithm never summarizes tool pairs ──────────────────────────────

def test_greedy_drops_not_summarizes_tool_messages():
    """Tool messages with pre-computed summaries should still be dropped (not
    placed in summarize) when they overflow the budget."""
    conv = Conversation.from_openai_messages(TOOL_CALL_CONVERSATION)
    # Pre-set summaries on tool call messages
    import dataclasses
    msgs = [
        dataclasses.replace(m, summary="summary") if m.tool_calls or m.role == "tool" else m
        for m in conv.messages
    ]
    conv = conv.with_messages(msgs)

    # Very tight budget — should force drops
    budget = TokenBudget(max_tokens=15)
    algo = GreedyByWeightAlgorithm(prefer_summarize=True)
    result = algo.select(conv, budget, ConstraintSet(), TokenCounter())

    # Tool messages in drop set must not appear in summarize
    tool_ids = {m.id for m in conv.messages if m.role == "tool" or m.tool_calls}
    assert not (tool_ids & set(result.summarize)), (
        "tool/tool_calls messages must not be placed in summarize"
    )


# ── link_tool_dependencies edge cases ─────────────────────────────────────────

def test_link_tool_dependencies_no_tools():
    msgs = [
        Message(role="user", content="hi", sequence_index=0),
        Message(role="assistant", content="hello", sequence_index=1),
    ]
    result = _link_tool_dependencies(msgs)
    assert result[0].dependencies == []
    assert result[1].dependencies == []


def test_link_tool_dependencies_unmatched_tool_call_id():
    """A tool result with a call_id that has no matching assistant is left as-is."""
    msgs = [
        Message(role="tool", content="result", tool_call_id="unknown_id", sequence_index=0),
    ]
    result = _link_tool_dependencies(msgs)
    assert result[0].dependencies == []


def test_link_tool_dependencies_multiple_parallel_tool_calls():
    """One assistant message with two parallel tool calls → two tool results."""
    msgs = [
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
                {"id": "c2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
            ],
            sequence_index=0,
        ),
        Message(role="tool", content="r1", tool_call_id="c1", sequence_index=1),
        Message(role="tool", content="r2", tool_call_id="c2", sequence_index=2),
    ]
    result = _link_tool_dependencies(msgs)
    asst, r1, r2 = result
    assert r1.id in asst.dependencies
    assert r2.id in asst.dependencies
    assert asst.id in r1.dependencies
    assert asst.id in r2.dependencies
