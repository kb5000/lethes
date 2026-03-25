"""Tests for the flag parser and session flags."""

import pytest
from lethes.flags.parser import extract_flags
from lethes.flags.session import SessionFlags
from lethes.models.conversation import Conversation
from lethes.models.message import Message


def test_no_flags():
    flags, rest = extract_flags("hello world")
    assert flags == {}
    assert rest == "hello world"


def test_simple_flag():
    flags, rest = extract_flags("!full hello")
    assert "full" in flags
    assert rest == "hello"


def test_key_value_flag():
    flags, rest = extract_flags("!context=10 message")
    assert flags["context"] == "10"
    assert rest == "message"


def test_multiple_flags():
    flags, rest = extract_flags("!nosum,context=5 body text")
    assert "nosum" in flags
    assert flags["context"] == "5"
    assert rest == "body text"


def test_persistent_plus():
    flags, _ = extract_flags("!+pin message")
    assert "+pin" in flags


def test_persistent_minus():
    flags, _ = extract_flags("!-pin message")
    assert "-pin" in flags


def test_double_quoted_value():
    flags, rest = extract_flags('!key="hello world" body')
    assert flags["key"] == "hello world"
    assert rest == "body"


def test_no_body():
    flags, rest = extract_flags("!full")
    assert "full" in flags
    assert rest == ""


def test_session_flags_persistent():
    msgs = [
        Message(role="user", content="!+pin first message"),
        Message(role="assistant", content="response"),
        Message(role="user", content="second message"),
    ]
    conv = Conversation(msgs)
    session_flags, _ = SessionFlags.from_conversation(conv)
    assert "pin" in session_flags


def test_session_flags_temporary():
    msgs = [
        Message(role="user", content="!nosum only this turn"),
    ]
    conv = Conversation(msgs)
    session_flags, _ = SessionFlags.from_conversation(conv)
    assert "nosum" in session_flags


def test_session_flags_strips_content():
    msgs = [Message(role="user", content="!full actual message")]
    conv = Conversation(msgs)
    _, modified_conv = SessionFlags.from_conversation(conv)
    assert modified_conv.messages[0].content == "actual message"
