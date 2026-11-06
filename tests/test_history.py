from copy import deepcopy

import pytest

from lethes.history import History, normalize
from lethes.tokens import ByteTokenCounter, UnsupportedContent


def test_versions_include_content_role_and_ancestry():
    source = [
        {"role": "user", "content": "a", "id": "same"},
        {"role": "assistant", "content": "b", "id": "reply"},
    ]
    original = History.from_messages(source)
    changed = deepcopy(source)
    changed[0]["content"] = "edited"
    edited = History.from_messages(changed)
    assert original.heads[0] != edited.heads[0]
    assert original.heads[1] != edited.heads[1]
    changed = deepcopy(source)
    changed[-1]["role"] = "user"
    assert History.from_messages(changed).head != original.head


def test_transport_ids_and_usage_do_not_change_path():
    a = [{"role": "user", "content": "repeat", "id": "x"}]
    b = [{"role": "user", "content": "repeat", "id": "y", "usage": {"total": 9}}]
    assert History.from_messages(a).head == History.from_messages(b).head
    assert History.from_messages(a + a).heads[0] != History.from_messages(a + a).heads[1]


def test_detaches_nested_content():
    raw = {"role": "user", "content": [{"type": "text", "text": "before"}]}
    normalized = normalize(raw)
    raw["content"][0]["text"] = "after"
    assert normalized["content"][0]["text"] == "before"


@pytest.mark.parametrize("raw", [[], [{"role": "invalid", "content": "x"}]])
def test_invalid_history(raw):
    with pytest.raises(ValueError):
        History.from_messages(raw)


def test_images_are_not_silently_billed_as_url_text():
    with pytest.raises(UnsupportedContent):
        ByteTokenCounter().count(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.org/a.png"}}
                    ],
                }
            ]
        )


def test_provider_cache_controls_and_reasoning_extensions_are_preserved():
    raw = {
        "role": "assistant",
        "content": "text",
        "reasoning_details": [{"id": "r"}],
        "cache_control": {"type": "ephemeral"},
    }
    assert normalize(raw) == raw
