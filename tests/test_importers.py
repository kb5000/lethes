import pytest

from lethes.importers import open_webui_paths


def export():
    return [
        {
            "id": "chat",
            "chat": {
                "params": {"system": "Narrator"},
                "history": {
                    "currentId": "d",
                    "messages": {
                        "a": {
                            "id": "a",
                            "parentId": None,
                            "role": "user",
                            "content": "Begin",
                            "childrenIds": [],
                            "models": ["model"],
                            "user": {"email": "private"},
                        },
                        "b": {
                            "id": "b",
                            "parentId": "a",
                            "role": "assistant",
                            "content": "Old branch",
                        },
                        "c": {
                            "id": "c",
                            "parentId": "a",
                            "role": "assistant",
                            "content": "New branch",
                        },
                        "d": {"id": "d", "parentId": "c", "role": "user", "content": "Continue"},
                    },
                },
            },
        }
    ]


def test_export_follows_parent_links_not_dictionary_order_or_stale_children():
    paths = open_webui_paths(export())
    assert len(paths) == 2
    assert [m["content"] for m in paths[-1].messages] == [
        "Narrator",
        "Begin",
        "New branch",
        "Continue",
    ]
    assert paths[-1].active and not paths[0].active
    assert "Old branch" not in str(paths[-1].messages)
    assert "private" not in str(paths)
    assert len(open_webui_paths(export(), all_branches=False)) == 1


@pytest.mark.parametrize("mutation", ["missing-parent", "cycle", "missing-current"])
def test_malformed_export_is_rejected(mutation):
    data = export()
    history = data[0]["chat"]["history"]
    if mutation == "missing-parent":
        history["messages"]["d"]["parentId"] = "missing"
    elif mutation == "cycle":
        history["messages"]["a"]["parentId"] = "d"
    else:
        history["currentId"] = "missing"
    with pytest.raises(ValueError):
        open_webui_paths(data)


def test_flat_export_and_ambiguous_selection():
    paths = open_webui_paths({"messages": [{"role": "user", "content": "Hello"}]})
    assert paths[0].messages == [{"role": "user", "content": "Hello"}]
    data = export()
    del data[0]["chat"]["history"]["currentId"]
    with pytest.raises(ValueError, match="Ambiguous"):
        open_webui_paths(data)
