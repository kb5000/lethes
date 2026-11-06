import httpx
import pytest

from lethes.history import History
from lethes.providers import OpenAIEmbeddings, OpenAIQueryExpander
from lethes.retrieval import HybridRetriever, terms


def path(*values):
    return History.from_messages(
        [
            {"role": "user" if i % 2 == 0 else "assistant", "content": v}
            for i, v in enumerate(values)
        ]
    )


async def test_chinese_names_without_spaces_and_neighbor_exception():
    history = path(
        "银月剑只有在下雨时才能出鞘。",
        "除非持有人献出一段记忆。",
        "旅店老板准备了晚餐。",
        "她握住银月剑的剑柄。",
    )
    result = await HybridRetriever(neighbors=1).search(history, before=3)
    assert result.spans[0].start == 0
    assert result.spans[0].end >= 2
    assert all(s.end <= 3 for s in result.spans)
    assert "银月" in terms("握住银月剑")


async def test_multiple_recent_messages_resolve_short_latest_reply():
    history = path(
        "The silver sword consumes memory.",
        "They discussed dinner.",
        "She prepares the silver sword.",
        "Yes.",
    )
    found = await HybridRetriever(neighbors=0).search(history, before=2)
    assert any(s.start == 0 for s in found.spans)


async def test_semantic_retrieval_can_find_no_keyword_overlap():
    class Semantic:
        async def embed(self, texts):
            return [[1.0, 0.0] if "healing" in t or "restores" in t else [0.0, 1.0] for t in texts]

    history = path(
        "Every healing costs a memory.", "Dinner is ready.", "She restores his wounded body."
    )
    found = await HybridRetriever(embedder=Semantic(), neighbors=0).search(history, 2)
    assert any(s.start == 0 for s in found.spans)


async def test_optional_helpers_fail_open_to_lexical_retrieval():
    class Broken:
        async def embed(self, texts):
            raise RuntimeError("offline")

        async def expand(self, recent):
            raise ValueError("malformed")

    history = path("silver sword", "dinner", "sword")
    result = await HybridRetriever(embedder=Broken(), expander=Broken()).search(history, 2)
    assert result.spans
    assert len(result.warnings) == 2


async def test_empty_query_and_no_past():
    assert not (await HybridRetriever().search(path(""), 0)).spans
    assert not (await HybridRetriever().search(path("something", ""), 1)).spans


async def test_embeddings_batch_reorder_cache_and_reject_missing_indices():
    calls = []

    def respond(request):
        payload = __import__("json").loads(request.content)
        calls.append(payload)
        data = [
            {"index": i, "embedding": [float(i + 1), 1.0]}
            for i in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(200, json={"data": data})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        helper = OpenAIEmbeddings("https://fake/v1", "secret", "embedding", client=client)
        a = await helper.embed(["a", "b", "a"])
        assert a[0] == a[2] and a[0] != a[1]
        assert await helper.embed(["a", "b"]) == a[:2]
        assert len(calls) == 1


async def test_provider_http_failure_is_safe_runtime_error():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(429, json={"error": "secret"}))
    ) as client:
        helper = OpenAIEmbeddings("https://fake/v1", "key", "embedding", client=client)
        with pytest.raises(RuntimeError, match="Embedding request failed"):
            await helper.embed(["text"])


async def test_query_helper_only_receives_recent_and_returns_bounded_queries():
    calls = []

    def respond(request):
        import json

        calls.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '["ability cost", "old promise"]'}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        helper = OpenAIQueryExpander("https://fake/v1", "key", "query", client)
        assert await helper.expand("she heals him") == ["ability cost", "old promise"]
        assert calls[0]["messages"][-1]["content"] == "she heals him"


async def test_current_question_not_outvoted_by_many_long_background_queries():
    old = [{"role": "assistant", "content": "Memory furnace converts minds into logic."}]
    old += [
        {"role": "assistant", "content": "Imperial council considers peace and trade." * 20}
        for _ in range(50)
    ]
    recent = [
        {"role": "assistant", "content": "Imperial council considers peace and trade." * 20}
        for _ in range(7)
    ]
    query = {"role": "user", "content": "How does the memory furnace convert minds?"}
    result = await HybridRetriever(neighbors=0).search(
        History.from_messages(old + recent + [query]), len(old)
    )
    assert result.spans[0].start == 0


async def test_generic_continue_messages_are_not_retrieval_centers():
    h = path("继续", "The sword consumes memory.", "继续", "She remembers the sword.", "继续")
    result = await HybridRetriever(neighbors=0).search(h, 3)
    assert result.spans
    assert all(s.start == 1 for s in result.spans)
