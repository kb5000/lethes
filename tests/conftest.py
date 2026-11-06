import pytest

from lethes import ContextConfig, ContextManager, Retrieval, Span


def story(turns=50):
    messages = [{"role": "system", "content": "Continue the fictional story."}]
    for i in range(turns):
        messages.extend(
            [
                {
                    "role": "user",
                    "content": f"Scene {i}: " + "The traveler walks through the forest. " * 3,
                },
                {
                    "role": "assistant",
                    "content": f"Reply {i}: " + "The wind carries a distant song. " * 3,
                },
            ]
        )
    messages.append({"role": "user", "content": "She reaches for the silver sword."})
    return messages


class ScriptedRetriever:
    def __init__(self, spans=()):
        self.spans = spans
        self.calls = 0

    async def search(self, history, before):
        self.calls += 1
        return Retrieval(tuple(s for s in self.spans if s.end <= before))


@pytest.fixture
def retriever():
    return ScriptedRetriever((Span(1, 3),))


@pytest.fixture
def manager(retriever):
    return ContextManager(
        ContextConfig(
            context_window=6500,
            reply_reserve=500,
            safety_reserve=200,
            recent_tokens=650,
            minimum_recent_turns=1,
            rebuild_fill=0.7,
        ),
        retriever=retriever,
    )


async def prepare(manager, messages, **kwargs):
    return await manager.prepare(messages, user_id="u", chat_id="chat", **kwargs)
