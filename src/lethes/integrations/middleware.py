"""
Generic async middleware shim — framework-agnostic message proxy.

Works with any system that passes an OpenAI-format ``messages`` list through
a callable chain.
"""

from __future__ import annotations

from typing import Any

from ..engine.orchestrator import ContextOrchestrator
from ..models.conversation import Conversation


class LethesMiddleware:
    """
    Thin async callable that wraps :class:`~lethes.engine.orchestrator.ContextOrchestrator`
    for use in arbitrary pipelines.

    Usage::

        middleware = LethesMiddleware(orchestrator=my_orchestrator)

        # In your pipeline:
        processed = await middleware(raw_messages, model_id="gpt-4o")
        # processed is a list[dict] in OpenAI format

    Or as a decorator factory::

        @middleware.wrap
        async def call_llm(messages, **kwargs):
            return await openai_client.chat.completions.create(
                model="gpt-4o", messages=messages, **kwargs
            )
    """

    def __init__(
        self,
        orchestrator: ContextOrchestrator,
        session_id: str | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._session_id = session_id

    async def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        model_id: str | None = None,
        session_id: str | None = None,
        event_emitter: Any = None,
    ) -> list[dict[str, Any]]:
        """
        Orchestrate *messages* and return the managed list.

        Parameters
        ----------
        messages:
            OpenAI-format message list.
        model_id:
            Model identifier — used for cost estimation if a pricing table is configured.
        session_id:
            Override the session ID for prefix tracking.
        event_emitter:
            Optional status callback (Open WebUI or custom).
        """
        sid = session_id or self._session_id
        conversation = Conversation.from_openai_messages(messages, session_id=sid)
        result = await self._orchestrator.process(
            conversation,
            model_id=model_id,
            event_emitter=event_emitter,
        )
        return result.conversation.to_openai_messages()

    def wrap(self, fn):
        """
        Decorator: automatically orchestrate the ``messages`` kwarg before
        calling *fn*.

        Example::

            @middleware.wrap
            async def send(messages, **kwargs):
                return await client.chat(..., messages=messages, **kwargs)
        """
        import functools

        @functools.wraps(fn)
        async def wrapper(*args, messages: list | None = None, **kwargs):
            if messages is not None:
                messages = await self(messages)
            return await fn(*args, messages=messages, **kwargs)

        return wrapper
