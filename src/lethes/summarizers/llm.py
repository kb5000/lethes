"""
LLMSummarizer — compress messages via any OpenAI-compatible chat API.

No vendor SDK required — uses raw ``httpx`` so it works with OpenAI,
Azure OpenAI, Anthropic (via proxy), Ollama, vLLM, etc.

Ported and generalised from ``example.py``'s ``llm_comp_text()`` and
``generate_summary()`` methods.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

import httpx

from ..observability import get_logger

if TYPE_CHECKING:
    from ..models.message import Message

logger = get_logger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are a context compression assistant. "
    "Given the conversation context and a message to compress, produce a concise summary "
    "that preserves all key information, decisions, and requests. "
    "Output the compressed text directly without any preamble. "
    "Target length: {target_pct}% of the original."
)

_DEFAULT_USER_PROMPT = (
    "Context:\n{context}\n\n"
    "Message to compress:\n{message}\n\n"
    "Compressed summary:"
)


class LLMSummarizer:
    """
    Summarise messages by calling an OpenAI-compatible ``/v1/chat/completions``
    endpoint.

    Parameters
    ----------
    api_base:
        Base URL, e.g. ``"https://api.openai.com/v1"``.
    api_key:
        Bearer token.
    model:
        Chat model name, e.g. ``"gpt-4o-mini"``.
    system_prompt:
        Override the default system prompt.  Use ``{target_pct}`` as placeholder.
    user_prompt:
        Override the default user prompt template.
        Placeholders: ``{context}``, ``{message}``.
    retry_attempts:
        Number of times to retry on failure.
    timeout:
        HTTP timeout in seconds.
    extra_body:
        Extra fields merged into the request JSON (e.g. ``{"thinking": {...}}``).
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        retry_attempts: int = 3,
        timeout: float = 80.0,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._user_prompt = user_prompt or _DEFAULT_USER_PROMPT
        self._retry_attempts = retry_attempts
        self._extra_body = extra_body or {}
        self._client = httpx.AsyncClient(timeout=timeout)

    async def summarize(
        self,
        messages: list["Message"],
        *,
        target_ratio: float = 0.3,
        context_messages: list["Message"] | None = None,
    ) -> str:
        from ..utils.content import get_text_content

        target_pct = int(target_ratio * 100)
        context_str = _build_context_string(context_messages or [])
        message_str = _build_message_string(messages)

        system = self._system_prompt.format(target_pct=target_pct)
        user = self._user_prompt.format(context=context_str, message=message_str)

        log = logger.bind(model=self._model, n_messages=len(messages))
        for attempt in range(self._retry_attempts):
            t0 = time.perf_counter()
            try:
                result = await self._call_api(system, user)
                log.debug("summarizer.done",
                    attempt=attempt + 1,
                    input_chars=len(message_str),
                    output_chars=len(result),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                return result
            except Exception as exc:
                log.warning("summarizer.attempt_failed",
                    attempt=attempt + 1,
                    max_attempts=self._retry_attempts,
                    error=str(exc),
                    elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                )
                if attempt < self._retry_attempts - 1:
                    await asyncio.sleep(1)

        log.error("summarizer.all_failed", attempts=self._retry_attempts)
        return "-"

    def name(self) -> str:
        return f"llm({self._model})"

    async def _call_api(self, system: str, user: str) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self._extra_body,
        }
        resp = await self._client.post(
            f"{self._api_base}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def _build_context_string(messages: list["Message"]) -> str:
    from ..utils.content import get_text_content

    parts = []
    for m in messages:
        parts.append(f"{m.role}: {get_text_content(m.content)}")
    return "\n\n".join(parts)


def _build_message_string(messages: list["Message"]) -> str:
    return _build_context_string(messages)
