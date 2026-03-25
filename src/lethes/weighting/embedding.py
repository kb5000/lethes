"""
Embedding-based relevance scoring via cosine similarity.

Uses any OpenAI-compatible ``/v1/embeddings`` endpoint — works with
OpenAI, Azure OpenAI, Ollama, vLLM, etc.

Scores are cached by ``(message_content_hash, query_hash)`` so repeated
requests against unchanged history avoid re-embedding.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

import httpx

from ..utils.ids import cache_key_for_strings

if TYPE_CHECKING:
    from ..cache.base import CacheBackend
    from ..models.conversation import Conversation
    from ..models.message import Message

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "lethes:embed:"


class EmbeddingSimilarityStrategy:
    """
    Score messages by cosine similarity between their embedding and the
    embedding of the current query.

    Parameters
    ----------
    api_base:
        Base URL of an OpenAI-compatible API, e.g. ``"https://api.openai.com/v1"``.
    api_key:
        Bearer token.
    model:
        Embedding model name, e.g. ``"text-embedding-3-small"``.
    cache:
        Optional :class:`~lethes.cache.base.CacheBackend` for storing
        embeddings to avoid re-computing them on subsequent requests.
    topic_boost:
        If provided, a :class:`~lethes.weighting.base.TopicDetector` whose
        same-topic messages receive an additional score multiplier.
    topic_boost_factor:
        Multiplier applied to messages in the same topic cluster as the query.
    """

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str = "text-embedding-3-small",
        cache: "CacheBackend | None" = None,
        topic_boost: Any = None,  # TopicDetector | None
        topic_boost_factor: float = 1.3,
        timeout: float = 30.0,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._cache = cache
        self._topic_boost = topic_boost
        self._topic_boost_factor = topic_boost_factor
        self._client = httpx.AsyncClient(timeout=timeout)

    async def score(
        self,
        messages: list["Message"],
        query: str,
        conversation: "Conversation",
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        from ..utils.content import get_text_content

        if not messages or not query.strip():
            return {m.id: 1.0 for m in messages}

        # Gather texts to embed
        msg_texts = [get_text_content(m.content) for m in messages]

        # Fetch query embedding and all message embeddings (with caching)
        query_vec = await self._get_embedding(query)
        msg_vecs = await self._get_embeddings_batch(msg_texts)

        if query_vec is None:
            return {m.id: 1.0 for m in messages}

        # Cosine similarity
        raw_scores = []
        for vec in msg_vecs:
            if vec is None:
                raw_scores.append(0.5)  # neutral default for failed embeddings
            else:
                raw_scores.append(_cosine(query_vec, vec))

        # Shift from [-1,1] to [0,1] and normalise
        shifted = [(s + 1) / 2 for s in raw_scores]
        max_s = max(shifted) if shifted else 1.0
        if max_s == 0:
            normalised = [1.0] * len(shifted)
        else:
            normalised = [max(0.05, s / max_s) for s in shifted]

        # Apply optional topic boost
        scores: dict[str, float] = {}
        topic_map: dict[str, str] = {}
        if self._topic_boost is not None:
            try:
                topic_map = self._topic_boost.detect(conversation)
                # Find the topic of the last user message
                last_user = conversation.last_user_message()
                query_topic = topic_map.get(last_user.id) if last_user else None
            except Exception:
                query_topic = None
        else:
            query_topic = None

        for m, s in zip(messages, normalised):
            if query_topic and topic_map.get(m.id) == query_topic:
                s = min(1.0, s * self._topic_boost_factor)
            scores[m.id] = s

        return scores

    def name(self) -> str:
        return f"embedding({self._model})"

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _get_embedding(self, text: str) -> list[float] | None:
        key = _CACHE_PREFIX + cache_key_for_strings(self._model, text)
        if self._cache:
            cached = await self._cache.get(key)
            if cached:
                return json.loads(cached)

        vec = await self._embed_single(text)
        if vec is not None and self._cache:
            await self._cache.set(key, json.dumps(vec), ttl=3600)
        return vec

    async def _get_embeddings_batch(
        self, texts: list[str]
    ) -> list[list[float] | None]:
        """Batch embed; fall back to individual requests if batch fails."""
        results: list[list[float] | None] = [None] * len(texts)

        # Check cache first
        uncached_indices = []
        for i, text in enumerate(texts):
            key = _CACHE_PREFIX + cache_key_for_strings(self._model, text)
            if self._cache:
                cached = await self._cache.get(key)
                if cached:
                    results[i] = json.loads(cached)
                    continue
            uncached_indices.append(i)

        if not uncached_indices:
            return results

        # Batch call for uncached texts
        batch_texts = [texts[i] for i in uncached_indices]
        try:
            vecs = await self._embed_batch(batch_texts)
            for idx, vec in zip(uncached_indices, vecs):
                results[idx] = vec
                if vec is not None and self._cache:
                    key = _CACHE_PREFIX + cache_key_for_strings(self._model, texts[idx])
                    await self._cache.set(key, json.dumps(vec), ttl=3600)
        except Exception as exc:
            logger.warning("Batch embedding failed (%s), falling back to individual", exc)
            for idx in uncached_indices:
                results[idx] = await self._embed_single(texts[idx])

        return results

    async def _embed_single(self, text: str) -> list[float] | None:
        try:
            resp = await self._client.post(
                f"{self._api_base}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception as exc:
            logger.warning("Embedding request failed: %s", exc)
            return None

    async def _embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        resp = await self._client.post(
            f"{self._api_base}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self._model, "input": texts},
        )
        resp.raise_for_status()
        data = resp.json()
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
