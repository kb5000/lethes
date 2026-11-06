"""High-recall, path-local retrieval. Generated queries never become story facts."""

from __future__ import annotations

import math
import re
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Protocol

from .history import History, digest, text
from .state import Span

CONTINUATIONS = frozenset({"继续", "继续。", "continue", "go on", "好", "嗯", "yes"})


def terms(value: str) -> list[str]:
    result = re.findall(r"[a-z0-9_]+", value.casefold())
    for run in re.findall(r"[\u3400-\u9fff]+", value):
        result.extend(run)
        result.extend(run[i : i + 2] for i in range(len(run) - 1))
    return result


class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class QueryExpander(Protocol):
    async def expand(self, recent: str) -> list[str]: ...


@dataclass(frozen=True)
class Retrieval:
    spans: tuple[Span, ...]
    warnings: tuple[str, ...] = ()
    focuses: tuple[int, ...] = ()
    required: tuple[Span, ...] = ()
    scores: tuple[float, ...] = ()


class Retriever(Protocol):
    async def search(self, history: History, before: int) -> Retrieval: ...


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Embedding dimensions differ or are empty")
    numerator = sum(x * y for x, y in zip(a, b, strict=False))
    denominator = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return numerator / denominator if denominator else 0.0


def windows(value: str, size: int, overlap: int) -> list[str]:
    if not size or len(value) <= size:
        return [value]
    result = []
    for start in range(0, len(value), size - overlap):
        result.append(value[start : start + size])
        if start + size >= len(value):
            break
    return result


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        expander: QueryExpander | None = None,
        query_messages: int = 8,
        neighbors: int = 1,
        semantic_floor: float = 0.25,
        embedding_chunk_chars: int = 0,
        embedding_chunk_overlap: int = 64,
        embedding_query_prefix: str = "",
        embedding_document_prefix: str = "",
    ) -> None:
        if (
            embedding_chunk_chars < 0
            or embedding_chunk_overlap < 0
            or (embedding_chunk_chars and embedding_chunk_overlap >= embedding_chunk_chars)
            or not -1 <= semantic_floor <= 1
        ):
            raise ValueError("Invalid semantic window or similarity floor")
        self.embedder = embedder
        self.expander = expander
        self.query_messages = query_messages
        self.neighbors = neighbors
        self.semantic_floor = semantic_floor
        self.embedding_chunk_chars = embedding_chunk_chars
        self.embedding_chunk_overlap = embedding_chunk_overlap
        self.embedding_query_prefix = embedding_query_prefix
        self.embedding_document_prefix = embedding_document_prefix
        self._term_cache: OrderedDict[str, Counter[str]] = OrderedDict()

    def _terms(self, value: str) -> Counter[str]:
        key = digest(value)
        if key not in self._term_cache:
            self._term_cache[key] = Counter(terms(value))
            if len(self._term_cache) > 4096:
                self._term_cache.popitem(last=False)
        return self._term_cache[key]

    async def search(self, history: History, before: int) -> Retrieval:
        messages = history.messages
        first = history.instruction_count
        if before <= first:
            return Retrieval(())
        # Only the supplied path is indexed. Sibling/future messages never enter candidates.
        docs = [
            text(m) if text(m).strip().casefold() not in CONTINUATIONS else ""
            for m in messages[first:before]
        ]
        recent_messages = messages[max(first, before, len(messages) - self.query_messages) :]
        continuations = CONTINUATIONS
        recent = [
            text(m)
            for m in recent_messages
            if text(m).strip().casefold() not in continuations and text(m).strip()
        ]
        # Background context supports an explicit current question; its many separate
        # chunks must not outvote that question merely by producing more rankings.
        weights = {q: 0.5 for q in recent + ["\n".join(recent)] if q.strip()}
        latest_user = next((text(m) for m in reversed(recent_messages) if m["role"] == "user"), "")
        wants_earliest = bool(
            re.search(
                r"最初|最早|最开始|earliest|first mention|originally|at the beginning",
                latest_user,
                re.IGNORECASE,
            )
        )
        if latest_user.strip() and latest_user.strip().casefold() not in continuations:
            weights[latest_user] = 4.0
        warnings = []
        if self.expander and recent:
            try:
                for query in await self.expander.expand("\n".join(recent)):
                    if query.strip():
                        weights[query] = max(weights.get(query, 0), 2.0)
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append(f"query_expansion_failed:{type(exc).__name__}")
        queries = list(weights)
        if not queries:
            return Retrieval((), tuple(warnings))
        counts = [self._terms(d) for d in docs]
        df = Counter(term for count in counts for term in count)
        avg = sum(sum(c.values()) for c in counts) / max(1, len(counts)) or 1
        normalizers = [1.2 * (0.25 + 0.75 * sum(c.values()) / avg) for c in counts]
        idfs = {
            term: math.log(1 + (len(docs) - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()
        }
        ranks: dict[int, float] = {}
        primary_hits: list[int] = []
        for query in queries:
            query_terms = set(terms(query))
            scores = []
            for i, count in enumerate(counts):
                score = 0.0
                for term in query_terms.intersection(count):
                    freq = count[term]
                    score += idfs[term] * freq * 2.2 / (freq + normalizers[i])
                if score > 0:
                    scores.append((score, i))
            for rank, (_, index) in enumerate(sorted(scores, reverse=True)):
                if rank == 0 and query == latest_user:
                    primary_hits.append(index)
                ranks[index] = ranks.get(index, 0) + weights[query] / (20 + rank)
            if query == latest_user and wants_earliest and scores:
                # Temporal intent changes which original passage is needed, not its
                # factual interpretation. Keep the best match and an early strong match.
                floor = max(score for score, _ in scores) * (2 / 3)
                primary_hits.append(min(i for score, i in scores if score >= floor))
        if self.embedder:
            try:
                dense_docs = [
                    (i, window)
                    for i, doc in enumerate(docs)
                    if doc.strip()
                    for window in windows(
                        doc, self.embedding_chunk_chars, self.embedding_chunk_overlap
                    )
                ]
                dense_queries = [
                    (query, window)
                    for query in queries
                    for window in windows(
                        query, self.embedding_chunk_chars, self.embedding_chunk_overlap
                    )
                ]
                vectors = await self.embedder.embed(
                    [self.embedding_document_prefix + d for _, d in dense_docs]
                    + [self.embedding_query_prefix + q for _, q in dense_queries]
                )
                if len(vectors) != len(dense_docs) + len(dense_queries):
                    raise ValueError("Wrong embedding count")
                semantic_scores: dict[str, dict[int, float]] = {q: {} for q in queries}
                for (query, _), query_vector in zip(
                    dense_queries, vectors[len(dense_docs) :], strict=True
                ):
                    for (original_index, _), vector in zip(
                        dense_docs, vectors[: len(dense_docs)], strict=True
                    ):
                        score = cosine(vector, query_vector)
                        current = semantic_scores[query]
                        current[original_index] = max(current.get(original_index, -1.0), score)
                for query, by_message in semantic_scores.items():
                    # One vote per original message/query, regardless of window count.
                    scored = [(score, index) for index, score in by_message.items()]
                    for rank, (score, index) in enumerate(sorted(scored, reverse=True)):
                        if score >= self.semantic_floor:
                            if rank == 0 and query == latest_user:
                                primary_hits.append(index)
                            ranks[index] = ranks.get(index, 0) + weights[query] / (20 + rank)
            except (OSError, ValueError, RuntimeError) as exc:
                warnings.append(f"embedding_failed:{type(exc).__name__}")
        # Reserve first consideration for each channel's strongest direct-query
        # match. This changes ordering, not the recall set or a top-k cutoff.
        ordered = list(
            dict.fromkeys(
                primary_hits
                + [i for i, _ in sorted(ranks.items(), key=lambda item: (-item[1], item[0]))]
            )
        )
        spans = tuple(
            Span(
                max(first, first + i - self.neighbors), min(before, first + i + self.neighbors + 1)
            )
            for i in ordered
        )
        required = tuple(
            Span(
                max(first, first + i - self.neighbors), min(before, first + i + self.neighbors + 1)
            )
            for i in dict.fromkeys(primary_hits)
        )
        return Retrieval(
            spans,
            tuple(warnings),
            tuple(first + i for i in ordered),
            required,
            tuple(ranks[i] for i in ordered),
        )
