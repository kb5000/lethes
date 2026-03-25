"""
Keyword-based relevance scoring via BM25.

Requires the optional ``bm25`` extra::

    pip install lethes[bm25]   # installs rank_bm25

Falls back to a simple TF-IDF-like term overlap if ``rank_bm25`` is not
installed (lower accuracy but zero extra deps).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..models.conversation import Conversation
    from ..models.message import Message


def _default_tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class KeywordRelevanceStrategy:
    """
    Score each message by its BM25 relevance to the current query.

    When ``rank_bm25`` is installed the industry-standard Okapi BM25
    algorithm is used.  Otherwise a fast term-frequency overlap fallback
    is used.

    Parameters
    ----------
    tokenizer:
        Callable that converts a string to a list of tokens.
        Defaults to a simple lowercase + split tokeniser.
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._tokenize = tokenizer or _default_tokenize
        self._use_bm25 = _has_bm25()

    async def score(
        self,
        messages: list["Message"],
        query: str,
        conversation: "Conversation",
        context: dict[str, Any] | None = None,
    ) -> dict[str, float]:
        from ..utils.content import get_text_content

        if not messages:
            return {}

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return {m.id: 1.0 for m in messages}

        corpus = [self._tokenize(get_text_content(m.content)) for m in messages]

        if self._use_bm25:
            raw_scores = _bm25_scores(corpus, query_tokens)
        else:
            raw_scores = _overlap_scores(corpus, query_tokens)

        # Normalise to [0, 1] — keep a floor of 0.1 so no message gets
        # a weight of exactly 0 (it may still be somewhat relevant).
        max_score = max(raw_scores) if raw_scores else 1.0
        if max_score == 0:
            return {m.id: 1.0 for m in messages}

        return {
            m.id: max(0.1, s / max_score)
            for m, s in zip(messages, raw_scores)
        }

    def name(self) -> str:
        return "keyword_bm25" if self._use_bm25 else "keyword_overlap"


# ── BM25 and fallback implementations ────────────────────────────────────────

def _has_bm25() -> bool:
    try:
        import rank_bm25  # noqa: F401
        return True
    except ImportError:
        return False


def _bm25_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    from rank_bm25 import BM25Okapi  # type: ignore[import]

    bm25 = BM25Okapi(corpus)
    scores: list[float] = bm25.get_scores(query).tolist()
    return scores


def _overlap_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    """
    Simple TF overlap: score = number of query terms present in document,
    weighted by how rare the term is across the corpus (IDF-like).
    """
    n = len(corpus)
    query_set = set(query)

    # Document frequency
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1

    idf: dict[str, float] = {
        term: math.log((n + 1) / (df.get(term, 0) + 1)) + 1
        for term in query_set
    }

    scores = []
    for doc in corpus:
        doc_counts = Counter(doc)
        s = sum(
            (1 + math.log(doc_counts[t])) * idf.get(t, 1.0)
            for t in query_set
            if doc_counts[t] > 0
        )
        scores.append(s)
    return scores
