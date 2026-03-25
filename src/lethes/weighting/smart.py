"""
SmartWeightingStrategy — recommended default for production use.

Combines three orthogonal signals into a single relevance score per message:

1. **Keyword relevance** — BM25 (or TF-IDF overlap) of message text vs. query.
   Same backend as :class:`~lethes.weighting.keyword.KeywordRelevanceStrategy`.

2. **Pair coherence** — an assistant reply inherits a fraction of its preceding
   user message's relevance score.  Keeps question-answer pairs together even
   when the answer itself shares no keywords with the new query.

3. **Role penalty** — tool-call intermediate messages (``role="tool"`` or
   ``tool_calls`` set) receive a configurable penalty multiplier, reflecting
   that they are rarely needed verbatim once the conversation moves on.

The per-turn ``context`` dict (passed by the orchestrator) can override any
parameter for a single turn:

* ``context["tool_penalty"]`` — float, overrides :attr:`tool_penalty`
* ``context["pair_coherence"]`` — float, overrides :attr:`pair_coherence`
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
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def _has_bm25() -> bool:
    try:
        import rank_bm25  # noqa: F401
        return True
    except ImportError:
        return False


class SmartWeightingStrategy:
    """
    Production-ready weighting that combines keyword relevance, pair coherence,
    and role-based penalties into a single score.

    Parameters
    ----------
    tool_penalty:
        Weight multiplier applied to tool-call intermediate messages
        (``role="tool"`` or assistant messages with ``tool_calls``).
        ``0.5`` means they compete at half the weight of regular messages.
        Set to ``1.0`` to disable the penalty.
    pair_coherence:
        When an assistant message follows a user message, the assistant's
        score is raised to at least ``pair_coherence × user_score``.
        Keeps Q&A pairs from being separated.  ``0.0`` disables this.
    keyword_floor:
        Minimum keyword score before the role penalty is applied.  Prevents
        highly relevant (but penalised) messages from being scored at zero.
    tokenizer:
        Optional custom tokeniser.  Defaults to lowercase-and-split.
    """

    def __init__(
        self,
        tool_penalty: float = 0.5,
        pair_coherence: float = 0.8,
        keyword_floor: float = 0.05,
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._tool_penalty = tool_penalty
        self._pair_coherence = pair_coherence
        self._keyword_floor = keyword_floor
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

        ctx = context or {}
        tool_penalty = float(ctx.get("tool_penalty", self._tool_penalty))
        pair_coherence = float(ctx.get("pair_coherence", self._pair_coherence))

        # ── Step 1: keyword relevance ─────────────────────────────────────
        query_tokens = self._tokenize(query)
        if not query_tokens:
            # No query → neutral scores (vary only by role penalty)
            kw_scores = [1.0] * len(messages)
        else:
            corpus = [self._tokenize(get_text_content(m.content)) for m in messages]
            if self._use_bm25:
                raw = _bm25_scores(corpus, query_tokens)
            else:
                raw = _overlap_scores(corpus, query_tokens)

            max_s = max(raw) if raw else 1.0
            if max_s == 0:
                kw_scores = [1.0] * len(messages)
            else:
                kw_scores = [max(self._keyword_floor, s / max_s) for s in raw]

        # ── Step 2: pair coherence ────────────────────────────────────────
        # If assistant[i] follows user[i-1] with a high score, lift assistant[i].
        coherent_scores = list(kw_scores)
        if pair_coherence > 0:
            for i, msg in enumerate(messages):
                if msg.role == "assistant" and i > 0:
                    prev = messages[i - 1]
                    if prev.role == "user":
                        floor = pair_coherence * coherent_scores[i - 1]
                        coherent_scores[i] = max(coherent_scores[i], floor)

        # ── Step 3: role penalty ──────────────────────────────────────────
        final: dict[str, float] = {}
        for msg, s in zip(messages, coherent_scores):
            is_tool_msg = msg.role == "tool" or bool(msg.tool_calls)
            multiplier = tool_penalty if is_tool_msg else 1.0
            final[msg.id] = min(1.0, s * multiplier)

        return final

    def name(self) -> str:
        bm25_label = "bm25" if self._use_bm25 else "overlap"
        return (
            f"smart({bm25_label},tool_penalty={self._tool_penalty},"
            f"pair_coherence={self._pair_coherence})"
        )


# ── Keyword scoring backends (shared with KeywordRelevanceStrategy) ────────────

def _bm25_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    from rank_bm25 import BM25Okapi  # type: ignore[import]
    bm25 = BM25Okapi(corpus)
    return bm25.get_scores(query).tolist()


def _overlap_scores(corpus: list[list[str]], query: list[str]) -> list[float]:
    """TF overlap with IDF-like term weighting."""
    n = len(corpus)
    query_set = set(query)
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
