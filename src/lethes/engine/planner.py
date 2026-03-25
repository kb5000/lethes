"""ContextPlan — the immutable intermediate representation between selection and execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextPlan:
    """
    Captures the full decision made for one orchestration pass.

    Included in :class:`~lethes.engine.orchestrator.OrchestratorResult` for
    logging, dry-run/explain mode, and testing.
    """

    keep_full: frozenset[str]
    """Message IDs to include at full content."""

    summarize: frozenset[str]
    """Message IDs to include at compressed content."""

    drop: frozenset[str]
    """Message IDs excluded from the outgoing context."""

    algorithm_name: str
    """Name of the algorithm that produced this plan."""

    weighting_strategy_name: str
    """Name of the weighting strategy used."""

    pre_plan_tokens: int
    """Total tokens in the conversation before selection."""

    post_plan_tokens: int
    """Estimated tokens after applying this plan (full + summary tokens)."""

    @classmethod
    def from_selection_result(
        cls,
        result: "SelectionResult",  # type: ignore[name-defined]  # noqa: F821
        algorithm_name: str,
        weighting_strategy_name: str,
        pre_plan_tokens: int,
    ) -> "ContextPlan":
        from ..algorithms.base import SelectionResult  # local import to avoid circular

        return cls(
            keep_full=frozenset(result.keep_full),
            summarize=frozenset(result.summarize),
            drop=frozenset(result.drop),
            algorithm_name=algorithm_name,
            weighting_strategy_name=weighting_strategy_name,
            pre_plan_tokens=pre_plan_tokens,
            post_plan_tokens=result.estimated_tokens,
        )

    def summarize_groups(self) -> list[list[str]]:
        """
        Group consecutive message IDs in ``summarize`` into contiguous segments
        (useful for batching summarisation calls).
        This requires the original message order — the orchestrator uses this.
        Returns a flat list here; grouping by position is done by the orchestrator.
        """
        return [list(self.summarize)]

    @property
    def total_kept(self) -> int:
        return len(self.keep_full) + len(self.summarize)

    @property
    def total_dropped(self) -> int:
        return len(self.drop)
