"""Cost estimation — combine token counts with the pricing table."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.conversation import Conversation
    from ..models.pricing import ModelPricingTable
    from ..utils.tokens import TokenCounter


class CostEstimator:
    """
    Estimates the USD cost of sending a :class:`~lethes.models.conversation.Conversation`
    to a model, optionally accounting for prefix-cache savings.
    """

    def __init__(
        self,
        pricing_table: "ModelPricingTable",
        token_counter: "TokenCounter",
    ) -> None:
        self._pricing = pricing_table
        self._counter = token_counter

    def estimate(
        self,
        conversation: "Conversation",
        model_id: str,
        cached_tokens: int = 0,
        expected_output_tokens: int = 500,
    ) -> float:
        """
        Return estimated cost in USD.

        Parameters
        ----------
        cached_tokens:
            Number of input tokens expected to be prefix-cache hits.
        expected_output_tokens:
            Rough estimate of the model's response length (default 500 tokens).
        """
        total_input = conversation.total_tokens(self._counter)
        return self._pricing.estimate_cost(
            model_id,
            input_tokens=total_input,
            cached_tokens=cached_tokens,
            output_tokens=expected_output_tokens,
        )

    def token_count(self, conversation: "Conversation") -> int:
        return conversation.total_tokens(self._counter)
