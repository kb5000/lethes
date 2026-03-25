"""
lethes — Constraint-driven LLM context management library.

Quick start::

    from lethes import ContextOrchestrator, Conversation, TokenBudget
    from lethes.algorithms import GreedyByWeightAlgorithm
    from lethes.weighting import KeywordRelevanceStrategy

    orchestrator = ContextOrchestrator(
        budget=TokenBudget(max_tokens=8000),
        algorithm=GreedyByWeightAlgorithm(),
        weighting=KeywordRelevanceStrategy(),
    )

    result = await orchestrator.process(
        Conversation.from_openai_messages(raw_messages)
    )
    ready_messages = result.conversation.to_openai_messages()

Open WebUI integration::

    from lethes.integrations.open_webui import OpenWebUIFilter as Filter
"""

from .engine.orchestrator import ContextOrchestrator, OrchestratorResult
from .models.budget import Budget, CompositeBudget, CostBudget, TokenBudget
from .models.conversation import Conversation
from .models.message import Message
from .models.pricing import ModelPricingEntry, ModelPricingTable

__all__ = [
    # Core
    "ContextOrchestrator",
    "Conversation",
    "Message",
    "OrchestratorResult",
    # Budgets
    "Budget",
    "CompositeBudget",
    "CostBudget",
    "TokenBudget",
    # Pricing
    "ModelPricingEntry",
    "ModelPricingTable",
]

__version__ = "0.1.0"
