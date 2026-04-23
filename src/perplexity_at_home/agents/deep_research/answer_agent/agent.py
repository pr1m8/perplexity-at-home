"""Deep-research answer-synthesis child agent.

Purpose:
    Build the structured-output child agent that converts the accumulated
    deep-research state into a final markdown report.

Design:
    - Uses the global ``DeepResearchContext`` as its runtime context schema.
    - Uses structured output and no tools.
    - Is intended to be called from a node in the parent ``deep_research`` graph.
    - Keeps synthesis behavior in the prompt and response schema.

Attributes:
    build_answer_agent:
        Factory for the deep-research answer child agent.

Examples:
    .. code-block:: python

        agent = build_answer_agent()
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.deep_research.answer_agent.models import (
    DeepResearchAnswer,
)
from perplexity_at_home.agents.deep_research.answer_agent.prompts import (
    answer_prompt,
)
from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.settings import get_settings, resolve_model


def build_answer_agent(model: str | None = None) -> Any:
    """Build the deep-research answer child agent.

    Args:
        model: Model identifier used for structured synthesis.

    Returns:
        Any: A configured LangChain child agent.

    Raises:
        RuntimeError: Propagated if agent construction fails.
    """
    memory = MemorySaver()
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_deep_research_answer_model)

    return create_agent(
        model=resolved_model,
        tools=[],
        middleware=[answer_prompt],
        context_schema=DeepResearchContext,
        checkpointer=memory,
        response_format=ToolStrategy(DeepResearchAnswer),
        name="deep_research_answer_agent",
    )
