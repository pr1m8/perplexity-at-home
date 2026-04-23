"""Deep-research query-generation child agent.

Purpose:
    Build the structured-output child agent that converts planned subquestions
    into per-subquestion query plans for the parent deep-research workflow.

Design:
    - Uses the global ``DeepResearchContext`` as its runtime context schema.
    - Uses structured output and no tools.
    - Is intended to be called from a node in the parent ``deep_research`` graph.
    - Keeps all query-planning behavior in the prompt and response schema rather
      than in ad hoc post-processing logic.

Attributes:
    build_query_agent:
        Factory for the deep-research query-generation child agent.

Examples:
    .. code-block:: python

        agent = build_query_agent()
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.query_agent.models import (
    DeepResearchQueryPlans,
)
from perplexity_at_home.agents.deep_research.query_agent.prompts import (
    query_agent_prompt,
)
from perplexity_at_home.settings import get_settings, resolve_model


def build_query_agent(model: str | None = None) -> Any:
    """Build the deep-research query-generation child agent.

    Args:
        model: Model identifier used for structured query planning.

    Returns:
        A configured LangChain child agent.

    Raises:
        RuntimeError: Propagated if agent construction fails.

    Examples:
        >>> agent = build_query_agent()
        >>> agent is not None
        True
    """
    memory = MemorySaver()
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_deep_research_query_model)

    return create_agent(
        model=resolved_model,
        tools=[],
        middleware=[query_agent_prompt],
        context_schema=DeepResearchContext,
        checkpointer=memory,
        response_format=ToolStrategy(DeepResearchQueryPlans),
        name="deep_research_query_agent",
    )
