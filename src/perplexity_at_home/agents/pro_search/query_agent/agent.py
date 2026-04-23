"""Pro-search query-generation agent construction.

Purpose:
    Build the structured-output query-generation agent used inside the parent
    pro-search flow.

Design:
    - Uses ``create_agent`` with no tools.
    - Returns a validated ``ProSearchQueryPlan`` object.
    - Registers runtime context and lightweight state schemas.
    - Uses dynamic prompt middleware so the current datetime and query-budget
      settings are injected at invocation time rather than frozen at build time.

Attributes:
    build_query_generator_agent:
        Factory for the pro-search query-generation agent.

Examples:
    .. code-block:: python

        agent = build_query_generator_agent()
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.pro_search.query_agent.context import QueryAgentContext
from perplexity_at_home.agents.pro_search.query_agent.models import ProSearchQueryPlan
from perplexity_at_home.agents.pro_search.query_agent.prompts import query_generator_prompt
from perplexity_at_home.agents.pro_search.query_agent.state import QueryAgentState
from perplexity_at_home.settings import get_settings, resolve_model


def build_query_generator_agent(model: str | None = None) -> Any:
    """Build the pro-search query-generation agent.

    Args:
        model: Model identifier used for the structured query-generation agent.

    Returns:
        Any: A configured LangChain agent instance for structured query generation.

    Raises:
        RuntimeError: Propagated if agent construction fails because of invalid
            model or schema configuration.

    Examples:
        >>> agent = build_query_generator_agent()
        >>> agent is not None
        True
    """
    memory = MemorySaver()
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_pro_search_query_model)

    return create_agent(
        model=resolved_model,
        tools=[],
        middleware=[query_generator_prompt],
        context_schema=QueryAgentContext,
        state_schema=QueryAgentState,
        checkpointer=memory,
        response_format=ToolStrategy(ProSearchQueryPlan),
        name="pro_search_query_generator_agent",
    )
