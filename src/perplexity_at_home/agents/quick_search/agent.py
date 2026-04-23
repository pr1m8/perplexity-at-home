"""Quick-search agent construction.

Purpose:
    Build the main quick-search agent using the Tavily quick tool bundle and the
    rich quick-search system prompt.

Design:
    - Uses the existing Tavily quick bundle from the shared tool layer.
    - Preserves the original rich quick-search prompt.
    - Registers runtime context, short-term state, and structured output.
    - Uses an in-memory checkpointer for short-term conversational continuity.

Attributes:
    build_quick_search_agent:
        Factory for the main quick-search agent instance.

Examples:
    .. code-block:: python

        agent = build_quick_search_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "What is Apple's current price?"}]},
            context=QuickSearchContext(),
        )
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.quick_search.context import QuickSearchContext
from perplexity_at_home.agents.quick_search.models import QuickSearchAnswer
from perplexity_at_home.agents.quick_search.prompts import build_quick_search_system_prompt
from perplexity_at_home.agents.quick_search.state import QuickSearchState
from perplexity_at_home.settings import get_settings, resolve_model
from perplexity_at_home.tools.tavily import build_quick_bundle


def build_quick_search_agent(model: str | None = None) -> object:
    """Build the main quick-search agent.

    The resulting agent is configured for fast, citation-aware Tavily-backed
    answers and is intended to be the first end-to-end implementation of the
    quick-search lane.

    Returns:
        object: A configured LangChain agent instance.

    Raises:
        RuntimeError: Propagated if agent construction fails because of invalid
            model, tool, or schema configuration.

    Examples:
        >>> agent = build_quick_search_agent()
        >>> agent is not None
        True
    """
    memory = MemorySaver()
    tools = list(build_quick_bundle().values())
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_quick_search_model)

    return create_agent(
        model=resolved_model,
        tools=tools,
        system_prompt=build_quick_search_system_prompt(),
        context_schema=QuickSearchContext,
        state_schema=QuickSearchState,
        checkpointer=memory,
        response_format=ToolStrategy(QuickSearchAnswer),
        name="quick_search_agent",
    )
