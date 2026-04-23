"""Deep-research retrieval child agent.

Purpose:
    Build the structured-output child agent that executes Tavily retrieval
    using the full V2 tool surface for the parent deep-research workflow.

Design:
    - Uses the global ``DeepResearchContext`` as its runtime context schema.
    - Uses structured output plus the full Tavily tool set.
    - Is intended to be called from a node in the parent ``deep_research`` graph.
    - Keeps retrieval behavior in the prompt and response schema rather than in
      ad hoc post-processing logic.

Attributes:
    build_retrieval_agent:
        Factory for the deep-research retrieval child agent.

Examples:
    .. code-block:: python

        agent = build_retrieval_agent()
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.retrieval_agent.models import (
    RetrievalAgentResult,
)
from perplexity_at_home.agents.deep_research.retrieval_agent.prompts import (
    retrieval_agent_prompt,
)
from perplexity_at_home.settings import get_settings
from perplexity_at_home.tools.tavily import (
    build_crawl_tool,
    build_extract_tool,
    build_get_research_tool,
    build_map_tool,
    build_research_tool,
    build_search_tool,
)


def build_retrieval_agent(
    model: str | None = None,
    *,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> Any:
    """Build the deep-research retrieval child agent.

    Args:
        model: Model identifier used for structured retrieval.

    Returns:
        Any: A configured LangChain child agent.

    Raises:
        RuntimeError: Propagated if agent construction fails.

    Examples:
        >>> agent = build_retrieval_agent()
        >>> agent is not None
        True
    """
    resolved_checkpointer = checkpointer or MemorySaver()
    settings = get_settings()
    resolved_model = settings.build_chat_model(
        settings.resolved_deep_research_retrieval_model,
        explicit_model=model,
    )

    tools = [
        build_search_tool(),
        build_extract_tool(),
        build_map_tool(),
        build_crawl_tool(),
        build_research_tool(),
        build_get_research_tool(),
    ]

    return create_agent(
        model=resolved_model,
        tools=tools,
        middleware=[retrieval_agent_prompt],
        context_schema=DeepResearchContext,
        checkpointer=resolved_checkpointer,
        store=store,
        response_format=ToolStrategy(RetrievalAgentResult),
        debug=debug,
        name="deep_research_retrieval_agent",
    )
