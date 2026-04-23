"""Deep-research reflection child agent.

Purpose:
    Build the structured-output child agent that evaluates current evidence and
    returns the next-step decision for the parent deep-research workflow.

Design:
    - Uses the global ``DeepResearchContext`` as its runtime context schema.
    - Uses structured output and no tools.
    - Is intended to be called from a node in the parent ``deep_research`` graph.
    - Keeps all reflection behavior in the prompt and response schema.

Attributes:
    build_reflection_agent:
        Factory for the deep-research reflection child agent.

Examples:
    .. code-block:: python

        agent = build_reflection_agent()
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.reflection_agent.models import (
    ReflectionDecision,
)
from perplexity_at_home.agents.deep_research.reflection_agent.prompts import (
    reflection_prompt,
)
from perplexity_at_home.settings import get_settings, resolve_model


def build_reflection_agent(
    model: str | None = None,
    *,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> Any:
    """Build the deep-research reflection child agent.

    Args:
        model: Model identifier used for structured reflection.

    Returns:
        A configured LangChain child agent.

    Raises:
        RuntimeError: Propagated if agent construction fails.

    Examples:
        >>> agent = build_reflection_agent()
        >>> agent is not None
        True
    """
    resolved_checkpointer = checkpointer or MemorySaver()
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_deep_research_reflection_model)

    return create_agent(
        model=resolved_model,
        tools=[],
        middleware=[reflection_prompt],
        context_schema=DeepResearchContext,
        checkpointer=resolved_checkpointer,
        store=store,
        response_format=ToolStrategy(ReflectionDecision),
        debug=debug,
        name="deep_research_reflection_agent",
    )
