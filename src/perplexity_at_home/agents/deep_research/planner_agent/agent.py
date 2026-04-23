"""Deep-research planner child agent.

Purpose:
    Build the structured-output child agent that converts a user research
    request into a research brief and subquestion plan for the parent
    deep-research workflow.

Design:
    - Uses the global ``DeepResearchContext`` as its runtime context schema.
    - Uses structured output and no tools.
    - Is intended to be called from a node in the parent ``deep_research`` graph.
    - Keeps all planning behavior in the prompt and response schema rather than
      in ad hoc post-processing logic.

Attributes:
    build_planner_agent:
        Factory for the deep-research planner child agent.

Examples:
    .. code-block:: python

        agent = build_planner_agent()
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.planner_agent.models import PlannerOutput
from perplexity_at_home.agents.deep_research.planner_agent.prompts import planner_prompt
from perplexity_at_home.settings import get_settings, resolve_model


def build_planner_agent(
    model: str | None = None,
    *,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> Any:
    """Build the deep-research planner child agent.

    Args:
        model: Model identifier used for structured planning.

    Returns:
        A configured LangChain child agent.

    Raises:
        RuntimeError: Propagated if agent construction fails.

    Examples:
        >>> agent = build_planner_agent()
        >>> agent is not None
        True
    """
    resolved_checkpointer = checkpointer or MemorySaver()
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_deep_research_planner_model)

    return create_agent(
        model=resolved_model,
        tools=[],
        middleware=[planner_prompt],
        context_schema=DeepResearchContext,
        checkpointer=resolved_checkpointer,
        store=store,
        response_format=ToolStrategy(PlannerOutput),
        debug=debug,
        name="deep_research_planner_agent",
    )
