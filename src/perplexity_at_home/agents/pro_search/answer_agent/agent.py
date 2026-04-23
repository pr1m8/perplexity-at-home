"""Pro-search answer-synthesis child agent.

Purpose:
    Build the structured-output child agent that converts aggregated evidence
    into a final markdown answer for the parent pro-search workflow.

Design:
    - Uses the global ``ProSearchContext`` as its runtime context schema.
    - Uses structured output and no tools.
    - Is intended to be called from a node in the parent ``pro_search`` graph
      after evidence aggregation.
    - Keeps all synthesis behavior in the prompt and response schema rather
      than in ad hoc post-processing logic.

Attributes:
    build_answer_agent:
        Factory for the pro-search answer-synthesis child agent.

Examples:
    .. code-block:: python

        agent = build_answer_agent()
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Question: ...\\n\\nEvidence JSON:\\n{...}",
                    }
                ]
            },
            context=ProSearchContext(),
        )
"""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import MemorySaver

from perplexity_at_home.agents.pro_search.answer_agent.models import ProSearchAnswer
from perplexity_at_home.agents.pro_search.answer_agent.prompts import answer_agent_prompt
from perplexity_at_home.agents.pro_search.context import ProSearchContext
from perplexity_at_home.settings import get_settings, resolve_model


def build_answer_agent(
    model: str | None = None,
    *,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> Any:
    """Build the pro-search answer-synthesis child agent.

    Args:
        model: Model identifier used for structured answer synthesis.

    Returns:
        Any: A configured LangChain child agent.

    Raises:
        RuntimeError: Propagated if agent construction fails because of invalid
            model or schema configuration.

    Examples:
        >>> agent = build_answer_agent()
        >>> agent is not None
        True
    """
    resolved_checkpointer = checkpointer or MemorySaver()
    settings = get_settings()
    resolved_model = resolve_model(model, settings.resolved_pro_search_answer_model)

    return create_agent(
        model=resolved_model,
        tools=[],
        middleware=[answer_agent_prompt],
        context_schema=ProSearchContext,
        checkpointer=resolved_checkpointer,
        store=store,
        response_format=ToolStrategy(ProSearchAnswer),
        debug=debug,
        name="pro_search_answer_agent",
    )
