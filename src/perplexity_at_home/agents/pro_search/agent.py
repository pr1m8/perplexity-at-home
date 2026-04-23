"""Top-level pro-search workflow wrapper.

Purpose:
    Provide a small, organized wrapper around the compiled pro-search graph so
    the workflow can be invoked as a coherent top-level object.

Design:
    - Builds and owns the compiled pro-search graph.
    - Accepts a ``ProSearchContext`` object at construction time.
    - Exposes sync and async invocation helpers.
    - Runs the full flow: planning, batched search execution, aggregation,
      and answer synthesis.

Attributes:
    ProSearchAgent:
        Lightweight wrapper around the compiled pro-search graph.
    build_pro_search_agent:
        Factory for the top-level pro-search workflow wrapper.

Examples:
    .. code-block:: python

        agent = build_pro_search_agent()
        result = agent.invoke(
            "What changed recently in the Tavily LangChain integration?"
        )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from perplexity_at_home.agents.pro_search.context import ProSearchContext
from perplexity_at_home.agents.pro_search.graph import build_pro_search_graph


@dataclass(slots=True, kw_only=True)
class ProSearchAgent:
    """Lightweight wrapper around the compiled pro-search graph."""

    context: ProSearchContext
    graph: Any

    def invoke(self, user_question: str) -> dict[str, Any]:
        """Run the pro-search workflow synchronously."""
        if not user_question.strip():
            raise ValueError("The user question must not be empty.")

        return self.graph.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ],
                "user_question": user_question,
                "planned_queries": [],
                "raw_query_results": [],
                "aggregated_results": [],
                "search_errors": [],
                "search_tool_calls_built": False,
                "is_complete": False,
            },
            context=self.context,
            config={"configurable": {"thread_id": self.context.thread_id}},
        )

    async def ainvoke(self, user_question: str) -> dict[str, Any]:
        """Run the pro-search workflow asynchronously."""
        if not user_question.strip():
            raise ValueError("The user question must not be empty.")

        return await self.graph.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ],
                "user_question": user_question,
                "planned_queries": [],
                "raw_query_results": [],
                "aggregated_results": [],
                "search_errors": [],
                "search_tool_calls_built": False,
                "is_complete": False,
            },
            context=self.context,
            config={"configurable": {"thread_id": self.context.thread_id}},
        )


def build_pro_search_agent(
    *,
    context: ProSearchContext | None = None,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> ProSearchAgent:
    """Build the top-level pro-search workflow wrapper."""
    resolved_context = context or ProSearchContext()
    graph = build_pro_search_graph(
        checkpointer=checkpointer,
        store=store,
        debug=debug,
    )
    return ProSearchAgent(
        context=resolved_context,
        graph=graph,
    )
