"""Top-level pro-search workflow package.

Purpose:
    Expose the main public API for the pro-search workflow, including the
    top-level runtime context, graph state, graph builder, and parent workflow
    wrapper.

Design:
    The top-level ``pro_search`` package owns the shared orchestration layer for
    pro-search. Child agent packages such as ``query_agent`` and
    ``answer_agent`` remain focused on their own prompts, models, and agent
    builders, while this package re-exports the global workflow surface.

Attributes:
    ProSearchContextBase:
        Shared runtime context fields for pro-search flows.
    ProSearchContext:
        Main runtime context for the top-level pro-search workflow.
    QueryExecutionRecord:
        Metadata describing one planned query/tool-call pairing.
    AggregatedResultRecord:
        Flattened normalized evidence record ready for later synthesis.
    ProSearchStateBase:
        Shared graph state fields for pro-search flows.
    ProSearchState:
        Main graph state schema for the pro-search workflow.
    build_pro_search_graph:
        Factory for the compiled pro-search retrieval graph.
    ProSearchAgent:
        Lightweight wrapper around the compiled pro-search graph.
    build_pro_search_agent:
        Factory for the top-level pro-search workflow wrapper.

Examples:
    .. code-block:: python

        from perplexity_at_home.agents.pro_search import (
            ProSearchContext,
            build_pro_search_agent,
        )

        agent = build_pro_search_agent(
            context=ProSearchContext(),
        )
"""

from __future__ import annotations

from perplexity_at_home.agents.pro_search.agent import (
    ProSearchAgent,
    build_pro_search_agent,
)
from perplexity_at_home.agents.pro_search.context import (
    ProSearchContext,
    ProSearchContextBase,
)
from perplexity_at_home.agents.pro_search.graph import build_pro_search_graph
from perplexity_at_home.agents.pro_search.state import (
    AggregatedResultRecord,
    ProSearchState,
    ProSearchStateBase,
    QueryExecutionRecord,
)

__all__ = [
    "AggregatedResultRecord",
    "ProSearchAgent",
    "ProSearchContext",
    "ProSearchContextBase",
    "ProSearchState",
    "ProSearchStateBase",
    "QueryExecutionRecord",
    "build_pro_search_agent",
    "build_pro_search_graph",
]