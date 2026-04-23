"""State schema for the pro-search query-generation agent.

Purpose:
    Define dynamic short-term memory for the pro-search query agent.

Design:
    - State stores evolving working memory rather than static configuration.
    - The query generator is intentionally light on state because it mainly
      transforms the user's question into a structured search plan.
    - The built-in ``messages`` field remains available through ``AgentState``.

Attributes:
    QueryAgentStateBase:
        Shared dynamic state fields for query-generation flows.
    QueryAgentState:
        Main state schema for the pro-search query-generation agent.

Examples:
    .. code-block:: python

        state = {
            "messages": [],
            "original_question": "What changed recently in the Tavily LangChain integration?",
            "normalized_question": "Recent changes in Tavily's LangChain integration",
            "query_generation_notes": "Favor official sources and freshness.",
            "generated_query_count": 3,
        }
"""

from __future__ import annotations

from typing_extensions import NotRequired

from langchain.agents import AgentState


class QueryAgentStateBase(AgentState):
    """Base dynamic state for query-generation flows.

    Attributes:
        original_question:
            Original user question being transformed into a search plan.
        normalized_question:
            Canonical restatement of the question if produced before or during
            query generation.
        query_generation_notes:
            Optional notes about ambiguity, scope, or retrieval strategy.
    """

    original_question: NotRequired[str]
    normalized_question: NotRequired[str]
    query_generation_notes: NotRequired[str]


class QueryAgentState(QueryAgentStateBase):
    """Dynamic state for the main pro-search query-generation agent.

    Attributes:
        generated_query_count:
            Number of queries produced in the latest generation pass.
    """

    generated_query_count: NotRequired[int]