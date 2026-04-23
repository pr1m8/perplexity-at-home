"""State schemas for the pro-search workflow.

Purpose:
    Define shared graph state for the deterministic pro-search retrieval and
    answer-synthesis workflow.

Design:
    The current version of the pro-search graph focuses on:
    query generation -> batched search-tool execution -> result aggregation ->
    answer synthesis.

    The state uses a message channel with the ``add_messages`` reducer so tool
    call messages and tool result messages remain valid for LangGraph workflows.

Attributes:
    QueryExecutionRecord:
        Metadata for one planned query/tool-call pairing.
    AggregatedResultRecord:
        Flattened normalized evidence record ready for synthesis.
    ProSearchStateBase:
        Shared state fields for the pro-search workflow.
    ProSearchState:
        Main state schema for the top-level pro-search workflow.

Examples:
    .. code-block:: python

        state = {
            "messages": [{"role": "user", "content": "What changed recently in Tavily LangChain?"}],
            "user_question": "What changed recently in Tavily LangChain?",
            "query_plan": None,
            "planned_queries": [],
            "raw_query_results": [],
            "aggregated_results": [],
            "search_errors": [],
            "final_answer": None,
            "is_complete": False,
        }
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class QueryExecutionRecord(TypedDict, total=False):
    """Metadata describing one planned query/tool-call pairing."""

    tool_call_id: str
    query: str
    priority: int
    intent: str
    rationale: str
    target_topic: str
    prefer_recent_sources: bool
    preferred_source_types: list[str]


class AggregatedResultRecord(TypedDict, total=False):
    """Flattened normalized evidence record ready for later synthesis."""

    query: str
    query_priority: int
    query_intent: str
    query_rationale: str
    query_topic: str
    query_answer: str | None
    url: str
    title: str
    content: str
    score: float | None
    raw_content: str | None


class ProSearchStateBase(TypedDict, total=False):
    """Base shared graph state for pro-search flows.

    Attributes:
        messages:
            Message history for the workflow. This channel uses the
            ``add_messages`` reducer so AI tool-call messages and ToolMessages
            are accumulated as valid graph state.
        user_question:
            Original user question for the pro-search run.
        normalized_question:
            Canonical restatement of the question if produced by the query agent.
        query_plan:
            Structured query plan returned by the query-generation child agent.
        planned_queries:
            Planned query/tool-call metadata derived from the query plan.
        raw_query_results:
            Per-query raw Tavily payloads and associated metadata before final
            aggregation.
        aggregated_results:
            Flattened and normalized evidence items ready for synthesis.
        search_errors:
            Non-fatal execution errors encountered while running batch search.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_question: str
    normalized_question: str
    query_plan: dict[str, Any]
    planned_queries: list[QueryExecutionRecord]
    raw_query_results: list[dict[str, Any]]
    aggregated_results: list[AggregatedResultRecord]
    search_errors: list[str]


class ProSearchState(ProSearchStateBase, total=False):
    """Main graph state for the top-level pro-search workflow.

    Attributes:
        query_count:
            Number of queries contained in the latest query plan.
        search_tool_calls_built:
            Whether tool-call messages have already been built for the current plan.
        is_complete:
            Whether the workflow has completed and a final answer is available.
        final_answer:
            Structured final answer produced by the answer child agent.
    """

    query_count: int
    search_tool_calls_built: bool
    is_complete: bool
    final_answer: dict[str, Any]