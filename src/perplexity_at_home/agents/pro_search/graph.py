"""LangGraph workflow for the pro-search graph.

Purpose:
    Build the deterministic pro-search graph:
    query generation -> batched Tavily execution via ToolNode ->
    aggregation -> answer synthesis.

Design:
    - Uses ``StateGraph`` with explicit workflow nodes.
    - Uses the structured-output query-generation child agent for planning.
    - Uses a deterministic node to convert the query plan into batched Tavily
      tool calls.
    - Uses ``ToolNode`` to execute Tavily search calls in parallel.
    - Uses a deterministic node to aggregate normalized evidence.
    - Uses the structured-output answer child agent to synthesize the final
      markdown answer.
    - Serializes the answer-agent input payload into a string before placing it
      into ``messages`` so it remains valid LangChain message content.

Attributes:
    build_pro_search_graph:
        Build the compiled pro-search workflow graph.

Examples:
    .. code-block:: python

        from perplexity_at_home.agents.pro_search.context import ProSearchContext
        from perplexity_at_home.agents.pro_search.graph import build_pro_search_graph

        context = ProSearchContext()
        graph = build_pro_search_graph()

        result = graph.invoke(
            {
                "messages": [
                    {"role": "user", "content": "What changed recently in Tavily LangChain?"}
                ],
                "user_question": "What changed recently in Tavily LangChain?",
                "planned_queries": [],
                "raw_query_results": [],
                "aggregated_results": [],
                "search_errors": [],
                "search_tool_calls_built": False,
                "is_complete": False,
            },
            context=context,
            config={"configurable": {"thread_id": "pro-search"}},
        )
"""

from __future__ import annotations

import ast
import json
from typing import Any, Protocol
from uuid import uuid4

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from perplexity_at_home.agents.pro_search.answer_agent import build_answer_agent
from perplexity_at_home.agents.pro_search.context import ProSearchContext
from perplexity_at_home.agents.pro_search.query_agent import build_query_generator_agent
from perplexity_at_home.agents.pro_search.state import (
    AggregatedResultRecord,
    ProSearchState,
    QueryExecutionRecord,
)
from perplexity_at_home.tools.tavily import build_pro_bundle
from perplexity_at_home.tools.tavily.normalize import (
    extract_answer,
    normalize_search_payload,
)


class SupportsInvoke(Protocol):
    """Protocol for child agents invoked by the top-level graph."""

    def invoke(
        self,
        input: dict[str, Any],
        *,
        context: ProSearchContext,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Invoke the child agent."""


def _extract_latest_user_question(state: ProSearchState) -> str:
    """Extract the best available user question from graph state.

    Args:
        state: Current pro-search graph state.

    Returns:
        The user question to use for downstream planning and synthesis.

    Raises:
        ValueError: Raised if no usable user question can be extracted.

    Examples:
        >>> _extract_latest_user_question({"user_question": "What changed recently?"})
        'What changed recently?'
    """
    user_question = state.get("user_question")
    if isinstance(user_question, str) and user_question.strip():
        return user_question

    messages = state.get("messages", [])
    for message in reversed(messages):
        if isinstance(message, dict):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
            continue

        msg_type = getattr(message, "type", None)
        content = getattr(message, "content", None)
        if msg_type == "human" and isinstance(content, str) and content.strip():
            return content

    raise ValueError("Could not extract a user question from the pro-search state.")


def _coerce_tool_message_payload(message: ToolMessage) -> dict[str, Any]:
    """Coerce a ``ToolMessage`` payload into a dictionary when possible.

    Args:
        message: Tool message returned by ``ToolNode``.

    Returns:
        A best-effort dictionary payload representation.

    Raises:
        TypeError: Raised if message content has an unexpected incompatible type.

    Examples:
        >>> tool_message = ToolMessage(content='{"answer": "ok"}', tool_call_id="abc")
        >>> _coerce_tool_message_payload(tool_message)["answer"]
        'ok'
    """
    artifact = getattr(message, "artifact", None)
    if isinstance(artifact, dict):
        return artifact

    content = message.content
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        stripped = content.strip()
        if not stripped:
            return {}

        try:
            loaded = json.loads(stripped)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

        try:
            loaded = ast.literal_eval(stripped)
            if isinstance(loaded, dict):
                return loaded
        except (ValueError, SyntaxError):
            pass

        return {"raw_text": stripped}

    if isinstance(content, list):
        return {"content_blocks": content}

    return {"raw_content_repr": repr(content)}


def _deduplicate_aggregated_results(
    aggregated_results: list[AggregatedResultRecord],
) -> list[AggregatedResultRecord]:
    """Deduplicate aggregated results by URL while preserving order.

    Args:
        aggregated_results: Flattened list of normalized evidence items.

    Returns:
        A deduplicated result list preserving first occurrence order.

    Raises:
        TypeError: Raised if the input list contains incompatible values.

    Examples:
        >>> _deduplicate_aggregated_results(
        ...     [
        ...         {"url": "https://example.com", "title": "A"},
        ...         {"url": "https://example.com", "title": "A again"},
        ...     ]
        ... )
        [{'url': 'https://example.com', 'title': 'A'}]
    """
    seen_urls: set[str] = set()
    deduplicated: list[AggregatedResultRecord] = []

    for item in aggregated_results:
        url = item.get("url")
        if not isinstance(url, str) or not url:
            deduplicated.append(item)
            continue

        if url in seen_urls:
            continue

        seen_urls.add(url)
        deduplicated.append(item)

    return deduplicated


def build_pro_search_graph(
    *,
    query_agent: SupportsInvoke | None = None,
    answer_agent: SupportsInvoke | None = None,
    checkpointer: Any = None,
    store: Any = None,
    debug: bool = False,
) -> Any:
    """Build the full pro-search graph.

    Returns:
        A compiled LangGraph workflow for planning, batched search execution,
        evidence aggregation, and answer synthesis.

    Raises:
        RuntimeError: Propagated if graph construction fails.

    Examples:
        >>> graph = build_pro_search_graph()
        >>> graph is not None
        True
    """
    resolved_query_agent = query_agent or build_query_generator_agent(
        checkpointer=checkpointer,
        store=store,
        debug=debug,
    )
    resolved_answer_agent = answer_agent or build_answer_agent(
        checkpointer=checkpointer,
        store=store,
        debug=debug,
    )
    pro_bundle = build_pro_bundle()
    search_tool = pro_bundle["search"]
    tool_node = ToolNode([search_tool])

    def generate_query_plan(
        state: ProSearchState,
        runtime: Runtime[ProSearchContext],
    ) -> dict[str, Any]:
        """Generate a structured query plan from the user question.

        Args:
            state: Current graph state.
            runtime: LangGraph runtime carrying the global pro-search context.

        Returns:
            Partial state update containing the structured query plan and
            related metadata.

        Raises:
            ValueError: Raised if no usable user question can be extracted.
        """
        context = runtime.context
        user_question = _extract_latest_user_question(state)

        result = resolved_query_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_question,
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-query-agent",
                }
            },
        )

        structured = result["structured_response"]
        query_plan = structured.model_dump(mode="json")

        return {
            "user_question": user_question,
            "normalized_question": query_plan["normalized_question"],
            "query_plan": query_plan,
            "query_count": query_plan["query_count"],
        }

    def build_batch_search_calls(
        state: ProSearchState,
        runtime: Runtime[ProSearchContext],
    ) -> dict[str, Any]:
        """Convert the query plan into a batched AI tool-call message.

        Args:
            state: Current graph state after query planning.
            runtime: LangGraph runtime carrying the global pro-search context.

        Returns:
            Partial state update containing planned query metadata and an AI
            message with Tavily tool calls for ``ToolNode`` execution.

        Raises:
            KeyError: Raised if the expected query plan structure is missing.
        """
        context = runtime.context
        query_plan = state["query_plan"]

        planned_queries: list[QueryExecutionRecord] = []
        tool_calls: list[dict[str, Any]] = []

        for query_entry in query_plan["queries"][: context.max_queries]:
            tool_call_id = f"tavily-search-{uuid4().hex}"

            planned_query: QueryExecutionRecord = {
                "tool_call_id": tool_call_id,
                "query": query_entry["query"],
                "priority": query_entry["priority"],
                "intent": query_entry["intent"],
                "rationale": query_entry["rationale"],
                "target_topic": query_entry["target_topic"],
                "prefer_recent_sources": query_entry["prefer_recent_sources"],
                "preferred_source_types": query_entry["preferred_source_types"],
            }
            planned_queries.append(planned_query)

            tool_calls.append(
                {
                    "id": tool_call_id,
                    "name": search_tool.name,
                    "args": {
                        "query": query_entry["query"],
                    },
                    "type": "tool_call",
                }
            )

        ai_message = AIMessage(
            content="Executing the planned Tavily pro-search batch.",
            tool_calls=tool_calls,
        )

        return {
            "planned_queries": planned_queries,
            "search_tool_calls_built": True,
            "messages": [ai_message],
        }

    def aggregate_search_results(state: ProSearchState) -> dict[str, Any]:
        """Aggregate ``ToolNode`` results into flattened normalized evidence.

        Args:
            state: Current graph state after batched tool execution.

        Returns:
            Partial state update containing raw query results, normalized
            aggregated evidence, and any collected search errors.
        """
        planned_by_id = {
            planned_query["tool_call_id"]: planned_query
            for planned_query in state.get("planned_queries", [])
        }

        raw_query_results: list[dict[str, Any]] = []
        aggregated_results: list[AggregatedResultRecord] = []
        search_errors: list[str] = list(state.get("search_errors", []))

        for message in state.get("messages", []):
            if not isinstance(message, ToolMessage):
                continue

            tool_call_id = message.tool_call_id
            if not isinstance(tool_call_id, str):
                continue

            planned_query = planned_by_id.get(tool_call_id)
            if planned_query is None:
                continue

            payload = _coerce_tool_message_payload(message)
            normalized_hits = normalize_search_payload(payload)
            query_answer = extract_answer(payload)

            raw_query_results.append(
                {
                    "query": planned_query,
                    "tool_call_id": tool_call_id,
                    "payload": payload,
                    "normalized_hits": normalized_hits,
                    "answer": query_answer,
                }
            )

            if "raw_text" in payload and not normalized_hits:
                search_errors.append(
                    f"Tool call {tool_call_id} returned unparsed text output for "
                    f"query '{planned_query['query']}'."
                )

            for hit in normalized_hits:
                aggregated_results.append(
                    {
                        "query": planned_query["query"],
                        "query_priority": planned_query["priority"],
                        "query_intent": planned_query["intent"],
                        "query_rationale": planned_query["rationale"],
                        "query_topic": planned_query["target_topic"],
                        "query_answer": query_answer,
                        "url": hit.get("url", ""),
                        "title": hit.get("title", ""),
                        "content": hit.get("content", ""),
                        "score": hit.get("score"),
                        "raw_content": hit.get("raw_content"),
                    }
                )

        deduplicated_results = _deduplicate_aggregated_results(aggregated_results)

        return {
            "raw_query_results": raw_query_results,
            "aggregated_results": deduplicated_results,
            "search_errors": search_errors,
        }

    def synthesize_answer(
        state: ProSearchState,
        runtime: Runtime[ProSearchContext],
    ) -> dict[str, Any]:
        """Synthesize the final markdown answer from aggregated evidence.

        Args:
            state: Current graph state after evidence aggregation.
            runtime: LangGraph runtime carrying the global pro-search context.

        Returns:
            Partial state update containing the final structured answer and a
            completion marker.

        Raises:
            KeyError: Raised if required state fields are missing.
        """
        context = runtime.context
        user_question = state["user_question"]
        aggregated_results = state.get("aggregated_results", [])

        answer_input = {
            "question": user_question,
            "normalized_question": state.get("normalized_question"),
            "query_plan": state.get("query_plan"),
            "aggregated_results": aggregated_results,
            "search_errors": state.get("search_errors", []),
        }

        result = resolved_answer_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(answer_input, indent=2, default=str),
                    }
                ]
            },
            context=context,
            config={
                "configurable": {
                    "thread_id": f"{context.thread_id}-answer-agent",
                }
            },
        )

        structured = result["structured_response"]
        return {
            "final_answer": structured.model_dump(mode="json"),
            "is_complete": True,
        }

    graph = StateGraph(ProSearchState)
    graph.add_node("generate_query_plan", generate_query_plan)
    graph.add_node("build_batch_search_calls", build_batch_search_calls)
    graph.add_node("run_search_tools", tool_node)
    graph.add_node("aggregate_search_results", aggregate_search_results)
    graph.add_node("synthesize_answer", synthesize_answer)

    graph.add_edge(START, "generate_query_plan")
    graph.add_edge("generate_query_plan", "build_batch_search_calls")
    graph.add_edge("build_batch_search_calls", "run_search_tools")
    graph.add_edge("run_search_tools", "aggregate_search_results")
    graph.add_edge("aggregate_search_results", "synthesize_answer")
    graph.add_edge("synthesize_answer", END)

    return graph.compile(
        checkpointer=checkpointer,
        store=store,
        debug=debug,
        name="pro_search",
    )
