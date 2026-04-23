"""State schemas for the quick-search agent.

Purpose:
    Define dynamic short-term memory for the quick-search agent.

Design:
    Quick-search state stores values that evolve during execution, such as the
    generated query, gathered search results, retry count, and latest answer
    sufficiency judgment. These values are separate from runtime context, which
    is static for a single invocation.

    The state schema extends ``AgentState`` so the built-in ``messages`` field
    remains available while quick-search-specific fields are added.

Attributes:
    SufficiencyVerdict:
        Literal verdict used to describe whether the gathered evidence answers
        the user's question.
    QuickSearchStateBase:
        Shared state fields for quick-search flows.
    QuickSearchState:
        Main state schema for the end-user quick-search agent.

Examples:
    .. code-block:: python

        state = {
            "messages": [],
            "retry_count": 0,
            "generated_query": "current Apple stock price",
            "gathered_results": [],
            "sufficiency_verdict": "answered",
        }
"""

from __future__ import annotations

from typing import Any, Literal
from typing_extensions import NotRequired

from langchain.agents import AgentState


type SufficiencyVerdict = Literal["answered", "partially_answered", "not_answered"]


class QuickSearchStateBase(AgentState):
    """Base dynamic state for quick-search flows.

    This base state captures the common working memory needed by quick-search
    variants. It intentionally remains small so future subagents can inherit it
    cleanly.

    Attributes:
        retry_count:
            Number of retry or refinement attempts used so far in the current
            run.
        generated_query:
            The main search query produced for the current quick-search run.
        gathered_results:
            Normalized search or extraction results collected so far.

    Examples:
        >>> state = {
        ...     "messages": [],
        ...     "retry_count": 0,
        ...     "generated_query": "latest Apple stock price",
        ...     "gathered_results": [],
        ... }
    """

    retry_count: NotRequired[int]
    generated_query: NotRequired[str]
    gathered_results: NotRequired[list[dict[str, Any]]]


class QuickSearchState(QuickSearchStateBase):
    """Dynamic state for the main quick-search answering agent.

    This state extends :class:`QuickSearchStateBase` with the current answer
    sufficiency judgment, which can later be used to drive retries, fallback
    behavior, or downstream answer-judgment subagents.

    Attributes:
        sufficiency_verdict:
            Latest judgment describing whether the currently gathered evidence
            fully answers the user's question.

    Examples:
        >>> state = {
        ...     "messages": [],
        ...     "retry_count": 0,
        ...     "generated_query": "latest Apple stock price",
        ...     "gathered_results": [],
        ...     "sufficiency_verdict": "answered",
        ... }
    """

    sufficiency_verdict: NotRequired[SufficiencyVerdict]