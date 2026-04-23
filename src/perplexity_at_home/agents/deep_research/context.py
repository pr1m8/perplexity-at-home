"""Runtime context models for the deep-research workflow.

Purpose:
    Define static per-invocation configuration for the top-level deep-research
    graph and its child agents.

Design:
    The deep-research context stores fixed run configuration such as datetime,
    retrieval budgets, branching limits, tool-permission toggles, and source
    preferences. These values are intended to remain stable for the full life of
    a single workflow invocation.

    This context should be read by graph nodes and child agents, but not mutated
    during execution. Anything that evolves during the run belongs in graph
    state, not here.

Attributes:
    DeepResearchContextBase:
        Shared static configuration for all deep-research flows.
    DeepResearchContext:
        Main runtime context for the top-level deep-research workflow.

Examples:
    .. code-block:: python

        context = DeepResearchContext(
            current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
            timezone_name="America/Toronto",
            max_subquestions=5,
            target_queries_per_subquestion=3,
            max_iterations=3,
            max_parallel_retrieval_branches=4,
            max_results_per_query=6,
            max_extract_urls_per_pass=5,
            allow_map=True,
            allow_crawl=True,
            allow_tavily_research=True,
            prefer_freshness=True,
            prefer_primary_sources=True,
            thread_id="deep-research",
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class DeepResearchContextBase:
    """Base runtime context for deep-research flows.

    Args:
        current_datetime: Human-readable current datetime string used for
            temporal grounding and interpretation of relative time expressions.
        timezone_name: Canonical timezone name associated with the current run.
        max_subquestions: Maximum number of subquestions the planner may produce
            for a single deep-research run.
        target_queries_per_subquestion: Desired number of complementary queries
            to generate for each subquestion.
        max_iterations: Maximum number of reflection/requery iterations allowed
            before the workflow must synthesize the best available answer.
        max_parallel_retrieval_branches: Maximum number of concurrent retrieval
            branches the workflow may fan out into at one time.
        max_results_per_query: Maximum number of normalized search results to
            retain per query during aggregation.
        max_extract_urls_per_pass: Maximum number of URLs that may be sent to an
            extraction stage in a single pass.
        prefer_freshness: Whether recent/current information should be favored
            when the question is time-sensitive.
        prefer_primary_sources: Whether primary or authoritative sources should
            be preferred when available.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.

    Examples:
        >>> context = DeepResearchContextBase(
        ...     current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
        ... )
        >>> context.max_iterations
        3
    """

    current_datetime: str | None = field(
        default=None,
        metadata={
            "description": (
                "Human-readable current datetime string used for temporal "
                "grounding across the full deep-research workflow."
            ),
        },
    )
    timezone_name: str = field(
        default="America/Toronto",
        metadata={
            "description": "Canonical timezone name for the current workflow run.",
        },
    )
    max_subquestions: int = field(
        default=5,
        metadata={
            "description": (
                "Maximum number of subquestions the planner may produce for a "
                "single deep-research run."
            ),
        },
    )
    target_queries_per_subquestion: int = field(
        default=3,
        metadata={
            "description": (
                "Desired number of complementary queries to generate for each "
                "subquestion."
            ),
        },
    )
    max_iterations: int = field(
        default=3,
        metadata={
            "description": (
                "Maximum number of reflection/requery iterations allowed before "
                "the workflow must synthesize an answer."
            ),
        },
    )
    max_parallel_retrieval_branches: int = field(
        default=4,
        metadata={
            "description": (
                "Maximum number of concurrent retrieval branches the workflow "
                "may run at one time."
            ),
        },
    )
    max_results_per_query: int = field(
        default=6,
        metadata={
            "description": (
                "Maximum number of normalized search results to retain for each "
                "executed query."
            ),
        },
    )
    max_extract_urls_per_pass: int = field(
        default=5,
        metadata={
            "description": (
                "Maximum number of URLs that may be sent to an extraction stage "
                "in a single pass."
            ),
        },
    )
    prefer_freshness: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the workflow should favor recent or current information "
                "when freshness matters."
            ),
        },
    )
    prefer_primary_sources: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the workflow should prefer primary or authoritative "
                "sources when they are available."
            ),
        },
    )


@dataclass(slots=True, kw_only=True)
class DeepResearchContext(DeepResearchContextBase):
    """Runtime context for the main deep-research workflow.

    Args:
        allow_map: Whether Tavily map is permitted as a retrieval strategy.
        allow_crawl: Whether Tavily crawl is permitted as a retrieval strategy.
        allow_tavily_research: Whether Tavily research/get-research may be used
            as a higher-level asynchronous research branch.
        allow_interrupts_for_clarification: Whether the workflow may interrupt
            to ask the user a clarifying question before continuing.
        thread_id: Default thread identifier used for nested child-agent calls.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.

    Examples:
        >>> context = DeepResearchContext(
        ...     current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
        ...     allow_map=True,
        ...     allow_crawl=True,
        ...     allow_tavily_research=True,
        ... )
        >>> context.allow_tavily_research
        True
    """

    allow_map: bool = field(
        default=True,
        metadata={
            "description": "Whether Tavily map is allowed in the workflow.",
        },
    )
    allow_crawl: bool = field(
        default=True,
        metadata={
            "description": "Whether Tavily crawl is allowed in the workflow.",
        },
    )
    allow_tavily_research: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether Tavily research/get-research may be used as a "
                "higher-level retrieval branch."
            ),
        },
    )
    allow_interrupts_for_clarification: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the workflow may interrupt to ask the user a "
                "clarifying question before continuing."
            ),
        },
    )
    thread_id: str = field(
        default="deep-research",
        metadata={
            "description": (
                "Default thread identifier used for nested child-agent "
                "invocations."
            ),
        },
    )