"""Runtime context models for the pro-search workflow.

Purpose:
    Define static per-invocation configuration for the top-level pro-search
    graph and its child agents.

Design:
    The pro-search context stores fixed run configuration such as datetime,
    query-budget settings, source-preference hints, and retrieval limits. These
    values are stable for the lifetime of a single workflow invocation and are
    intended to be consumed by graph nodes and child agents.

Attributes:
    ProSearchContextBase:
        Shared runtime context fields for the pro-search workflow.
    ProSearchContext:
        Main runtime context for the top-level pro-search workflow.

Examples:
    .. code-block:: python

        context = ProSearchContext(
            current_datetime="The current date and time is: 2026-04-22 23:15:57 EDT",
            timezone_name="America/Toronto",
            target_queries=3,
            min_queries=2,
            max_queries=4,
            max_results_per_query=5,
            prefer_freshness=True,
            prefer_primary_sources=True,
            prefer_query_diversity=True,
            default_topic="general",
            allow_multi_query=True,
            disallow_stale_year_anchors=True,
            allow_extract=True,
            thread_id="pro-search",
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class ProSearchContextBase:
    """Base runtime context for pro-search flows.

    Args:
        current_datetime: Human-readable current datetime string used for
            temporal grounding and interpretation of relative time expressions.
        timezone_name: Canonical timezone name for the current run.
        target_queries: Desired number of complementary queries for the run.
        min_queries: Minimum number of queries expected unless the question is
            exceptionally narrow.
        max_queries: Hard upper bound on the number of queries the workflow may
            generate and execute.
        max_results_per_query: Maximum number of Tavily results to retain per
            query during downstream aggregation.
        prefer_freshness: Whether the workflow should bias toward recent/current
            information when appropriate.
        prefer_primary_sources: Whether the workflow should favor queries likely
            to surface primary or authoritative sources.
        prefer_query_diversity: Whether the workflow should prefer
            complementary query angles over near-duplicate phrasing.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.
    """

    current_datetime: str | None = field(
        default=None,
        metadata={
            "description": (
                "Human-readable current datetime string used for temporal "
                "grounding across the full pro-search workflow."
            ),
        },
    )
    timezone_name: str = field(
        default="America/Toronto",
        metadata={
            "description": "Canonical timezone name for the current workflow run.",
        },
    )
    target_queries: int = field(
        default=3,
        metadata={
            "description": (
                "Desired number of complementary queries for the current "
                "pro-search run."
            ),
        },
    )
    min_queries: int = field(
        default=2,
        metadata={
            "description": (
                "Minimum number of queries expected unless the question is "
                "exceptionally narrow."
            ),
        },
    )
    max_queries: int = field(
        default=4,
        metadata={
            "description": (
                "Hard upper bound on the number of queries the workflow may "
                "generate and execute."
            ),
        },
    )
    max_results_per_query: int = field(
        default=5,
        metadata={
            "description": (
                "Maximum number of normalized Tavily results to retain for "
                "each executed query."
            ),
        },
    )
    prefer_freshness: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether recent or current information should be favored when relevant."
            ),
        },
    )
    prefer_primary_sources: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the workflow should prefer primary or authoritative sources."
            ),
        },
    )
    prefer_query_diversity: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the workflow should favor complementary query angles "
                "over near-duplicate paraphrases."
            ),
        },
    )


@dataclass(slots=True, kw_only=True)
class ProSearchContext(ProSearchContextBase):
    """Runtime context for the main pro-search workflow.

    Args:
        default_topic: Default downstream Tavily topic hint used when the
            question does not strongly imply a more specific topic.
        allow_multi_query: Whether multiple complementary queries may be
            generated and executed.
        disallow_stale_year_anchors: Whether stale year anchors should be
            avoided unless the user explicitly requested them.
        allow_extract: Whether downstream source extraction is permitted in the
            workflow.
        thread_id: Default thread identifier used for nested child-agent calls.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.
    """

    default_topic: str = field(
        default="general",
        metadata={
            "description": (
                "Default downstream Tavily topic hint for generated queries "
                "when no more specific topic is implied."
            ),
        },
    )
    allow_multi_query: bool = field(
        default=True,
        metadata={
            "description": "Whether multiple complementary queries may be generated.",
        },
    )
    disallow_stale_year_anchors: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether stale year anchors should be avoided unless explicitly requested."
            ),
        },
    )
    allow_extract: bool = field(
        default=True,
        metadata={
            "description": "Whether Tavily extraction is allowed in the workflow.",
        },
    )
    thread_id: str = field(
        default="pro-search",
        metadata={
            "description": (
                "Default thread identifier used for nested child-agent invocations."
            ),
        },
    )