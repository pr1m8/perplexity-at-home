"""Runtime context for the pro-search query-generation agent.

Purpose:
    Define static per-invocation configuration for the pro-search query agent.

Design:
    - Context stores fixed run configuration, not evolving working memory.
    - These values guide breadth, freshness, source quality, and query-count
      expectations.
    - The context is intended to be read by prompt builders and, later, parent
      orchestration logic.

Attributes:
    QueryAgentContextBase:
        Shared runtime context fields for query generation.
    QueryAgentContext:
        Main runtime context for the pro-search query-generation agent.

Examples:
    .. code-block:: python

        context = QueryAgentContext(
            current_datetime="The current date and time is: 2026-04-22 23:10:01 EDT",
            timezone_name="America/Toronto",
            target_queries=3,
            min_queries=2,
            max_queries=4,
            prefer_freshness=True,
            prefer_primary_sources=True,
            default_topic="general",
            allow_multi_query=True,
            disallow_stale_year_anchors=True,
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class QueryAgentContextBase:
    """Base runtime context for pro-search query generation.

    Args:
        current_datetime: Human-readable current datetime string used for
            temporal grounding and interpretation of relative time references.
        timezone_name: Canonical timezone name associated with the current run.
        target_queries: Desired number of complementary queries for this run.
        min_queries: Minimum number of queries expected unless the question is
            extremely narrow.
        max_queries: Hard upper bound on the number of queries the agent may
            generate.
        prefer_freshness: Whether the agent should bias toward recent/current
            information when appropriate.
        prefer_primary_sources: Whether the agent should favor queries likely to
            surface primary or authoritative sources.
        prefer_query_diversity: Whether the agent should favor complementary
            query angles rather than small redundant variations.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.
    """

    current_datetime: str | None = field(
        default=None,
        metadata={
            "description": "Human-readable current datetime string used for temporal grounding.",
        },
    )
    timezone_name: str = field(
        default="America/Toronto",
        metadata={
            "description": "Canonical timezone name for the current run.",
        },
    )
    target_queries: int = field(
        default=3,
        metadata={
            "description": "Desired number of complementary queries for this run.",
        },
    )
    min_queries: int = field(
        default=2,
        metadata={
            "description": "Minimum number of queries expected unless the question is extremely narrow.",
        },
    )
    max_queries: int = field(
        default=4,
        metadata={
            "description": "Hard upper bound on the number of queries the agent may generate.",
        },
    )
    prefer_freshness: bool = field(
        default=True,
        metadata={
            "description": "Whether recent or current information should be favored when relevant.",
        },
    )
    prefer_primary_sources: bool = field(
        default=True,
        metadata={
            "description": "Whether the agent should prefer queries likely to surface primary or authoritative sources.",
        },
    )
    prefer_query_diversity: bool = field(
        default=True,
        metadata={
            "description": "Whether the agent should favor complementary query angles over near-duplicates.",
        },
    )


@dataclass(slots=True, kw_only=True)
class QueryAgentContext(QueryAgentContextBase):
    """Runtime context for the main pro-search query-generation agent.

    Args:
        default_topic: Default downstream Tavily topic hint to use when the
            question does not strongly imply a more specific topic.
        allow_multi_query: Whether multiple complementary queries may be
            generated.
        disallow_stale_year_anchors: Whether the agent should avoid injecting an
            older year into freshness-sensitive queries unless the user asked
            for that year explicitly.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.
    """

    default_topic: str = field(
        default="general",
        metadata={
            "description": "Default downstream topic hint for generated queries.",
        },
    )
    allow_multi_query: bool = field(
        default=True,
        metadata={
            "description": "Whether multiple focused queries may be generated for the plan.",
        },
    )
    disallow_stale_year_anchors: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the agent should avoid inserting an older year into "
                "freshness-sensitive queries unless the user explicitly asked "
                "for that year."
            ),
        },
    )