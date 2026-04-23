"""Runtime context models for the quick-search agent.

Purpose:
    Define static per-invocation configuration for the quick-search agent.

Design:
    The quick-search context stores values that are decided before an agent run
    begins and remain stable for the life of that invocation. This includes
    temporal context, result-size limits, and simple execution policy flags.

    These fields are intended to be read by prompt builders, middleware, or
    tools, rather than mutated during execution.

Attributes:
    QuickSearchContextBase:
        Shared context fields common to all quick-search flows.
    QuickSearchContext:
        Main runtime context for the end-user quick-search agent.

Examples:
    .. code-block:: python

        context = QuickSearchContext(
            current_datetime="Wednesday, April 22, 2026, 11:30 AM America/Toronto",
            timezone_name="America/Toronto",
            max_queries=1,
            max_results_per_query=5,
            max_search_passes=1,
            allow_extract=True,
            require_citations=True,
        )
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, kw_only=True)
class QuickSearchContextBase:
    """Base runtime context for quick-search flows.

    This base context contains static configuration shared across quick-search
    variants, such as the main answering agent and any future query-generation
    or answer-judgment subagents.

    Args:
        current_datetime: Human-readable current datetime string used by the
            prompt to interpret relative time references such as ``today``,
            ``yesterday``, ``latest``, or ``current``.
        timezone_name: Canonical timezone name associated with the current run,
            such as ``America/Toronto``.
        max_queries: Maximum number of search queries the quick-search flow is
            allowed to generate for a single run. For quick search, ``1`` is the
            default and preferred starting point.
        max_results_per_query: Maximum number of Tavily search results to
            retrieve for each generated query.
        max_search_passes: Maximum number of search/refinement passes allowed in
            the run before the agent must answer with the best available
            evidence.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.

    Examples:
        >>> context = QuickSearchContextBase(
        ...     current_datetime="Wednesday, April 22, 2026, 11:30 AM America/Toronto",
        ...     timezone_name="America/Toronto",
        ... )
        >>> context.max_queries
        1
    """

    current_datetime: str | None = field(
        default=None,
        metadata={
            "description": (
                "Human-readable current datetime string used for temporal "
                "grounding in the system prompt."
            ),
        },
    )
    timezone_name: str = field(
        default="America/Toronto",
        metadata={
            "description": (
                "Canonical timezone name for the run, used to interpret "
                "relative dates and times."
            ),
        },
    )
    max_queries: int = field(
        default=1,
        metadata={
            "description": (
                "Maximum number of search queries allowed for a single "
                "quick-search run."
            ),
        },
    )
    max_results_per_query: int = field(
        default=5,
        metadata={
            "description": (
                "Maximum number of Tavily results to retrieve for each search "
                "query."
            ),
        },
    )
    max_search_passes: int = field(
        default=1,
        metadata={
            "description": (
                "Maximum number of search/refinement passes allowed before the "
                "agent must answer with the current evidence."
            ),
        },
    )


@dataclass(slots=True, kw_only=True)
class QuickSearchContext(QuickSearchContextBase):
    """Runtime context for the main quick-search answering agent.

    This context extends :class:`QuickSearchContextBase` with toggles that
    affect how the main quick-search agent is allowed to behave.

    Args:
        allow_extract: Whether the agent may use page extraction in addition to
            quick search when a returned source needs closer inspection.
        require_citations: Whether the final response is expected to include
            explicit citations when tool-derived evidence is used.

    Returns:
        None.

    Raises:
        TypeError: Raised by the dataclass constructor if incompatible field
            types are provided.

    Examples:
        >>> context = QuickSearchContext(
        ...     current_datetime="Wednesday, April 22, 2026, 11:30 AM America/Toronto",
        ...     allow_extract=True,
        ...     require_citations=True,
        ... )
        >>> context.require_citations
        True
    """

    allow_extract: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether Tavily extraction is allowed when a specific source "
                "needs deeper inspection."
            ),
        },
    )
    require_citations: bool = field(
        default=True,
        metadata={
            "description": (
                "Whether the final answer should include citations when search "
                "results were used."
            ),
        },
    )