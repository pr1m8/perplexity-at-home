"""Structured models for the pro-search query-generation agent.

Purpose:
    Define validated payloads produced by the pro-search query-generation agent.

Design:
    - Uses a local package-level base model for shared Pydantic configuration.
    - Produces a ranked, multi-query search plan for downstream execution.
    - Encodes query intent, source preferences, and freshness bias explicitly so
      the parent pro-search flow can execute the plan intelligently.

Attributes:
    QueryIntent:
        Semantic role of a generated query within the overall search plan.
    SearchTopicHint:
        Topic hint for downstream Tavily execution.
    CoverageAngle:
        Retrieval coverage category that the query plan is intended to cover.
    ProSearchQueryAgentModel:
        Shared base model for all query-agent payloads.
    GeneratedQuery:
        One generated search query plus execution hints.
    ProSearchQueryPlanBase:
        Shared fields for a structured query-generation result.
    ProSearchQueryPlan:
        Full structured output for the query-generation agent.

Examples:
    .. code-block:: python

        plan = ProSearchQueryPlan(
            original_question="What changed recently in the Tavily LangChain integration?",
            normalized_question="Recent changes in Tavily's LangChain integration",
            research_goal="Identify recent important changes and strong supporting sources.",
            requires_freshness=True,
            ambiguity_note=None,
            target_queries=3,
            min_queries=2,
            max_queries=4,
            coverage_strategy=["direct", "primary_source", "freshness"],
            query_count=3,
            queries=[
                GeneratedQuery(
                    query="Tavily LangChain integration recent changes",
                    rationale="Primary direct query for the main topic.",
                    priority=1,
                    intent="direct",
                    target_topic="general",
                ),
            ],
        )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type QueryIntent = Literal[
    "direct",
    "broadening",
    "disambiguation",
    "freshness",
    "comparison",
    "primary_source",
]
type SearchTopicHint = Literal["general", "news", "finance"]
type CoverageAngle = Literal[
    "direct",
    "primary_source",
    "freshness",
    "alternative_phrasing",
    "broadening",
    "disambiguation",
    "comparison",
]


class ProSearchQueryAgentModel(BaseModel):
    """Base model for all pro-search query-agent payloads.

    Returns:
        ProSearchQueryAgentModel: A validated query-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(ProSearchQueryAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class GeneratedQuery(ProSearchQueryAgentModel):
    """A single generated search query for downstream execution.

    Args:
        query: Focused search query text to send to downstream Tavily search.
        rationale: Short explanation of why this query is useful.
        priority: Rank order where ``1`` is the highest-priority query.
        intent: Semantic role of the query in the overall plan.
        target_topic: Preferred Tavily topic hint for downstream execution.
        prefer_recent_sources: Whether downstream execution should prefer recent
            sources for this query.
        preferred_source_types: Optional source-type hints, such as official
            docs, company pages, government sites, GitHub, or major news.
        follow_up_of: Optional priority number of an earlier query that this
            query refines or follows up on.

    Returns:
        GeneratedQuery: A validated generated-query object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> generated = GeneratedQuery(
        ...     query="Tavily LangChain integration recent changes",
        ...     rationale="Primary direct query for the main topic.",
        ...     priority=1,
        ...     intent="direct",
        ...     target_topic="general",
        ... )
        >>> generated.priority
        1
    """

    query: str = Field(
        min_length=1,
        max_length=400,
        description=(
            "Focused search query text for downstream execution. Keep it concise, "
            "high-signal, and executable as a real web search query."
        ),
    )
    rationale: str = Field(
        min_length=1,
        description=(
            "Brief explanation of why this query is useful for answering the "
            "user's question."
        ),
    )
    priority: int = Field(
        ge=1,
        description="Priority rank for the query, where 1 is the highest priority.",
    )
    intent: QueryIntent = Field(
        description=(
            "Semantic role of the query within the plan, such as direct, "
            "primary_source, freshness, or disambiguation."
        ),
    )
    target_topic: SearchTopicHint = Field(
        default="general",
        description="Preferred Tavily topic hint for downstream search execution.",
    )
    prefer_recent_sources: bool = Field(
        default=False,
        description=(
            "Whether downstream execution should prefer more recent sources for "
            "this query."
        ),
    )
    preferred_source_types: list[str] = Field(
        default_factory=list,
        description=(
            "Optional source-type hints for downstream execution, such as "
            "official docs, GitHub, company updates, government sites, or major news."
        ),
    )
    follow_up_of: int | None = Field(
        default=None,
        description=(
            "Optional priority number of an earlier query that this query is "
            "intended to refine or follow up on."
        ),
    )


class ProSearchQueryPlanBase(ProSearchQueryAgentModel):
    """Base structured output for a pro-search query-generation run.

    Args:
        original_question: Original user question received by the parent flow.
        normalized_question: Canonical restatement of the user's question after
            light cleanup or scoping.
        research_goal: Concise description of what the downstream search should
            learn, verify, or retrieve.
        requires_freshness: Whether recent or current information is important
            for the query plan.
        ambiguity_note: Optional note describing unresolved ambiguity, scope
            uncertainty, or entity disambiguation issues.
        target_queries: Desired number of complementary queries for this run.
        min_queries: Minimum number of queries expected unless the question is
            exceptionally narrow.
        max_queries: Hard upper bound on the number of generated queries.
        coverage_strategy: Coverage angles the plan is intentionally trying to
            satisfy, such as direct, primary_source, or freshness.
        query_count: Number of queries actually returned in the plan.

    Returns:
        ProSearchQueryPlanBase: A validated query-plan base object.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    original_question: str = Field(
        min_length=1,
        description="Original user question supplied to the pro-search flow.",
    )
    normalized_question: str = Field(
        min_length=1,
        description=(
            "Canonical restatement of the user's question after normalization or "
            "light disambiguation."
        ),
    )
    research_goal: str = Field(
        min_length=1,
        description=(
            "Concise statement of what the downstream pro-search flow should "
            "discover, verify, or retrieve."
        ),
    )
    requires_freshness: bool = Field(
        description=(
            "Whether recent or current information is important for answering "
            "the question."
        ),
    )
    ambiguity_note: str | None = Field(
        default=None,
        description=(
            "Optional note describing ambiguity, missing scope, or unresolved "
            "entity disambiguation concerns."
        ),
    )
    target_queries: int = Field(
        ge=1,
        description=(
            "Desired number of complementary queries for this run. The model "
            "should usually aim for this number when the question supports it."
        ),
    )
    min_queries: int = Field(
        ge=1,
        description=(
            "Minimum number of queries expected for the plan unless the question "
            "is extremely narrow."
        ),
    )
    max_queries: int = Field(
        ge=1,
        description="Hard upper bound on the number of queries the plan may contain.",
    )
    coverage_strategy: list[CoverageAngle] = Field(
        min_length=1,
        description=(
            "Coverage angles intentionally represented in the plan, such as "
            "direct, primary_source, freshness, or alternative_phrasing."
        ),
    )
    query_count: int = Field(
        ge=1,
        description="Number of generated queries returned in the plan.",
    )


class ProSearchQueryPlan(ProSearchQueryPlanBase):
    """Full structured output for the pro-search query-generation agent.

    Args:
        queries: Ranked generated queries for downstream execution.

    Returns:
        ProSearchQueryPlan: A validated full query plan.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> plan = ProSearchQueryPlan(
        ...     original_question="What changed recently in the Tavily LangChain integration?",
        ...     normalized_question="Recent changes in Tavily's LangChain integration",
        ...     research_goal="Identify recent important changes and strong supporting sources.",
        ...     requires_freshness=True,
        ...     ambiguity_note=None,
        ...     target_queries=3,
        ...     min_queries=2,
        ...     max_queries=4,
        ...     coverage_strategy=["direct", "primary_source", "freshness"],
        ...     query_count=3,
        ...     queries=[
        ...         GeneratedQuery(
        ...             query="Tavily LangChain integration recent changes",
        ...             rationale="Primary direct query.",
        ...             priority=1,
        ...             intent="direct",
        ...             target_topic="general",
        ...         ),
        ...     ],
        ... )
        >>> plan.query_count
        3
    """

    queries: list[GeneratedQuery] = Field(
        min_length=1,
        description=(
            "Ranked list of generated queries for the downstream pro-search execution flow."
        ),
    )