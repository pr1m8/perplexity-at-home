"""Structured models for the deep-research query-generation agent.

Purpose:
    Define validated payloads produced by the deep-research query agent.

Design:
    The deep-research query agent operates after planning and subquestion
    decomposition. Its job is to transform each planned subquestion into a
    retrieval-ready query plan that the parent deep-research graph can execute.

    Unlike a quick-search or pro-search query planner, this agent must support:
    - multiple subquestions,
    - multiple complementary queries per subquestion,
    - retrieval-strategy recommendations,
    - and downstream routing into different Tavily capabilities such as search,
      extract, map, crawl, or research.

    These models are intentionally graph-friendly. The parent workflow can
    serialize the output into graph state and use it to:
    - build batched search calls,
    - route subquestions into different retrieval branches,
    - and preserve query-planning rationale for reflection and debugging.

Attributes:
    QueryIntent:
        Semantic role of a generated query within a subquestion plan.
    SearchTopicHint:
        Topic hint for downstream Tavily execution.
    RetrievalStrategy:
        Recommended high-level retrieval strategy for a subquestion.
    DeepResearchQueryAgentModel:
        Shared base model for query-agent payloads.
    GeneratedQuery:
        One ranked search query plus execution hints.
    RetrievalRecommendation:
        Structured recommendation describing how a subquestion should be
        retrieved downstream.
    SubquestionQueryPlanBase:
        Base query-planning payload for one subquestion.
    SubquestionQueryPlan:
        Full query plan for one subquestion.
    DeepResearchQueryPlansBase:
        Base top-level output for the deep-research query agent.
    DeepResearchQueryPlans:
        Full top-level output containing plans for all subquestions.

Examples:
    .. code-block:: python

        plans = DeepResearchQueryPlans(
            original_question=(
                "Write a deep report on recent changes in Tavily's LangChain integration."
            ),
            normalized_question=(
                "Research recent important changes in Tavily's LangChain integration."
            ),
            plan_count=1,
            plans=[
                SubquestionQueryPlan(
                    subquestion_id="sq_1",
                    subquestion=(
                        "What recent capabilities or API surface changes were introduced?"
                    ),
                    research_focus=(
                        "Identify the most important recent integration or API surface changes."
                    ),
                    requires_freshness=True,
                    target_queries=3,
                    min_queries=2,
                    max_queries=4,
                    retrieval_recommendation=RetrievalRecommendation(
                        strategy="search_then_extract",
                        rationale="Search first, then extract the strongest official sources.",
                    ),
                    queries=[
                        GeneratedQuery(
                            query="Tavily LangChain integration recent changes",
                            rationale="Primary direct query for recent changes.",
                            priority=1,
                            intent="direct",
                            target_topic="general",
                        )
                    ],
                )
            ],
        )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type QueryIntent = Literal[
    "direct",
    "primary_source",
    "freshness",
    "disambiguation",
    "broadening",
    "verification",
    "alternative_phrasing",
    "comparison",
]
type SearchTopicHint = Literal["general", "news", "finance"]
type RetrievalStrategy = Literal[
    "search",
    "search_then_extract",
    "extract_known_urls",
    "map_then_extract",
    "crawl_domain",
    "tavily_research",
]


class DeepResearchQueryAgentModel(BaseModel):
    """Base model for all deep-research query-agent payloads.

    Returns:
        DeepResearchQueryAgentModel: A validated query-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(DeepResearchQueryAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class GeneratedQuery(DeepResearchQueryAgentModel):
    """A single generated search query for downstream execution.

    Args:
        query: Focused search query text to send to downstream retrieval.
        rationale: Short explanation of why this query is useful.
        priority: Rank order where ``1`` is the highest-priority query.
        intent: Semantic role of the query within the subquestion plan.
        target_topic: Preferred Tavily topic hint for downstream execution.
        prefer_recent_sources: Whether downstream execution should prefer recent
            sources for this query.
        preferred_source_types: Optional source-type hints, such as official
            docs, GitHub, government sites, company updates, or news.
        follow_up_of: Optional priority number of an earlier query that this
            query refines or follows up on.

    Returns:
        GeneratedQuery: A validated generated-query object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> generated = GeneratedQuery(
        ...     query="Tavily LangChain integration recent changes",
        ...     rationale="Primary direct query for the main recent-change dimension.",
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
            "Focused query text for downstream retrieval. Keep it concise, "
            "high-signal, and executable as a real web search query."
        ),
    )
    rationale: str = Field(
        min_length=1,
        description=(
            "Brief explanation of why this query is useful for answering the "
            "associated subquestion."
        ),
    )
    priority: int = Field(
        ge=1,
        description="Priority rank for the query, where 1 is the highest priority.",
    )
    intent: QueryIntent = Field(
        description=(
            "Semantic role of the query within the subquestion plan, such as "
            "direct, freshness, primary_source, or verification."
        ),
    )
    target_topic: SearchTopicHint = Field(
        default="general",
        description="Preferred Tavily topic hint for downstream search execution.",
    )
    prefer_recent_sources: bool = Field(
        default=False,
        description=(
            "Whether downstream retrieval should prefer more recent sources for "
            "this query."
        ),
    )
    preferred_source_types: list[str] = Field(
        default_factory=list,
        description=(
            "Optional source-type hints for downstream retrieval, such as "
            "official docs, GitHub, company updates, government sites, or news."
        ),
    )
    follow_up_of: int | None = Field(
        default=None,
        description=(
            "Optional priority number of an earlier query that this query is "
            "intended to refine or follow up on."
        ),
    )


class RetrievalRecommendation(DeepResearchQueryAgentModel):
    """Recommended retrieval strategy for a subquestion.

    Args:
        strategy: High-level downstream retrieval strategy recommended for this
            subquestion.
        rationale: Explanation of why this strategy is appropriate.
        preferred_domains: Optional domains that downstream retrieval should
            prioritize if known.
        known_urls: Optional URLs already known to be relevant for downstream
            extraction or verification.
        should_fan_out: Whether downstream retrieval should fan out into
            multiple retrieval calls or branches.
        recommended_max_branches: Optional suggested maximum number of
            retrieval branches for this subquestion.

    Returns:
        RetrievalRecommendation: A validated retrieval recommendation.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> recommendation = RetrievalRecommendation(
        ...     strategy="search_then_extract",
        ...     rationale="Search first, then extract the strongest official sources.",
        ... )
        >>> recommendation.strategy
        'search_then_extract'
    """

    strategy: RetrievalStrategy = Field(
        description=(
            "High-level downstream retrieval strategy recommended for this "
            "subquestion."
        ),
    )
    rationale: str = Field(
        min_length=1,
        description="Explanation of why this retrieval strategy is appropriate.",
    )
    preferred_domains: list[str] = Field(
        default_factory=list,
        description=(
            "Optional domains that downstream retrieval should prioritize when known."
        ),
    )
    known_urls: list[str] = Field(
        default_factory=list,
        description=(
            "Optional URLs already known to be relevant for extraction or verification."
        ),
    )
    should_fan_out: bool = Field(
        default=False,
        description=(
            "Whether downstream retrieval should fan out into multiple calls or branches."
        ),
    )
    recommended_max_branches: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional suggested maximum number of retrieval branches for this subquestion."
        ),
    )


class SubquestionQueryPlanBase(DeepResearchQueryAgentModel):
    """Base query-planning payload for one subquestion.

    Args:
        subquestion_id: Stable identifier for the subquestion.
        subquestion: The actual subquestion text being planned.
        research_focus: Concise statement of what downstream retrieval should
            try to learn or verify for this subquestion.
        requires_freshness: Whether recent or current information is important
            for this subquestion.
        ambiguity_note: Optional note about ambiguity, unresolved scope, or
            entity disambiguation concerns.
        target_queries: Desired number of complementary queries for this
            subquestion.
        min_queries: Minimum number of queries expected unless the subquestion
            is exceptionally narrow.
        max_queries: Hard upper bound on the number of generated queries for
            this subquestion.
        retrieval_recommendation: Recommended downstream retrieval strategy for
            this subquestion.

    Returns:
        SubquestionQueryPlanBase: A validated subquestion query-plan base object.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    subquestion_id: str = Field(
        min_length=1,
        description="Stable identifier for the subquestion.",
    )
    subquestion: str = Field(
        min_length=1,
        description="The subquestion text being planned.",
    )
    research_focus: str = Field(
        min_length=1,
        description=(
            "Concise statement of what downstream retrieval should try to "
            "learn or verify for this subquestion."
        ),
    )
    requires_freshness: bool = Field(
        description=(
            "Whether recent or current information is important for this subquestion."
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
            "Desired number of complementary queries for this subquestion."
        ),
    )
    min_queries: int = Field(
        ge=1,
        description=(
            "Minimum number of queries expected unless the subquestion is "
            "exceptionally narrow."
        ),
    )
    max_queries: int = Field(
        ge=1,
        description=(
            "Hard upper bound on the number of queries that may be generated "
            "for this subquestion."
        ),
    )
    retrieval_recommendation: RetrievalRecommendation = Field(
        description=(
            "Recommended downstream retrieval strategy for this subquestion."
        ),
    )


class SubquestionQueryPlan(SubquestionQueryPlanBase):
    """Full query plan for one subquestion.

    Args:
        queries: Ranked generated queries for downstream execution.

    Returns:
        SubquestionQueryPlan: A validated full subquestion query plan.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> plan = SubquestionQueryPlan(
        ...     subquestion_id="sq_1",
        ...     subquestion="What recent capabilities were added?",
        ...     research_focus="Identify recent capabilities and supporting evidence.",
        ...     requires_freshness=True,
        ...     target_queries=3,
        ...     min_queries=2,
        ...     max_queries=4,
        ...     retrieval_recommendation=RetrievalRecommendation(
        ...         strategy="search_then_extract",
        ...         rationale="Search first, then extract the strongest sources.",
        ...     ),
        ...     queries=[
        ...         GeneratedQuery(
        ...             query="Tavily LangChain integration recent capabilities",
        ...             rationale="Primary direct query.",
        ...             priority=1,
        ...             intent="direct",
        ...             target_topic="general",
        ...         )
        ...     ],
        ... )
        >>> len(plan.queries)
        1
    """

    queries: list[GeneratedQuery] = Field(
        min_length=1,
        description=(
            "Ranked list of generated queries for downstream execution on this subquestion."
        ),
    )


class DeepResearchQueryPlansBase(DeepResearchQueryAgentModel):
    """Base top-level output for the deep-research query agent.

    Args:
        original_question: Original user question or research request.
        normalized_question: Canonical restatement of the original question.
        plan_count: Number of subquestion query plans returned.
        global_notes: Optional notes about retrieval strategy, ambiguity, or
            workflow-level tradeoffs that may help downstream orchestration.

    Returns:
        DeepResearchQueryPlansBase: A validated top-level query-agent base output.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    original_question: str = Field(
        min_length=1,
        description="Original user question or research request.",
    )
    normalized_question: str = Field(
        min_length=1,
        description="Canonical restatement of the original question.",
    )
    plan_count: int = Field(
        ge=1,
        description="Number of subquestion query plans returned by the agent.",
    )
    global_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Optional workflow-level notes about retrieval strategy, ambiguity, "
            "or planning tradeoffs."
        ),
    )


class DeepResearchQueryPlans(DeepResearchQueryPlansBase):
    """Full top-level output for the deep-research query agent.

    Args:
        plans: Query plans for each subquestion in the current deep-research run.

    Returns:
        DeepResearchQueryPlans: A validated full top-level query-agent output.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> output = DeepResearchQueryPlans(
        ...     original_question="Research recent changes in Tavily's LangChain integration.",
        ...     normalized_question="Research recent important changes in Tavily's LangChain integration.",
        ...     plan_count=1,
        ...     plans=[
        ...         SubquestionQueryPlan(
        ...             subquestion_id="sq_1",
        ...             subquestion="What recent capabilities were added?",
        ...             research_focus="Identify recent capabilities and supporting evidence.",
        ...             requires_freshness=True,
        ...             target_queries=3,
        ...             min_queries=2,
        ...             max_queries=4,
        ...             retrieval_recommendation=RetrievalRecommendation(
        ...                 strategy="search_then_extract",
        ...                 rationale="Search first, then extract the strongest sources.",
        ...             ),
        ...             queries=[
        ...                 GeneratedQuery(
        ...                     query="Tavily LangChain integration recent capabilities",
        ...                     rationale="Primary direct query.",
        ...                     priority=1,
        ...                     intent="direct",
        ...                     target_topic="general",
        ...                 )
        ...             ],
        ...         )
        ...     ],
        ... )
        >>> output.plan_count
        1
    """

    plans: list[SubquestionQueryPlan] = Field(
        min_length=1,
        description=(
            "Query plans for each subquestion in the current deep-research run."
        ),
    )