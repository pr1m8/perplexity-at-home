"""Structured models for the deep-research retrieval agent.

Purpose:
    Define validated payloads produced by the deep-research retrieval agent.

Design:
    The retrieval agent receives a planned research payload and uses Tavily
    tools to gather evidence. It does not write the final answer. Instead, it
    returns a structured evidence bundle that the parent graph can pass to
    reflection and, later, answer synthesis.

    This V2 version supports the full Tavily tool surface:
    - search
    - extract
    - map
    - crawl
    - research
    - get_research

    The schema is intentionally strict about strategy accountability so the
    parent graph can detect when the agent ignored the recommended retrieval
    strategy and can route, retry, or warn accordingly.

Attributes:
    RetrievalStrategy:
        High-level retrieval strategy label.
    NextAction:
        High-level next step recommended for the parent graph.
    RetrievalAgentModel:
        Shared base model for retrieval-agent payloads.
    ToolUsageRecord:
        Structured record of one tool usage event.
    RetrievedEvidenceItem:
        Normalized evidence item retained from the retrieval pass.
    RetrievalAgentResultBase:
        Base structured output for the retrieval agent.
    RetrievalAgentResult:
        Full structured retrieval output including evidence and tool usage.

Examples:
    .. code-block:: python

        result = RetrievalAgentResult(
            retrieval_summary="Collected recent official and technical sources.",
            recommended_strategy="search_then_extract",
            applied_strategy="search_then_extract",
            followed_recommended_strategy=True,
            strategy_rationale="Search was used for discovery, then extraction deepened the strongest URLs.",
            evidence_items=[],
            used_tools=[],
            unresolved_gaps=[],
            recommended_next_action="reflect",
            confidence=0.82,
        )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type RetrievalStrategy = Literal[
    "search",
    "search_then_extract",
    "extract_known_urls",
    "map_then_extract",
    "crawl_domain",
    "tavily_research",
]
type NextAction = Literal[
    "reflect",
    "requery",
    "extract",
    "map",
    "crawl",
    "research",
    "synthesize",
]


class RetrievalAgentModel(BaseModel):
    """Base model for all retrieval-agent payloads.

    Returns:
        RetrievalAgentModel: A validated retrieval-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(RetrievalAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class ToolUsageRecord(RetrievalAgentModel):
    """Structured record of one tool usage event.

    Args:
        tool_name: Canonical logical name of the tool used, such as ``search``,
            ``extract``, ``map``, ``crawl``, ``research``, or ``get_research``.
        rationale: Why this tool was used for the current retrieval pass.
        input_summary: Short summary of the query, URL, or instruction supplied
            to the tool.
        output_summary: Short summary of what the tool returned or enabled.
        subquestion_id: Optional subquestion identifier associated with the tool use.

    Returns:
        ToolUsageRecord: A validated tool-usage record.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> usage = ToolUsageRecord(
        ...     tool_name="search",
        ...     rationale="Used for first-pass discovery.",
        ...     input_summary="Tavily LangChain integration recent changes",
        ...     output_summary="Returned official docs and repository sources.",
        ... )
        >>> usage.tool_name
        'search'
    """

    tool_name: str = Field(
        min_length=1,
        description=(
            "Canonical logical name of the tool used, such as search, extract, "
            "map, crawl, research, or get_research."
        ),
    )
    rationale: str = Field(
        min_length=1,
        description="Why this tool was used for the current retrieval pass.",
    )
    input_summary: str = Field(
        min_length=1,
        description="Short summary of the input supplied to the tool.",
    )
    output_summary: str | None = Field(
        default=None,
        description="Short summary of what the tool returned or enabled.",
    )
    subquestion_id: str | None = Field(
        default=None,
        description="Optional subquestion identifier associated with this tool use.",
    )


class RetrievedEvidenceItem(RetrievalAgentModel):
    """Normalized evidence item retained from the retrieval pass.

    Args:
        source_type: Retrieval path that produced this item, such as search,
            extract, map, crawl, or research.
        subquestion_id: Optional subquestion identifier most associated with
            this evidence item.
        url: Canonical URL of the evidence source when available.
        title: Human-readable source title when available.
        content: Snippet, extracted text, or summary retained as evidence.
        score: Optional relevance or ranking score.
        raw_content: Optional raw or long-form content retained for later use.
        supports: Optional note about what claim or angle this evidence item supports.
        source_authority: Optional coarse label such as ``official_docs``,
            ``official_repo``, ``major_news``, ``third_party``, or ``unknown``.
        is_primary_source: Whether this item should be treated as a primary or
            authoritative source.
        evidence_date: Optional coarse date string if the evidence item clearly
            exposes one.

    Returns:
        RetrievedEvidenceItem: A validated evidence item.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> item = RetrievedEvidenceItem(
        ...     source_type="extract",
        ...     url="https://docs.tavily.com/documentation/integrations/langchain",
        ...     title="LangChain integration docs",
        ...     content="The docs describe the current integration surface.",
        ...     is_primary_source=True,
        ... )
        >>> item.is_primary_source
        True
    """

    source_type: str = Field(
        min_length=1,
        description="Retrieval path that produced this evidence item.",
    )
    subquestion_id: str | None = Field(
        default=None,
        description="Optional subquestion identifier associated with this evidence item.",
    )
    url: str | None = Field(
        default=None,
        description="Canonical URL of the evidence source when available.",
    )
    title: str | None = Field(
        default=None,
        description="Human-readable source title when available.",
    )
    content: str = Field(
        min_length=1,
        description=(
            "Snippet, extracted text, or summary retained as evidence for later "
            "reflection or synthesis."
        ),
    )
    score: float | None = Field(
        default=None,
        description="Optional relevance or ranking score for the evidence item.",
    )
    raw_content: str | None = Field(
        default=None,
        description="Optional raw or long-form content retained for later use.",
    )
    supports: str | None = Field(
        default=None,
        description="Optional note describing what claim or angle this item supports.",
    )
    source_authority: str | None = Field(
        default=None,
        description=(
            "Optional coarse source-authority label such as official_docs, "
            "official_repo, major_news, third_party, or unknown."
        ),
    )
    is_primary_source: bool = Field(
        default=False,
        description=(
            "Whether this evidence item should be treated as a primary or "
            "authoritative source."
        ),
    )
    evidence_date: str | None = Field(
        default=None,
        description=(
            "Optional coarse date string if the evidence item clearly exposes one."
        ),
    )


class RetrievalAgentResultBase(RetrievalAgentModel):
    """Base structured output for the retrieval agent.

    Args:
        retrieval_summary: Concise operational summary of what the retrieval pass achieved.
        recommended_strategy: Strategy that the input payload recommended.
        applied_strategy: Strategy the retrieval agent actually followed.
        followed_recommended_strategy: Whether the retrieval agent substantially
            followed the recommended strategy.
        strategy_rationale: Explanation of why the applied strategy was chosen.
        used_tools: Structured records of the tools used during retrieval.
        unresolved_gaps: Important missing facts or angles that remain unresolved.
        recommended_next_action: Best next step for the parent graph.
        confidence: Confidence score for the usefulness and coverage of the
            current retrieval bundle.

    Returns:
        RetrievalAgentResultBase: A validated retrieval result base.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    retrieval_summary: str = Field(
        min_length=1,
        description=(
            "Concise operational summary of what the retrieval pass achieved."
        ),
    )
    recommended_strategy: RetrievalStrategy | None = Field(
        default=None,
        description=(
            "Strategy recommended by the input retrieval plan for the current pass."
        ),
    )
    applied_strategy: RetrievalStrategy = Field(
        description="Strategy the retrieval agent actually followed.",
    )
    followed_recommended_strategy: bool = Field(
        description=(
            "Whether the retrieval agent substantially followed the recommended strategy."
        ),
    )
    strategy_rationale: str = Field(
        min_length=1,
        description=(
            "Explanation of why the applied retrieval strategy was chosen."
        ),
    )
    used_tools: list[ToolUsageRecord] = Field(
        default_factory=list,
        description="Structured records of the tools used during retrieval.",
    )
    unresolved_gaps: list[str] = Field(
        default_factory=list,
        description="Important missing facts or angles that remain unresolved.",
    )
    recommended_next_action: NextAction = Field(
        description="Best next step for the parent graph after this retrieval pass.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the usefulness and coverage of the current retrieval bundle."
        ),
    )


class RetrievalAgentResult(RetrievalAgentResultBase):
    """Full structured output for the retrieval agent.

    Args:
        evidence_items: Normalized evidence items retained from the retrieval pass.

    Returns:
        RetrievalAgentResult: A validated full retrieval output.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> result = RetrievalAgentResult(
        ...     retrieval_summary="Collected official and recent technical evidence.",
        ...     recommended_strategy="search_then_extract",
        ...     applied_strategy="search_then_extract",
        ...     followed_recommended_strategy=True,
        ...     strategy_rationale="Search discovered candidate sources and extraction deepened them.",
        ...     evidence_items=[],
        ...     used_tools=[],
        ...     unresolved_gaps=[],
        ...     recommended_next_action="reflect",
        ...     confidence=0.81,
        ... )
        >>> result.recommended_next_action
        'reflect'
    """

    evidence_items: list[RetrievedEvidenceItem] = Field(
        default_factory=list,
        description="Normalized evidence items retained from the retrieval pass.",
    )
