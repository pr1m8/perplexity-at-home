"""Structured models for the deep-research reflection agent.

Purpose:
    Define validated payloads produced by the deep-research reflection agent.

Design:
    The reflection agent evaluates the current research state after one or more
    retrieval passes. Its job is to determine whether the workflow has enough
    evidence to proceed to synthesis or whether it should continue researching.

    The reflection output is intentionally action-oriented so the parent graph
    can use it directly for conditional routing and requery loops.

Attributes:
    RecommendedNextAction:
        High-level next action recommended by the reflection agent.
    GapSeverity:
        Severity label for an identified research gap.
    ReflectionAgentModel:
        Shared base model for reflection-agent payloads.
    ResearchGap:
        A specific missing fact, unresolved facet, or incomplete angle.
    ConflictingClaim:
        A specific conflict or contradiction identified in the evidence.
    FollowUpQuery:
        A suggested follow-up query for another research pass.
    ReflectionDecisionBase:
        Base structured reflection output.
    ReflectionDecision:
        Full reflection output including follow-up queries.

Examples:
    .. code-block:: python

        decision = ReflectionDecision(
            is_sufficient=False,
            recommended_next_action="requery",
            rationale=(
                "The evidence covers recent changes but does not clearly "
                "explain user impact."
            ),
            open_gaps=[
                ResearchGap(
                    gap_id="gap_1",
                    description="Missing evidence about how usage patterns changed.",
                    severity="high",
                    affected_subquestion_ids=["sq_2"],
                )
            ],
            conflicting_claims=[],
            followup_queries=[
                FollowUpQuery(
                    query="Tavily LangChain integration recommended usage patterns recent changes",
                    rationale="Target the missing user-impact dimension.",
                    priority=1,
                    target_subquestion_ids=["sq_2"],
                )
            ],
        )
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type RecommendedNextAction = Literal[
    "synthesize",
    "requery",
    "extract",
    "map",
    "crawl",
    "research",
    "clarify",
]
type GapSeverity = Literal["high", "medium", "low"]


class ReflectionAgentModel(BaseModel):
    """Base model for all reflection-agent payloads.

    Returns:
        ReflectionAgentModel: A validated reflection-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(ReflectionAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class ResearchGap(ReflectionAgentModel):
    """A specific missing fact, unresolved facet, or incomplete angle.

    Args:
        gap_id: Stable identifier for the research gap.
        description: Human-readable description of what is missing.
        severity: Severity label for the research gap.
        affected_subquestion_ids: Subquestion identifiers most affected by the gap.

    Returns:
        ResearchGap: A validated research-gap object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> gap = ResearchGap(
        ...     gap_id="gap_1",
        ...     description="Missing evidence about usage-pattern changes.",
        ...     severity="high",
        ...     affected_subquestion_ids=["sq_2"],
        ... )
        >>> gap.severity
        'high'
    """

    gap_id: str = Field(
        min_length=1,
        description="Stable identifier for the research gap.",
    )
    description: str = Field(
        min_length=1,
        description="Human-readable description of the missing evidence or facet.",
    )
    severity: GapSeverity = Field(
        description="Severity label for the research gap.",
    )
    affected_subquestion_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Subquestion identifiers most affected by this missing evidence or gap."
        ),
    )


class ConflictingClaim(ReflectionAgentModel):
    """A conflict or contradiction identified in the evidence.

    Args:
        description: Human-readable description of the conflict.
        claim_a: First conflicting claim or interpretation.
        claim_b: Second conflicting claim or interpretation.
        supporting_urls: URLs most directly associated with the conflict.
        affected_subquestion_ids: Subquestion identifiers affected by the conflict.

    Returns:
        ConflictingClaim: A validated conflicting-claim object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> conflict = ConflictingClaim(
        ...     description="Two sources disagree about release timing.",
        ...     claim_a="The change was introduced in January.",
        ...     claim_b="The change was introduced in March.",
        ... )
        >>> conflict.description
        'Two sources disagree about release timing.'
    """

    description: str = Field(
        min_length=1,
        description="Human-readable description of the conflict or contradiction.",
    )
    claim_a: str = Field(
        min_length=1,
        description="First conflicting claim or interpretation.",
    )
    claim_b: str = Field(
        min_length=1,
        description="Second conflicting claim or interpretation.",
    )
    supporting_urls: list[str] = Field(
        default_factory=list,
        description="URLs most directly associated with the conflicting claims.",
    )
    affected_subquestion_ids: list[str] = Field(
        default_factory=list,
        description="Subquestion identifiers affected by the conflict.",
    )


class FollowUpQuery(ReflectionAgentModel):
    """A suggested follow-up query for another research pass.

    Args:
        query: Search query text for a follow-up research pass.
        rationale: Explanation of why this query is the right next move.
        priority: Priority rank for the follow-up query.
        target_subquestion_ids: Subquestion identifiers this follow-up query addresses.
        recommended_action: Optional retrieval-action hint associated with the query.

    Returns:
        FollowUpQuery: A validated follow-up query object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> followup = FollowUpQuery(
        ...     query="Tavily LangChain integration release notes recent changes",
        ...     rationale="Targets missing primary-source release-note evidence.",
        ...     priority=1,
        ...     target_subquestion_ids=["sq_1"],
        ... )
        >>> followup.priority
        1
    """

    query: str = Field(
        min_length=1,
        max_length=400,
        description="Search query text for a follow-up research pass.",
    )
    rationale: str = Field(
        min_length=1,
        description="Explanation of why this follow-up query is useful.",
    )
    priority: int = Field(
        ge=1,
        description="Priority rank for the follow-up query.",
    )
    target_subquestion_ids: list[str] = Field(
        default_factory=list,
        description="Subquestion identifiers this follow-up query addresses.",
    )
    recommended_action: RecommendedNextAction | None = Field(
        default=None,
        description=(
            "Optional retrieval-action hint associated with the follow-up query."
        ),
    )


class ReflectionDecisionBase(ReflectionAgentModel):
    """Base structured reflection output.

    Args:
        is_sufficient: Whether the currently available evidence is sufficient
            to proceed to final synthesis.
        recommended_next_action: High-level next action recommended for the graph.
        rationale: Explanation of why the decision was made.
        open_gaps: Missing facts, unresolved facets, or incomplete angles.
        conflicting_claims: Conflicts or contradictions identified in the evidence.
        confidence: Confidence score for the reflection decision.

    Returns:
        ReflectionDecisionBase: A validated base reflection-decision object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> decision = ReflectionDecisionBase(
        ...     is_sufficient=True,
        ...     recommended_next_action="synthesize",
        ...     rationale="The available evidence sufficiently covers the main question.",
        ...     open_gaps=[],
        ...     conflicting_claims=[],
        ...     confidence=0.82,
        ... )
        >>> decision.recommended_next_action
        'synthesize'
    """

    is_sufficient: bool = Field(
        description=(
            "Whether the currently available evidence is sufficient to proceed "
            "to final synthesis."
        ),
    )
    recommended_next_action: RecommendedNextAction = Field(
        description="High-level next action recommended for the parent graph.",
    )
    rationale: str = Field(
        min_length=1,
        description="Explanation of why this reflection decision was made.",
    )
    open_gaps: list[ResearchGap] = Field(
        default_factory=list,
        description="Missing facts, unresolved facets, or incomplete angles.",
    )
    conflicting_claims: list[ConflictingClaim] = Field(
        default_factory=list,
        description="Conflicts or contradictions identified in the evidence.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the reflection decision, where 0.0 is very "
            "low confidence and 1.0 is very high confidence."
        ),
    )


class ReflectionDecision(ReflectionDecisionBase):
    """Full structured reflection output.

    Args:
        followup_queries: Suggested follow-up queries for another research pass.
        notes: Optional execution notes for the parent graph.

    Returns:
        ReflectionDecision: A validated full reflection-decision object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> decision = ReflectionDecision(
        ...     is_sufficient=False,
        ...     recommended_next_action="requery",
        ...     rationale="Important gaps remain in the evidence.",
        ...     open_gaps=[],
        ...     conflicting_claims=[],
        ...     confidence=0.74,
        ...     followup_queries=[],
        ... )
        >>> decision.is_sufficient
        False
    """

    followup_queries: list[FollowUpQuery] = Field(
        default_factory=list,
        description="Suggested follow-up queries for another research pass.",
    )
    notes: list[str] = Field(
        default_factory=list,
        description=(
            "Optional execution notes or caveats for the parent graph to consider."
        ),
    )