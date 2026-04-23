"""Structured models for the deep-research answer-synthesis agent.

Purpose:
    Define validated payloads produced by the deep-research answer agent.

Design:
    The answer agent consumes the final research state after planning,
    retrieval, and reflection. Its job is to synthesize a polished markdown
    report grounded in the available evidence and explicit about uncertainty.

    The schema is intentionally report-oriented so downstream code can render a
    good final answer directly while still preserving structured metadata such
    as citations, key findings, and unresolved questions.

Attributes:
    DeepResearchAnswerAgentModel:
        Shared base model for answer-agent payloads.
    AnswerCitation:
        Structured citation used in the final report.
    DeepResearchAnswerBase:
        Base structured answer payload for deep research.
    DeepResearchAnswer:
        Full deep-research answer payload including citations.

Examples:
    .. code-block:: python

        answer = DeepResearchAnswer(
            report_markdown="# Report\\n\\nA grounded answer.",
            executive_summary="Short executive summary.",
            key_findings=["Finding 1", "Finding 2"],
            citations=[],
            confidence=0.78,
            used_search=True,
            evidence_count=7,
            uncertainty_note=None,
            unresolved_questions=[],
        )
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DeepResearchAnswerAgentModel(BaseModel):
    """Base model for all deep-research answer-agent payloads.

    Returns:
        DeepResearchAnswerAgentModel: A validated answer-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(DeepResearchAnswerAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class AnswerCitation(DeepResearchAnswerAgentModel):
    """Structured citation used in the final report.

    Args:
        title: Human-readable title of the cited source.
        url: Canonical URL of the cited source.
        supports: Optional note describing what claim or section the source supports.

    Returns:
        AnswerCitation: A validated citation object.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    title: str = Field(
        min_length=1,
        description="Human-readable title of the cited source.",
    )
    url: str = Field(
        min_length=1,
        description="Canonical URL of the cited source.",
    )
    supports: str | None = Field(
        default=None,
        description=(
            "Optional note describing which claim, finding, or section this "
            "source supports."
        ),
    )


class DeepResearchAnswerBase(DeepResearchAnswerAgentModel):
    """Base structured answer payload for deep research.

    Args:
        report_markdown: Final markdown report shown to the user.
        executive_summary: Short executive summary of the report.
        key_findings: High-signal findings distilled from the evidence.
        confidence: Confidence score between ``0.0`` and ``1.0``.
        used_search: Whether search-derived evidence was used.
        evidence_count: Number of evidence items considered during synthesis.
        uncertainty_note: Optional note describing important uncertainty or caveats.
        unresolved_questions: Open questions or missing angles that remain unresolved.

    Returns:
        DeepResearchAnswerBase: A validated base answer payload.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    report_markdown: str = Field(
        min_length=1,
        description=(
            "Final markdown report shown to the user. It should be clear, "
            "structured, and grounded in the provided evidence."
        ),
    )
    executive_summary: str = Field(
        min_length=1,
        description="Short executive summary of the report.",
    )
    key_findings: list[str] = Field(
        default_factory=list,
        description="High-signal findings distilled from the evidence bundle.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the final report, where 0.0 is very low "
            "confidence and 1.0 is very high confidence."
        ),
    )
    used_search: bool = Field(
        description="Whether search-derived evidence was used in the report.",
    )
    evidence_count: int = Field(
        ge=0,
        description="Number of evidence items considered during synthesis.",
    )
    uncertainty_note: str | None = Field(
        default=None,
        description=(
            "Optional note describing important uncertainty, weak evidence, or "
            "conflicting claims."
        ),
    )
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description="Open questions or missing angles that remain unresolved.",
    )


class DeepResearchAnswer(DeepResearchAnswerBase):
    """Full deep-research answer payload including citations.

    Args:
        citations: Structured citations used to support the report.

    Returns:
        DeepResearchAnswer: A validated final answer payload.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> answer = DeepResearchAnswer(
        ...     report_markdown="# Report\\n\\nA grounded answer.",
        ...     executive_summary="Summary",
        ...     key_findings=[],
        ...     citations=[],
        ...     confidence=0.75,
        ...     used_search=True,
        ...     evidence_count=4,
        ...     uncertainty_note=None,
        ...     unresolved_questions=[],
        ... )
        >>> answer.evidence_count
        4
    """

    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations used to support the final report. Each entry "
            "should correspond to evidence actually present in the workflow state."
        ),
    )