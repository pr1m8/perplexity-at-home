"""Structured models for the pro-search answer-synthesis agent.

Purpose:
    Define validated payloads produced by the pro-search answer agent.

Design:
    - Uses a small local base model for shared Pydantic configuration.
    - Produces markdown-oriented structured output rather than free-form text.
    - Keeps citations explicit so the parent workflow can inspect, test, and
      serialize them cleanly.
    - Is intended to consume aggregated evidence from the parent pro-search
      graph rather than calling tools directly.

Attributes:
    ProSearchAnswerAgentModel:
        Shared base model for answer-agent payloads.
    AnswerCitation:
        Structured citation used to support the synthesized answer.
    ProSearchAnswerBase:
        Base answer payload for markdown synthesis.
    ProSearchAnswer:
        Full answer payload including citations and supporting metadata.

Examples:
    .. code-block:: python

        answer = ProSearchAnswer(
            answer_markdown="Tavily's LangChain integration recently added ...",
            citations=[
                AnswerCitation(
                    title="Official docs",
                    url="https://docs.tavily.com/documentation/integrations/langchain",
                    supports="Overview of current LangChain integration surface.",
                ),
            ],
            confidence=0.86,
            used_search=True,
            uncertainty_note=None,
            evidence_count=5,
            unresolved_questions=[],
        )
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProSearchAnswerAgentModel(BaseModel):
    """Base model for all pro-search answer-agent payloads.

    Returns:
        ProSearchAnswerAgentModel: A validated answer-agent model.

    Raises:
        ValidationError: Raised when model validation fails.

    Examples:
        >>> class ExamplePayload(ProSearchAnswerAgentModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class AnswerCitation(ProSearchAnswerAgentModel):
    """Structured citation used in a synthesized pro-search answer.

    Args:
        title: Human-readable title of the cited source.
        url: Canonical URL of the cited source.
        supports: Optional note describing what claim, fact, or section the
            source supports.

    Returns:
        AnswerCitation: A validated citation object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> citation = AnswerCitation(
        ...     title="Official Tavily LangChain docs",
        ...     url="https://docs.tavily.com/documentation/integrations/langchain",
        ...     supports="Current integration overview.",
        ... )
        >>> citation.title
        'Official Tavily LangChain docs'
    """

    title: str = Field(
        min_length=1,
        description="Human-readable title of the source cited in the answer.",
    )
    url: str = Field(
        min_length=1,
        description="Canonical URL of the cited source.",
    )
    supports: str | None = Field(
        default=None,
        description=(
            "Optional short note describing which claim, fact, or section this "
            "source supports."
        ),
    )


class ProSearchAnswerBase(ProSearchAnswerAgentModel):
    """Base structured answer payload for pro-search synthesis.

    Args:
        answer_markdown: Final markdown answer shown to the user.
        confidence: Confidence score from ``0.0`` to ``1.0``.
        used_search: Whether downstream search evidence was used.
        uncertainty_note: Optional note describing ambiguity, conflicting
            evidence, or verification limitations.
        evidence_count: Number of aggregated evidence items the answer relied on.
        unresolved_questions: Optional list of remaining open questions, missing
            facets, or ambiguities that could not be fully resolved.

    Returns:
        ProSearchAnswerBase: A validated base answer payload.

    Raises:
        ValidationError: Raised if any field is invalid.
    """

    answer_markdown: str = Field(
        min_length=1,
        description=(
            "Final markdown answer shown to the user. This should directly "
            "answer the question, stay organized, and cite evidence when appropriate."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the final synthesized answer, where 0.0 is "
            "very low confidence and 1.0 is very high confidence."
        ),
    )
    used_search: bool = Field(
        description=(
            "Whether the answer relied on aggregated search evidence from the "
            "parent pro-search workflow."
        ),
    )
    uncertainty_note: str | None = Field(
        default=None,
        description=(
            "Optional note describing weak evidence, conflicting sources, or "
            "important unresolved uncertainty."
        ),
    )
    evidence_count: int = Field(
        ge=0,
        description=(
            "Number of aggregated evidence items considered during answer synthesis."
        ),
    )
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Outstanding open questions, missing facts, or ambiguities that "
            "could not be fully resolved from the available evidence."
        ),
    )


class ProSearchAnswer(ProSearchAnswerBase):
    """Full structured answer payload for the pro-search answer agent.

    Args:
        citations: Structured citations used to support the final answer.

    Returns:
        ProSearchAnswer: A validated final answer payload.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> answer = ProSearchAnswer(
        ...     answer_markdown="A structured markdown answer.",
        ...     citations=[],
        ...     confidence=0.75,
        ...     used_search=True,
        ...     uncertainty_note=None,
        ...     evidence_count=4,
        ...     unresolved_questions=[],
        ... )
        >>> answer.evidence_count
        4
    """

    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations used to support the final answer. Each entry "
            "should correspond to a source actually represented in the evidence."
        ),
    )