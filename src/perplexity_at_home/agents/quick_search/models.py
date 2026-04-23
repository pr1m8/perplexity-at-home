"""Structured output models for the quick-search agent.

Purpose:
    Define validated response payloads returned by the quick-search agent.

Design:
    These models are used for structured output from the quick-search agent and
    are intentionally designed to be composable. A shared base model provides
    consistent configuration, a citation model captures supporting evidence, and
    the final answer model builds on a small answer base class.

Attributes:
    QuickSearchModel:
        Shared Pydantic base model for quick-search payloads.
    AnswerCitation:
        Structured citation attached to the final answer.
    QuickSearchAnswerBase:
        Base fields common to quick-search answer payloads.
    QuickSearchAnswer:
        Final structured answer including citations.

Examples:
    .. code-block:: python

        answer = QuickSearchAnswer(
            answer_markdown="Apple closed at $123.45. [Source](https://example.com)",
            citations=[
                AnswerCitation(
                    title="Example Source",
                    url="https://example.com",
                    supports="Closing price data for Apple.",
                )
            ],
            confidence=0.92,
            used_search=True,
            uncertainty_note=None,
        )
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class QuickSearchModel(BaseModel):
    """Base model for all quick-search structured payloads.

    This model centralizes shared Pydantic configuration for quick-search
    response payloads.

    Returns:
        QuickSearchModel: A validated quick-search model instance.

    Raises:
        ValidationError: Raised if model validation fails.

    Examples:
        >>> class ExamplePayload(QuickSearchModel):
        ...     value: str
        ...
        >>> ExamplePayload(value="ok").value
        'ok'
    """

    model_config = ConfigDict(extra="forbid")


class AnswerCitation(QuickSearchModel):
    """Citation used to support a quick-search answer.

    Args:
        title: Human-readable title of the supporting source.
        url: Canonical URL of the supporting source.
        supports: Optional short explanation of what specific claim or fact the
            source supports.

    Returns:
        AnswerCitation: A validated citation object.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> citation = AnswerCitation(
        ...     title="Investor Relations Release",
        ...     url="https://example.com/release",
        ...     supports="Official statement about a product launch.",
        ... )
        >>> citation.title
        'Investor Relations Release'
    """

    title: str = Field(
        min_length=1,
        description="Human-readable title of the source used to support the answer.",
    )
    url: str = Field(
        min_length=1,
        description="Canonical URL of the source cited in the final answer.",
    )
    supports: str | None = Field(
        default=None,
        description=(
            "Optional short explanation of which claim, number, or statement "
            "this source supports."
        ),
    )


class QuickSearchAnswerBase(QuickSearchModel):
    """Base final-answer shape for quick-search outputs.

    Args:
        answer_markdown: Final answer rendered in markdown.
        confidence: Model-estimated confidence score between ``0.0`` and ``1.0``.
        used_search: Whether Tavily search or extraction was used to produce the
            answer.
        uncertainty_note: Optional note describing ambiguity, conflicting
            evidence, weak sourcing, or other limitations.

    Returns:
        QuickSearchAnswerBase: A validated answer payload.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> answer = QuickSearchAnswerBase(
        ...     answer_markdown="A concise answer.",
        ...     confidence=0.8,
        ...     used_search=True,
        ... )
        >>> answer.used_search
        True
    """

    answer_markdown: str = Field(
        min_length=1,
        description=(
            "Final markdown answer shown to the user. This should directly "
            "answer the question and include citations inline when appropriate."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score for the final answer, where 0.0 indicates very "
            "low confidence and 1.0 indicates very high confidence."
        ),
    )
    used_search: bool = Field(
        description=(
            "Whether the agent used Tavily search or extraction tools while "
            "producing the answer."
        ),
    )
    uncertainty_note: str | None = Field(
        default=None,
        description=(
            "Optional note describing uncertainty, conflicting evidence, weak "
            "sources, or verification limitations."
        ),
    )


class QuickSearchAnswer(QuickSearchAnswerBase):
    """Full quick-search answer with citations.

    Args:
        citations: Structured list of citations used to support the final
            answer.

    Returns:
        QuickSearchAnswer: A validated final quick-search answer payload.

    Raises:
        ValidationError: Raised if any field is invalid.

    Examples:
        >>> answer = QuickSearchAnswer(
        ...     answer_markdown="Apple closed at $123.45. [Source](https://example.com)",
        ...     citations=[
        ...         AnswerCitation(
        ...             title="Example Source",
        ...             url="https://example.com",
        ...         )
        ...     ],
        ...     confidence=0.9,
        ...     used_search=True,
        ... )
        >>> len(answer.citations)
        1
    """

    citations: list[AnswerCitation] = Field(
        default_factory=list,
        description=(
            "Structured citations used to support the final answer. Each entry "
            "should correspond to a source actually used by the agent."
        ),
    )