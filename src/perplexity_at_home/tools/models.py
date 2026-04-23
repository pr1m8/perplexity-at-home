"""Shared models for external tool integrations.

Purpose:
    Define reusable validated configuration models shared by tool integration
    packages.

Design:
    - Keep this module provider-agnostic where possible.
    - Use small Pydantic models for stable constructor/config payloads.
    - Re-export or extend these models in provider-specific packages.

Examples:
    >>> SearchPresetBase(
    ...     name="quick_search",
    ...     max_results=5,
    ... )
    SearchPresetBase(name='quick_search', max_results=5)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SearchTopic = Literal["general", "news", "finance"]
SearchDepth = Literal["basic", "advanced"]
ExtractDepth = Literal["basic", "advanced"]
ResearchModel = Literal["mini", "pro", "auto"]
CitationFormat = Literal["numbered", "mla", "apa", "chicago"]
ToolBundleName = Literal["quick", "pro", "deep"]


class SearchPresetBase(BaseModel):
    """Base search configuration for web-search integrations.

    Args:
        name: Human-friendly preset name.
        max_results: Maximum number of search results.
        topic: Search topic category.
        search_depth: Search depth or quality tier.
        include_answer: Whether to include an answer field.
        include_raw_content: Whether to include raw content.

    Returns:
        A validated preset model.

    Raises:
        ValidationError: If field values are invalid.

    Examples:
        >>> SearchPresetBase(
        ...     name="quick_search",
        ...     max_results=5,
        ...     topic="general",
        ...     search_depth="basic",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    max_results: int = Field(ge=1, le=20)
    topic: SearchTopic = "general"
    search_depth: SearchDepth = "basic"
    include_answer: bool | Literal["basic", "advanced"] = False
    include_raw_content: bool = False


class ExtractPresetBase(BaseModel):
    """Base extraction configuration for page-content tools.

    Args:
        name: Human-friendly preset name.
        extract_depth: Extraction depth.
        include_images: Whether images should be included.

    Returns:
        A validated preset model.

    Raises:
        ValidationError: If field values are invalid.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    extract_depth: ExtractDepth = "basic"
    include_images: bool = False


class ResearchPresetBase(BaseModel):
    """Base deep-research configuration for task-oriented tools.

    Args:
        name: Human-friendly preset name.
        model: Research model tier.
        citation_format: Citation style for returned research.
        stream: Whether results should be streamed.
        output_schema: Optional structured output schema.

    Returns:
        A validated preset model.

    Raises:
        ValidationError: If field values are invalid.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    model: ResearchModel = "auto"
    citation_format: CitationFormat = "numbered"
    stream: bool = False
    output_schema: dict[str, object] | None = None


class ToolRef(BaseModel):
    """Simple named reference to a constructed tool.

    Args:
        name: Logical name used by the application.
        provider: Provider or integration name.
        kind: Tool kind within that provider.

    Returns:
        A validated tool reference.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    kind: str