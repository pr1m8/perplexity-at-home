"""Preset models and defaults for Tavily tools.

Purpose:
    Define Tavily-specific preset models and default preset instances.

Design:
    - Extend shared tool models from ``perplexity_at_home.tools.models``.
    - Keep all default Tavily presets centralized here.
    - Let factory functions remain mechanical and side-effect free.

Examples:
    >>> QUICK_SEARCH_PRESET.name
    'quick_search'
    >>> PRO_SEARCH_PRESET.search_depth
    'advanced'
"""

from __future__ import annotations

from pydantic import ConfigDict

from perplexity_at_home.tools.models import (
    ExtractPresetBase,
    ResearchPresetBase,
    SearchPresetBase,
)


class TavilySearchPreset(SearchPresetBase):
    """Tavily-specific search preset.

    Args:
        Inherits all arguments from ``SearchPresetBase``.

    Returns:
        A validated Tavily search preset.

    Raises:
        ValidationError: Raised when the preset is invalid.

    Examples:
        >>> TavilySearchPreset(
        ...     name="quick_search",
        ...     max_results=5,
        ... )
    """

    model_config = ConfigDict(extra="forbid")


class TavilyExtractPreset(ExtractPresetBase):
    """Tavily-specific extraction preset.

    Args:
        Inherits all arguments from ``ExtractPresetBase``.

    Returns:
        A validated Tavily extract preset.

    Raises:
        ValidationError: Raised when the preset is invalid.
    """

    model_config = ConfigDict(extra="forbid")


class TavilyResearchPreset(ResearchPresetBase):
    """Tavily-specific research preset.

    Args:
        Inherits all arguments from ``ResearchPresetBase``.

    Returns:
        A validated Tavily research preset.

    Raises:
        ValidationError: Raised when the preset is invalid.
    """

    model_config = ConfigDict(extra="forbid")


QUICK_SEARCH_PRESET = TavilySearchPreset(
    name="quick_search",
    max_results=5,
    topic="general",
    search_depth="basic",
    include_answer="basic",
    include_raw_content=False,
)

PRO_SEARCH_PRESET = TavilySearchPreset(
    name="pro_search",
    max_results=8,
    topic="general",
    search_depth="advanced",
    include_answer="advanced",
    include_raw_content=True,
)

NEWS_SEARCH_PRESET = TavilySearchPreset(
    name="news_search",
    max_results=8,
    topic="news",
    search_depth="advanced",
    include_answer="advanced",
    include_raw_content=True,
)

QUICK_EXTRACT_PRESET = TavilyExtractPreset(
    name="quick_extract",
    extract_depth="basic",
    include_images=False,
)

PRO_EXTRACT_PRESET = TavilyExtractPreset(
    name="pro_extract",
    extract_depth="advanced",
    include_images=False,
)

DEEP_RESEARCH_PRESET = TavilyResearchPreset(
    name="deep_research",
    model="pro",
    citation_format="numbered",
    stream=False,
    output_schema=None,
)

__all__ = [
    "DEEP_RESEARCH_PRESET",
    "NEWS_SEARCH_PRESET",
    "PRO_EXTRACT_PRESET",
    "PRO_SEARCH_PRESET",
    "QUICK_EXTRACT_PRESET",
    "QUICK_SEARCH_PRESET",
    "TavilyExtractPreset",
    "TavilyResearchPreset",
    "TavilySearchPreset",
]