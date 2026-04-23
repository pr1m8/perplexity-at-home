"""Factory functions for Tavily LangChain tools.

Purpose:
    Construct configured Tavily LangChain tools from preset models.

Design:
    - Keep this module mechanical and side-effect free.
    - Accept preset models and return concrete LangChain tool instances.
    - Leave orchestration and tool selection to higher layers.

Examples:
    >>> tool = build_search_tool()
    >>> tool is not None
    True
"""

from __future__ import annotations

from langchain_tavily import (
    TavilyCrawl,
    TavilyExtract,
    TavilyGetResearch,
    TavilyMap,
    TavilyResearch,
    TavilySearch,
)
from langchain_tavily._utilities import (
    TavilyCrawlAPIWrapper,
    TavilyExtractAPIWrapper,
    TavilyMapAPIWrapper,
    TavilyResearchAPIWrapper,
    TavilySearchAPIWrapper,
)

from perplexity_at_home.settings import get_settings
from perplexity_at_home.tools.tavily.presets import (
    DEEP_RESEARCH_PRESET,
    PRO_EXTRACT_PRESET,
    PRO_SEARCH_PRESET,
    QUICK_EXTRACT_PRESET,
    QUICK_SEARCH_PRESET,
    TavilyExtractPreset,
    TavilyResearchPreset,
    TavilySearchPreset,
)


def build_search_tool(
    preset: TavilySearchPreset = QUICK_SEARCH_PRESET,
) -> TavilySearch:
    """Build a Tavily search tool.

    Args:
        preset: Search preset to apply.

    Returns:
        A configured ``TavilySearch`` tool.
    """
    settings = get_settings()
    return TavilySearch(
        max_results=preset.max_results,
        topic=preset.topic,
        search_depth=preset.search_depth,
        include_answer=preset.include_answer,
        include_raw_content=preset.include_raw_content,
        api_wrapper=TavilySearchAPIWrapper(
            tavily_api_key=settings.require_tavily_api_key(),
        ),
    )


def build_pro_search_tool() -> TavilySearch:
    """Build the default pro-search Tavily tool.

    Returns:
        A configured ``TavilySearch`` using the pro preset.
    """
    return build_search_tool(PRO_SEARCH_PRESET)


def build_extract_tool(
    preset: TavilyExtractPreset = QUICK_EXTRACT_PRESET,
) -> TavilyExtract:
    """Build a Tavily extract tool.

    Args:
        preset: Extract preset to apply.

    Returns:
        A configured ``TavilyExtract`` tool.
    """
    settings = get_settings()
    return TavilyExtract(
        extract_depth=preset.extract_depth,
        include_images=preset.include_images,
        apiwrapper=TavilyExtractAPIWrapper(
            tavily_api_key=settings.require_tavily_api_key(),
        ),
    )


def build_pro_extract_tool() -> TavilyExtract:
    """Build the default pro extract Tavily tool.

    Returns:
        A configured ``TavilyExtract`` using the pro preset.
    """
    return build_extract_tool(PRO_EXTRACT_PRESET)


def build_map_tool() -> TavilyMap:
    """Build a Tavily map tool.

    Returns:
        A configured ``TavilyMap`` tool.
    """
    settings = get_settings()
    return TavilyMap(
        api_wrapper=TavilyMapAPIWrapper(
            tavily_api_key=settings.require_tavily_api_key(),
        )
    )


def build_crawl_tool() -> TavilyCrawl:
    """Build a Tavily crawl tool.

    Returns:
        A configured ``TavilyCrawl`` tool.
    """
    settings = get_settings()
    return TavilyCrawl(
        api_wrapper=TavilyCrawlAPIWrapper(
            tavily_api_key=settings.require_tavily_api_key(),
        )
    )


def build_research_tool(
    preset: TavilyResearchPreset = DEEP_RESEARCH_PRESET,
) -> TavilyResearch:
    """Build a Tavily deep-research tool.

    Args:
        preset: Research preset to apply.

    Returns:
        A configured ``TavilyResearch`` tool.
    """
    settings = get_settings()
    return TavilyResearch(
        model=preset.model,
        citation_format=preset.citation_format,
        stream=preset.stream,
        output_schema=preset.output_schema,
        api_wrapper=TavilyResearchAPIWrapper(
            tavily_api_key=settings.require_tavily_api_key(),
        ),
    )


def build_get_research_tool() -> TavilyGetResearch:
    """Build a Tavily research-result retrieval tool.

    Returns:
        A configured ``TavilyGetResearch`` tool.
    """
    settings = get_settings()
    return TavilyGetResearch(
        api_wrapper=TavilyResearchAPIWrapper(
            tavily_api_key=settings.require_tavily_api_key(),
        )
    )
