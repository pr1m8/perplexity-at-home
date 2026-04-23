"""Named Tavily tool bundles.

Purpose:
    Group Tavily tools into reusable bundles for quick search, pro search, and
    deep research.

Design:
    - Bundles return logical names mapped to concrete tools.
    - Higher layers may expose all or a subset of a bundle to a model.

Examples:
    >>> bundle = build_quick_bundle()
    >>> sorted(bundle.keys())
    ['extract', 'search']
"""

from __future__ import annotations

from langchain_core.tools import BaseTool

from perplexity_at_home.tools.tavily.factories import (
    build_crawl_tool,
    build_extract_tool,
    build_get_research_tool,
    build_map_tool,
    build_pro_extract_tool,
    build_pro_search_tool,
    build_research_tool,
    build_search_tool,
)


def build_quick_bundle() -> dict[str, BaseTool]:
    """Build the quick-search Tavily bundle.

    Returns:
        A mapping from logical tool name to tool instance.
    """
    return {
        "search": build_search_tool(),
        "extract": build_extract_tool(),
    }


def build_pro_bundle() -> dict[str, BaseTool]:
    """Build the pro-search Tavily bundle.

    Returns:
        A mapping from logical tool name to tool instance.
    """
    return {
        "search": build_pro_search_tool(),
        "extract": build_pro_extract_tool(),
        "map": build_map_tool(),
        "crawl": build_crawl_tool(),
    }


def build_deep_bundle() -> dict[str, BaseTool]:
    """Build the deep-research Tavily bundle.

    Returns:
        A mapping from logical tool name to tool instance.
    """
    return {
        "search": build_pro_search_tool(),
        "extract": build_pro_extract_tool(),
        "map": build_map_tool(),
        "crawl": build_crawl_tool(),
        "research": build_research_tool(),
        "get_research": build_get_research_tool(),
    }