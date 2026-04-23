"""Tavily integration layer.

Purpose:
    Re-export the public Tavily tool-integration surface for the application.
"""

from perplexity_at_home.tools.tavily.bundles import (
    build_deep_bundle,
    build_pro_bundle,
    build_quick_bundle,
)
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
from perplexity_at_home.tools.tavily.normalize import (
    extract_answer,
    normalize_search_hit,
    normalize_search_payload,
)
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

__all__ = [
    "DEEP_RESEARCH_PRESET",
    "PRO_EXTRACT_PRESET",
    "PRO_SEARCH_PRESET",
    "QUICK_EXTRACT_PRESET",
    "QUICK_SEARCH_PRESET",
    "TavilyExtractPreset",
    "TavilyResearchPreset",
    "TavilySearchPreset",
    "build_crawl_tool",
    "build_deep_bundle",
    "build_extract_tool",
    "build_get_research_tool",
    "build_map_tool",
    "build_pro_bundle",
    "build_pro_extract_tool",
    "build_pro_search_tool",
    "build_quick_bundle",
    "build_research_tool",
    "build_search_tool",
    "extract_answer",
    "normalize_search_hit",
    "normalize_search_payload",
]