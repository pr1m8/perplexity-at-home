"""Perplexity-at-home package exports."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from perplexity_at_home.agents import (
    DeepResearchAgent,
    DeepResearchContext,
    ProSearchAgent,
    ProSearchContext,
    QuickSearchContext,
    build_deep_research_agent,
    build_deep_research_graph,
    build_pro_search_agent,
    build_pro_search_graph,
    build_quick_search_agent,
    pro_search_agent_context,
    quick_search_agent_context,
    run_pro_search,
    run_quick_search,
)
from perplexity_at_home.agents.deep_research import (
    deep_research_agent_context,
    run_deep_research,
)
from perplexity_at_home.settings import AppSettings, PostgresSettings, get_settings

__all__ = [
    "AppSettings",
    "DeepResearchAgent",
    "DeepResearchContext",
    "PostgresSettings",
    "ProSearchAgent",
    "ProSearchContext",
    "QuickSearchContext",
    "build_deep_research_agent",
    "build_deep_research_graph",
    "build_pro_search_agent",
    "build_pro_search_graph",
    "build_quick_search_agent",
    "deep_research_agent_context",
    "get_settings",
    "pro_search_agent_context",
    "quick_search_agent_context",
    "run_deep_research",
    "run_pro_search",
    "run_quick_search",
]

try:
    __version__ = version("perplexity-at-home")
except PackageNotFoundError:
    __version__ = "0.0.0"
