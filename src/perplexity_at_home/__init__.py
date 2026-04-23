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
    "run_deep_research",
]

try:
    __version__ = version("perplexity-at-home")
except PackageNotFoundError:
    __version__ = "0.0.0"
