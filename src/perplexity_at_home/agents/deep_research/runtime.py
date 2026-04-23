"""Runtime helpers for the top-level deep-research workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from perplexity_at_home.agents.deep_research.agent import (
    DeepResearchAgent,
    build_deep_research_agent,
)
from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.core.persistence import persistence_context
from perplexity_at_home.utils import get_current_datetime_string

__all__ = [
    "deep_research_agent_context",
    "run_deep_research",
]


def _resolve_context(context: DeepResearchContext | None) -> DeepResearchContext:
    """Fill in a current datetime when the context does not provide one."""
    if context is None:
        return DeepResearchContext(current_datetime=get_current_datetime_string())

    if context.current_datetime is not None:
        return context

    return replace(context, current_datetime=get_current_datetime_string())


@asynccontextmanager
async def deep_research_agent_context(
    *,
    persistent: bool = False,
    setup_persistence: bool = False,
    context: DeepResearchContext | None = None,
    debug: bool = False,
) -> AsyncIterator[DeepResearchAgent]:
    """Yield a deep-research agent with in-memory or Postgres-backed state."""
    resolved_context = _resolve_context(context)

    if not persistent:
        yield build_deep_research_agent(context=resolved_context, debug=debug)
        return

    async with persistence_context(setup=setup_persistence) as (store, checkpointer):
        yield build_deep_research_agent(
            context=resolved_context,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
        )


async def run_deep_research(
    question: str,
    *,
    persistent: bool = False,
    setup_persistence: bool = False,
    context: DeepResearchContext | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the top-level deep-research workflow and return its final state."""
    async with deep_research_agent_context(
        persistent=persistent,
        setup_persistence=setup_persistence,
        context=context,
        debug=debug,
    ) as agent:
        return await agent.ainvoke(question)
