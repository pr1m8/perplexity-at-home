"""Runtime helpers for the top-level pro-search workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from perplexity_at_home.agents.pro_search.agent import (
    ProSearchAgent,
    build_pro_search_agent,
)
from perplexity_at_home.agents.pro_search.context import ProSearchContext
from perplexity_at_home.core.persistence import persistence_context
from perplexity_at_home.utils import get_current_datetime_string

__all__ = [
    "pro_search_agent_context",
    "run_pro_search",
]


def _resolve_context(context: ProSearchContext | None) -> ProSearchContext:
    """Fill in a current datetime when the context does not provide one."""
    if context is None:
        return ProSearchContext(current_datetime=get_current_datetime_string())

    if context.current_datetime is not None:
        return context

    return replace(context, current_datetime=get_current_datetime_string())


@asynccontextmanager
async def pro_search_agent_context(
    *,
    persistent: bool = False,
    setup_persistence: bool = False,
    context: ProSearchContext | None = None,
    debug: bool = False,
) -> AsyncIterator[ProSearchAgent]:
    """Yield a pro-search agent with in-memory or Postgres-backed state."""
    resolved_context = _resolve_context(context)

    if not persistent:
        yield build_pro_search_agent(context=resolved_context, debug=debug)
        return

    async with persistence_context(setup=setup_persistence) as (store, checkpointer):
        yield build_pro_search_agent(
            context=resolved_context,
            checkpointer=checkpointer,
            store=store,
            debug=debug,
        )


async def run_pro_search(
    question: str,
    *,
    persistent: bool = False,
    setup_persistence: bool = False,
    context: ProSearchContext | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the top-level pro-search workflow and return its final state."""
    async with pro_search_agent_context(
        persistent=persistent,
        setup_persistence=setup_persistence,
        context=context,
        debug=debug,
    ) as agent:
        return await agent.ainvoke(question)
