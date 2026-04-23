"""Runtime helpers for the top-level quick-search workflow."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any

from perplexity_at_home.agents.quick_search.agent import build_quick_search_agent
from perplexity_at_home.agents.quick_search.context import QuickSearchContext
from perplexity_at_home.core.persistence import persistence_context
from perplexity_at_home.utils import get_current_datetime_string

__all__ = [
    "quick_search_agent_context",
    "run_quick_search",
]


def _resolve_context(context: QuickSearchContext | None) -> QuickSearchContext:
    """Fill in a current datetime when the context does not provide one."""
    if context is None:
        return QuickSearchContext(current_datetime=get_current_datetime_string())

    if context.current_datetime is not None:
        return context

    return replace(context, current_datetime=get_current_datetime_string())


@asynccontextmanager
async def quick_search_agent_context(
    *,
    persistent: bool = False,
    setup_persistence: bool = False,
    debug: bool = False,
) -> AsyncIterator[Any]:
    """Yield a quick-search agent with in-memory or Postgres-backed state."""
    if not persistent:
        yield build_quick_search_agent(debug=debug)
        return

    async with persistence_context(setup=setup_persistence) as (store, checkpointer):
        yield build_quick_search_agent(
            checkpointer=checkpointer,
            store=store,
            debug=debug,
        )


async def run_quick_search(
    question: str,
    *,
    thread_id: str = "quick-search",
    persistent: bool = False,
    setup_persistence: bool = False,
    context: QuickSearchContext | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the top-level quick-search workflow and return its final state."""
    resolved_context = _resolve_context(context)
    async with quick_search_agent_context(
        persistent=persistent,
        setup_persistence=setup_persistence,
        debug=debug,
    ) as agent:
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            context=resolved_context,
            config={"configurable": {"thread_id": thread_id}},
        )
