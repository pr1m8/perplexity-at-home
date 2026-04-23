"""Combined LangGraph persistence helpers.

Purpose:
    Provide a layered async context manager that opens both the LangGraph
    Postgres store and Postgres checkpointer together.

Design:
    This module uses ``AsyncExitStack`` so both async context managers are
    opened and closed in one place. It is the most convenient API for:
    - graph compilation with both store and checkpointer
    - bootstrap/setup of both persistence layers
    - application lifespan initialization

Attributes:
    PersistencePair: Tuple alias for the opened store and checkpointer.

Examples:
    .. code-block:: python

        async with persistence_context() as (store, checkpointer):
            graph = builder.compile(
                checkpointer=checkpointer,
                store=store,
            )

    .. code-block:: python

        await setup_persistence()
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore

from perplexity_at_home.core.checkpoint import checkpoint_context
from perplexity_at_home.core.store import store_context

type PersistencePair = tuple[AsyncPostgresStore, AsyncPostgresSaver]

__all__ = [
    "PersistencePair",
    "persistence_context",
    "setup_persistence",
]


@asynccontextmanager
async def persistence_context(*, setup: bool = False) -> AsyncIterator[PersistencePair]:
    """Yield both the LangGraph store and checkpointer together.

    Args:
        setup: Whether to initialize both persistence layers before yielding
            them. This is mainly intended for first-run bootstrap.

    Yields:
        A tuple of ``(store, checkpointer)``.

    Raises:
        Exception: Propagates database or connection errors from the
            underlying LangGraph / psycopg layers.

    Examples:
        .. code-block:: python

            async with persistence_context() as (store, checkpointer):
                graph = builder.compile(
                    checkpointer=checkpointer,
                    store=store,
                )

        .. code-block:: python

            async with persistence_context(setup=True):
                pass
    """
    async with AsyncExitStack() as stack:
        store = await stack.enter_async_context(store_context(setup=setup))
        checkpointer = await stack.enter_async_context(checkpoint_context(setup=setup))
        yield store, checkpointer


async def setup_persistence() -> None:
    """Initialize both the LangGraph store and checkpoint tables.

    Returns:
        None

    Raises:
        Exception: Propagates database or connection errors from the
            underlying LangGraph / psycopg layers.

    Examples:
        .. code-block:: python

            await setup_persistence()
    """
    async with persistence_context(setup=True):
        return