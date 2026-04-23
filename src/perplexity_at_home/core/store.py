"""Helpers for LangGraph Postgres long-term store persistence.

Purpose:
    Provide a small, typed interface for creating and initializing the
    LangGraph async Postgres store.

Design:
    This module mirrors the checkpoint helper module so both persistence
    layers have the same ergonomics:
    - URI accessor
    - async runtime context manager
    - one-time setup helper

Examples:
    >>> from perplexity_at_home.core.store import get_store_uri
    >>> get_store_uri().startswith("postgresql://")
    True
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.store.postgres.aio import AsyncPostgresStore

from perplexity_at_home.settings import get_settings

__all__ = [
    "get_store_uri",
    "setup_store",
    "store_context",
]


def get_store_uri() -> str:
    """Return the Postgres URI used by the LangGraph store.

    Returns:
        The Postgres connection string.

    Examples:
        >>> uri = get_store_uri()
        >>> "postgresql://" in uri
        True
    """
    return get_settings().postgres.uri


@asynccontextmanager
async def store_context(*, setup: bool = False) -> AsyncIterator[AsyncPostgresStore]:
    """Yield an async LangGraph Postgres store.

    Args:
        setup: Whether to run ``store.setup()`` before yielding the store.
            This should usually be ``False`` during normal runtime usage and
            ``True`` during one-time bootstrap.

    Yields:
        The configured ``AsyncPostgresStore`` instance.

    Raises:
        Exception: Propagates database or connection errors from the
            underlying LangGraph / psycopg layers.

    Examples:
        .. code-block:: python

            async with store_context() as store:
                value = await store.aget(("memories",), "user-123")

        .. code-block:: python

            async with store_context(setup=True):
                pass
    """
    async with AsyncPostgresStore.from_conn_string(get_store_uri()) as store:
        if setup:
            await store.setup()
        yield store


async def setup_store() -> None:
    """Initialize the LangGraph store tables.

    Returns:
        None

    Raises:
        Exception: Propagates database or connection errors from the
            underlying LangGraph / psycopg layers.

    Examples:
        .. code-block:: python

            await setup_store()
    """
    async with store_context(setup=True):
        return