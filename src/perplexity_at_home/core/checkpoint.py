"""Helpers for LangGraph Postgres checkpoint persistence.

Purpose:
    Provide a small, typed interface for creating and initializing the
    LangGraph async Postgres checkpointer.

Design:
    This module exposes:
    - a connection-string helper
    - an async context manager for runtime use
    - a one-time setup helper for bootstrapping database tables

    The runtime context manager is intentionally separate from setup so that
    normal request or worker execution does not repeatedly perform schema
    initialization unless explicitly requested.

Examples:
    >>> from perplexity_at_home.core.checkpoint import get_checkpointer_uri
    >>> get_checkpointer_uri().startswith("postgresql://")
    True
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from perplexity_at_home.settings import get_settings

__all__ = [
    "checkpoint_context",
    "get_checkpointer_uri",
    "setup_checkpointer",
]


def get_checkpointer_uri() -> str:
    """Return the Postgres URI used by the LangGraph checkpointer.

    Returns:
        The Postgres connection string.

    Examples:
        >>> uri = get_checkpointer_uri()
        >>> "postgresql://" in uri
        True
    """
    return get_settings().postgres.uri


@asynccontextmanager
async def checkpoint_context(*, setup: bool = False) -> AsyncIterator[AsyncPostgresSaver]:
    """Yield an async LangGraph Postgres checkpointer.

    Args:
        setup: Whether to run ``checkpointer.setup()`` before yielding the
            checkpointer. This should usually be ``False`` during normal
            runtime usage and ``True`` during one-time bootstrap.

    Yields:
        The configured ``AsyncPostgresSaver`` instance.

    Raises:
        Exception: Propagates database or connection errors from the
            underlying LangGraph / psycopg layers.

    Examples:
        .. code-block:: python

            async with checkpoint_context() as checkpointer:
                graph = builder.compile(checkpointer=checkpointer)

        .. code-block:: python

            async with checkpoint_context(setup=True):
                pass
    """
    async with AsyncPostgresSaver.from_conn_string(get_checkpointer_uri()) as checkpointer:
        if setup:
            await checkpointer.setup()
        yield checkpointer


async def setup_checkpointer() -> None:
    """Initialize the LangGraph checkpoint tables.

    Returns:
        None

    Raises:
        Exception: Propagates database or connection errors from the
            underlying LangGraph / psycopg layers.

    Examples:
        .. code-block:: python

            await setup_checkpointer()
    """
    async with checkpoint_context(setup=True):
        return