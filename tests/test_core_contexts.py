from __future__ import annotations

from contextlib import asynccontextmanager

import pytest

import perplexity_at_home.core.checkpoint as checkpoint_module
import perplexity_at_home.core.persistence as persistence_module
import perplexity_at_home.core.store as store_module


class DummyAsyncResource:
    def __init__(self) -> None:
        self.setup_calls = 0

    async def setup(self) -> None:
        self.setup_calls += 1


@pytest.mark.asyncio
async def test_checkpoint_context_runs_setup(monkeypatch) -> None:
    resource = DummyAsyncResource()

    @asynccontextmanager
    async def fake_from_conn_string(uri: str):
        assert uri.startswith("postgresql://")
        yield resource

    monkeypatch.setattr(
        checkpoint_module.AsyncPostgresSaver,
        "from_conn_string",
        fake_from_conn_string,
    )
    monkeypatch.setattr(
        checkpoint_module,
        "get_checkpointer_uri",
        lambda: "postgresql://postgres:postgres@localhost:5442/perplexity_at_home?sslmode=disable",
    )

    async with checkpoint_module.checkpoint_context(setup=True) as checkpointer:
        assert checkpointer is resource

    assert resource.setup_calls == 1


@pytest.mark.asyncio
async def test_store_context_runs_setup(monkeypatch) -> None:
    resource = DummyAsyncResource()

    @asynccontextmanager
    async def fake_from_conn_string(uri: str):
        assert uri.startswith("postgresql://")
        yield resource

    monkeypatch.setattr(
        store_module.AsyncPostgresStore,
        "from_conn_string",
        fake_from_conn_string,
    )
    monkeypatch.setattr(
        store_module,
        "get_store_uri",
        lambda: "postgresql://postgres:postgres@localhost:5442/perplexity_at_home?sslmode=disable",
    )

    async with store_module.store_context(setup=True) as store:
        assert store is resource

    assert resource.setup_calls == 1


@pytest.mark.asyncio
async def test_persistence_context_yields_store_and_checkpointer(monkeypatch) -> None:
    store = object()
    checkpointer = object()

    @asynccontextmanager
    async def fake_store_context(*, setup: bool = False):
        assert setup is True
        yield store

    @asynccontextmanager
    async def fake_checkpoint_context(*, setup: bool = False):
        assert setup is True
        yield checkpointer

    monkeypatch.setattr(persistence_module, "store_context", fake_store_context)
    monkeypatch.setattr(persistence_module, "checkpoint_context", fake_checkpoint_context)

    async with persistence_module.persistence_context(setup=True) as resources:
        assert resources == (store, checkpointer)


@pytest.mark.asyncio
async def test_setup_helpers_delegate_to_contexts(monkeypatch) -> None:
    called = {
        "checkpoint": False,
        "store": False,
        "persistence": False,
    }

    @asynccontextmanager
    async def fake_checkpoint_context(*, setup: bool = False):
        called["checkpoint"] = setup
        yield object()

    @asynccontextmanager
    async def fake_store_context(*, setup: bool = False):
        called["store"] = setup
        yield object()

    @asynccontextmanager
    async def fake_persistence_context(*, setup: bool = False):
        called["persistence"] = setup
        yield (object(), object())

    monkeypatch.setattr(checkpoint_module, "checkpoint_context", fake_checkpoint_context)
    monkeypatch.setattr(store_module, "store_context", fake_store_context)
    monkeypatch.setattr(persistence_module, "persistence_context", fake_persistence_context)

    await checkpoint_module.setup_checkpointer()
    await store_module.setup_store()
    await persistence_module.setup_persistence()

    assert called == {
        "checkpoint": True,
        "store": True,
        "persistence": True,
    }
