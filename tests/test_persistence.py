from __future__ import annotations

from perplexity_at_home.core.checkpoint import get_checkpointer_uri
from perplexity_at_home.core.store import get_store_uri
from perplexity_at_home.settings import get_settings


def test_checkpoint_and_store_use_shared_postgres_uri(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("POSTGRES_HOST", "db.internal")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "graph_data")
    monkeypatch.setenv("POSTGRES_USER", "postgres")
    monkeypatch.setenv("POSTGRES_PASSWORD", "postgres")
    monkeypatch.setenv("POSTGRES_SSLMODE", "disable")
    get_settings.cache_clear()

    expected_uri = "postgresql://postgres:postgres@db.internal:6543/graph_data?sslmode=disable"

    assert get_checkpointer_uri() == expected_uri
    assert get_store_uri() == expected_uri
