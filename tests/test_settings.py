from __future__ import annotations

import pytest

import perplexity_at_home.settings as settings_module
from perplexity_at_home.settings import AppSettings


def test_default_model_is_gpt54() -> None:
    settings = AppSettings(_env_file=None)

    assert settings.default_model == "openai:gpt-5.4"
    assert settings.resolved_quick_search_model == "openai:gpt-5.4"
    assert settings.resolved_deep_research_planner_model == "openai:gpt-5.4"
    assert settings.postgres.uri == (
        "postgresql://postgres:postgres@localhost:5442/perplexity_at_home?sslmode=disable"
    )


def test_specific_overrides_fall_back_in_order() -> None:
    settings = AppSettings(
        _env_file=None,
        default_model="openai:gpt-5.4-mini",
        deep_research_model="openai:gpt-5.4",
        deep_research_answer_model="openai:gpt-5.4-mini",
        pro_search_model="openai:gpt-5.4",
    )

    assert settings.resolved_deep_research_query_model == "openai:gpt-5.4"
    assert settings.resolved_deep_research_answer_model == "openai:gpt-5.4-mini"
    assert settings.resolved_pro_search_query_model == "openai:gpt-5.4"


def test_postgres_uri_escapes_credentials() -> None:
    settings = AppSettings(
        _env_file=None,
        postgres_user_raw="local user",
        postgres_password_raw="p@ss word",
        postgres_host_raw="db.internal",
        postgres_port_raw=6543,
        postgres_database_raw="graph_data",
        postgres_sslmode_raw="require",
    )

    assert settings.postgres.uri == (
        "postgresql://local+user:p%40ss+word@db.internal:6543/graph_data?sslmode=require"
    )


def test_empty_env_values_are_ignored(monkeypatch) -> None:
    monkeypatch.setenv("POSTGRES_PORT", "")
    monkeypatch.setenv("PERPLEXITY_AT_HOME_POSTGRES__PORT", "")

    settings = AppSettings(_env_file=None)

    assert settings.postgres.port == 5442


def test_build_chat_model_uses_openai_key(monkeypatch) -> None:
    captured: dict[str, object] = {}
    fake_env: dict[str, str] = {}

    def fake_init_chat_model(model: str | None = None, **kwargs: object) -> str:
        captured["model"] = model
        captured.update(kwargs)
        return "chat-model"

    monkeypatch.setattr(settings_module, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(settings_module.os, "environ", fake_env)

    settings = AppSettings(_env_file=None, openai_api_key="test-openai-key")

    result = settings.build_chat_model(settings.default_model)

    assert result == "chat-model"
    assert captured["model"] == "gpt-5.4"
    assert captured["model_provider"] == "openai"
    assert settings.openai_api_key is not None
    assert captured["api_key"] == settings.openai_api_key
    assert fake_env["OPENAI_API_KEY"] == "test-openai-key"


def test_build_chat_model_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("PERPLEXITY_AT_HOME_OPENAI_API_KEY", raising=False)

    settings = AppSettings(_env_file=None)

    with pytest.raises(RuntimeError, match="OpenAI API key is required"):
        settings.build_chat_model(settings.default_model)


def test_apply_runtime_environment_exports_tracing_vars(monkeypatch) -> None:
    fake_env: dict[str, str] = {}
    monkeypatch.setattr(settings_module.os, "environ", fake_env)

    settings = AppSettings(
        _env_file=None,
        langchain_tracing_v2=True,
        langchain_project="perplexity-at-home",
        langsmith_api_key="test-langsmith-key",
    )

    settings.apply_runtime_environment()

    assert fake_env["LANGCHAIN_TRACING_V2"] == "true"
    assert fake_env["LANGCHAIN_PROJECT"] == "perplexity-at-home"
    assert fake_env["LANGSMITH_API_KEY"] == "test-langsmith-key"
