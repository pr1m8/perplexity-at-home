from __future__ import annotations

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
