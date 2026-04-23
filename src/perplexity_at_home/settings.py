"""Application settings and model selection helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import AliasChoices, BaseModel, Field, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

type SSLMode = Literal[
    "disable",
    "allow",
    "prefer",
    "require",
    "verify-ca",
    "verify-full",
]


class PostgresSettings(BaseModel):
    """Typed Postgres configuration for LangGraph persistence."""

    host: str = "localhost"
    port: int = 5442
    database: str = "perplexity_at_home"
    user: str = "postgres"
    password: SecretStr = SecretStr("postgres")
    sslmode: SSLMode = "disable"

    @computed_field(return_type=str)
    @property
    def uri(self) -> str:
        """Return the Postgres URI used by LangGraph."""
        username = quote_plus(self.user)
        encoded_password = quote_plus(self.password.get_secret_value())
        return (
            "postgresql://"
            f"{username}:{encoded_password}@{self.host}:{self.port}/"
            f"{self.database}?sslmode={self.sslmode}"
        )


class AppSettings(BaseSettings):
    """Environment-backed application settings."""

    model_config = SettingsConfigDict(
        env_prefix="PERPLEXITY_AT_HOME_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_ignore_empty=True,
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    env: Literal["development", "test", "staging", "production"] = "development"
    debug: bool = True
    langgraph_strict_msgpack: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "LANGGRAPH_STRICT_MSGPACK",
            "PERPLEXITY_AT_HOME_LANGGRAPH_STRICT_MSGPACK",
        ),
    )
    langchain_tracing_v2: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "LANGCHAIN_TRACING_V2",
            "PERPLEXITY_AT_HOME_LANGCHAIN_TRACING_V2",
        ),
    )
    langchain_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGCHAIN_PROJECT",
            "PERPLEXITY_AT_HOME_LANGCHAIN_PROJECT",
        ),
    )
    langsmith_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "LANGSMITH_API_KEY",
            "PERPLEXITY_AT_HOME_LANGSMITH_API_KEY",
        ),
    )

    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "PERPLEXITY_AT_HOME_OPENAI_API_KEY"),
    )
    tavily_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY", "PERPLEXITY_AT_HOME_TAVILY_API_KEY"),
    )

    default_model: str = "openai:gpt-5.4"

    quick_search_model: str | None = None

    pro_search_model: str | None = None
    pro_search_query_model: str | None = None
    pro_search_answer_model: str | None = None

    deep_research_model: str | None = None
    deep_research_planner_model: str | None = None
    deep_research_query_model: str | None = None
    deep_research_retrieval_model: str | None = None
    deep_research_reflection_model: str | None = None
    deep_research_answer_model: str | None = None

    postgres_host_raw: str = Field(
        default="localhost",
        validation_alias=AliasChoices("POSTGRES_HOST", "PERPLEXITY_AT_HOME_POSTGRES__HOST"),
        exclude=True,
    )
    postgres_port_raw: int = Field(
        default=5442,
        validation_alias=AliasChoices("POSTGRES_PORT", "PERPLEXITY_AT_HOME_POSTGRES__PORT"),
        exclude=True,
    )
    postgres_database_raw: str = Field(
        default="perplexity_at_home",
        validation_alias=AliasChoices("POSTGRES_DB", "PERPLEXITY_AT_HOME_POSTGRES__DATABASE"),
        exclude=True,
    )
    postgres_user_raw: str = Field(
        default="postgres",
        validation_alias=AliasChoices("POSTGRES_USER", "PERPLEXITY_AT_HOME_POSTGRES__USER"),
        exclude=True,
    )
    postgres_password_raw: SecretStr = Field(
        default=SecretStr("postgres"),
        validation_alias=AliasChoices(
            "POSTGRES_PASSWORD",
            "PERPLEXITY_AT_HOME_POSTGRES__PASSWORD",
        ),
        exclude=True,
    )
    postgres_sslmode_raw: SSLMode = Field(
        default="disable",
        validation_alias=AliasChoices(
            "POSTGRES_SSLMODE",
            "PERPLEXITY_AT_HOME_POSTGRES__SSLMODE",
        ),
        exclude=True,
    )

    @computed_field(return_type=PostgresSettings)
    @property
    def postgres(self) -> PostgresSettings:
        """Return the typed Postgres persistence configuration."""
        return PostgresSettings(
            host=self.postgres_host_raw,
            port=self.postgres_port_raw,
            database=self.postgres_database_raw,
            user=self.postgres_user_raw,
            password=self.postgres_password_raw,
            sslmode=self.postgres_sslmode_raw,
        )

    @property
    def resolved_quick_search_model(self) -> str:
        """Return the configured quick-search model."""
        return self.quick_search_model or self.default_model

    @property
    def resolved_pro_search_query_model(self) -> str:
        """Return the configured pro-search query-planning model."""
        return self.pro_search_query_model or self.pro_search_model or self.default_model

    @property
    def resolved_pro_search_answer_model(self) -> str:
        """Return the configured pro-search answer model."""
        return self.pro_search_answer_model or self.pro_search_model or self.default_model

    @property
    def resolved_deep_research_planner_model(self) -> str:
        """Return the configured deep-research planner model."""
        return (
            self.deep_research_planner_model
            or self.deep_research_model
            or self.default_model
        )

    @property
    def resolved_deep_research_query_model(self) -> str:
        """Return the configured deep-research query-planning model."""
        return self.deep_research_query_model or self.deep_research_model or self.default_model

    @property
    def resolved_deep_research_retrieval_model(self) -> str:
        """Return the configured deep-research retrieval model."""
        return (
            self.deep_research_retrieval_model
            or self.deep_research_model
            or self.default_model
        )

    @property
    def resolved_deep_research_reflection_model(self) -> str:
        """Return the configured deep-research reflection model."""
        return (
            self.deep_research_reflection_model
            or self.deep_research_model
            or self.default_model
        )

    @property
    def resolved_deep_research_answer_model(self) -> str:
        """Return the configured deep-research answer model."""
        return self.deep_research_answer_model or self.deep_research_model or self.default_model


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return the process-wide cached application settings."""
    return AppSettings()


def resolve_model(explicit_model: str | None, configured_model: str) -> str:
    """Resolve an explicit model override against a configured default."""
    return explicit_model or configured_model
