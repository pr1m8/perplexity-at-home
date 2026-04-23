"""Dashboard models shared by the Streamlit app and service layer."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class SearchWorkflow(StrEnum):
    """Supported research workflows exposed in the dashboard."""

    QUICK = "quick-search"
    PRO = "pro-search"
    DEEP = "deep-research"

    @property
    def label(self) -> str:
        """Return a human-readable workflow label."""
        return {
            SearchWorkflow.QUICK: "Quick Search",
            SearchWorkflow.PRO: "Pro Search",
            SearchWorkflow.DEEP: "Deep Research",
        }[self]


class DashboardCitation(BaseModel):
    """Normalized citation rendered by the dashboard."""

    title: str
    url: str
    supports: str | None = None


class DashboardRunRequest(BaseModel):
    """Input payload for one dashboard workflow run."""

    workflow: SearchWorkflow
    question: str = Field(min_length=1)
    thread_id: str = Field(default_factory=lambda: uuid4().hex)
    persistent: bool = False
    setup_persistence: bool = False
    debug: bool = False
    timezone_name: str = "America/Toronto"

    @field_validator("question")
    @classmethod
    def _validate_question(cls, value: str) -> str:
        """Reject blank questions."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("The dashboard question must not be empty.")
        return stripped


class DashboardRunResult(BaseModel):
    """Normalized dashboard result for any workflow."""

    workflow: SearchWorkflow
    question: str
    thread_id: str
    persistent: bool
    answer_markdown: str
    summary: str | None = None
    confidence: float | None = None
    citations: list[DashboardCitation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_state: dict[str, Any] = Field(default_factory=dict)
