"""Dashboard package for the Streamlit UI surface."""

from perplexity_at_home.dashboard.models import (
    DashboardCitation,
    DashboardRunRequest,
    DashboardRunResult,
    DashboardThreadRecord,
    DashboardTurnRecord,
    SearchWorkflow,
)
from perplexity_at_home.dashboard.presentation import build_mermaid_embed, format_thread_label
from perplexity_at_home.dashboard.service import DashboardService

__all__ = [
    "DashboardCitation",
    "DashboardRunRequest",
    "DashboardRunResult",
    "DashboardService",
    "DashboardThreadRecord",
    "DashboardTurnRecord",
    "SearchWorkflow",
    "build_mermaid_embed",
    "format_thread_label",
]
