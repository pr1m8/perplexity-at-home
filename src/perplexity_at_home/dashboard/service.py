"""Dashboard service that normalizes quick, pro, and deep workflow runs."""

from __future__ import annotations

from typing import Any

from perplexity_at_home.agents.deep_research import DeepResearchContext, run_deep_research
from perplexity_at_home.agents.pro_search import ProSearchContext, run_pro_search
from perplexity_at_home.agents.quick_search import QuickSearchContext, run_quick_search
from perplexity_at_home.dashboard.models import (
    DashboardCitation,
    DashboardRunRequest,
    DashboardRunResult,
    SearchWorkflow,
)
from perplexity_at_home.settings import AppSettings, get_settings
from perplexity_at_home.utils import get_current_datetime_string

__all__ = ["DashboardService"]


def _to_jsonable(value: Any) -> Any:
    """Convert nested workflow state into JSON-friendly primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}

    if isinstance(value, list | tuple | set):
        return [_to_jsonable(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _to_jsonable(model_dump(mode="json"))

    return repr(value)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Convert a model-like object into a JSON-friendly dictionary."""
    if isinstance(value, dict):
        return value

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, dict):
            return dumped

    return {}


def _normalize_citations(raw_citations: list[Any]) -> list[DashboardCitation]:
    """Normalize heterogeneous citation shapes into dashboard citations."""
    citations: list[DashboardCitation] = []
    for citation in raw_citations:
        payload = _coerce_mapping(citation)
        title = str(payload.get("title") or "").strip()
        url = str(payload.get("url") or "").strip()
        if not title or not url:
            continue
        supports = payload.get("supports")
        citations.append(
            DashboardCitation(
                title=title,
                url=url,
                supports=str(supports).strip() if isinstance(supports, str) and supports.strip() else None,
            )
        )
    return citations


class DashboardService:
    """Run the repository workflows behind a dashboard-friendly interface."""

    def __init__(self, *, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def default_model_for_workflow(self, workflow: SearchWorkflow) -> str:
        """Return the configured default model for a workflow."""
        if workflow is SearchWorkflow.QUICK:
            return self.settings.resolved_quick_search_model
        if workflow is SearchWorkflow.PRO:
            return self.settings.resolved_pro_search_answer_model
        return self.settings.resolved_deep_research_answer_model

    async def run(self, request: DashboardRunRequest) -> DashboardRunResult:
        """Run one dashboard request and normalize the response."""
        if request.workflow is SearchWorkflow.QUICK:
            context = QuickSearchContext(
                current_datetime=get_current_datetime_string(),
                timezone_name=request.timezone_name,
            )
            raw_state = await run_quick_search(
                request.question,
                thread_id=request.thread_id,
                persistent=request.persistent,
                setup_persistence=request.setup_persistence,
                context=context,
                debug=request.debug,
            )
            return self._normalize_quick_result(request, raw_state)

        if request.workflow is SearchWorkflow.PRO:
            context = ProSearchContext(
                current_datetime=get_current_datetime_string(),
                timezone_name=request.timezone_name,
                thread_id=request.thread_id,
            )
            raw_state = await run_pro_search(
                request.question,
                persistent=request.persistent,
                setup_persistence=request.setup_persistence,
                context=context,
                debug=request.debug,
            )
            return self._normalize_pro_result(request, raw_state)

        context = DeepResearchContext(
            current_datetime=get_current_datetime_string(),
            timezone_name=request.timezone_name,
            thread_id=request.thread_id,
        )
        raw_state = await run_deep_research(
            request.question,
            persistent=request.persistent,
            setup_persistence=request.setup_persistence,
            context=context,
            debug=request.debug,
        )
        return self._normalize_deep_result(request, raw_state)

    def _normalize_quick_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a quick-search result."""
        structured = _coerce_mapping(raw_state.get("structured_response"))
        answer_markdown = str(structured.get("answer_markdown") or "").strip()
        summary = structured.get("uncertainty_note")
        return DashboardRunResult(
            workflow=request.workflow,
            question=request.question,
            thread_id=request.thread_id,
            persistent=request.persistent,
            answer_markdown=answer_markdown or "_No answer markdown returned._",
            summary=str(summary) if isinstance(summary, str) and summary.strip() else None,
            confidence=structured.get("confidence"),
            citations=_normalize_citations(list(structured.get("citations") or [])),
            metadata={
                "used_search": structured.get("used_search"),
                "model": self.default_model_for_workflow(request.workflow),
            },
            raw_state=_to_jsonable(raw_state),
        )

    def _normalize_pro_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a pro-search result."""
        final_answer = _coerce_mapping(raw_state.get("final_answer"))
        answer_markdown = str(final_answer.get("answer_markdown") or "").strip()
        summary = final_answer.get("uncertainty_note")
        return DashboardRunResult(
            workflow=request.workflow,
            question=request.question,
            thread_id=request.thread_id,
            persistent=request.persistent,
            answer_markdown=answer_markdown or "_No answer markdown returned._",
            summary=str(summary) if isinstance(summary, str) and summary.strip() else None,
            confidence=final_answer.get("confidence"),
            citations=_normalize_citations(list(final_answer.get("citations") or [])),
            metadata={
                "evidence_count": final_answer.get("evidence_count"),
                "unresolved_questions": final_answer.get("unresolved_questions", []),
                "model": self.default_model_for_workflow(request.workflow),
            },
            raw_state=_to_jsonable(raw_state),
        )

    def _normalize_deep_result(
        self,
        request: DashboardRunRequest,
        raw_state: dict[str, Any],
    ) -> DashboardRunResult:
        """Normalize a deep-research result."""
        final_answer = _coerce_mapping(raw_state.get("final_answer"))
        answer_markdown = str(final_answer.get("report_markdown") or "").strip()
        summary = final_answer.get("executive_summary") or final_answer.get("uncertainty_note")
        return DashboardRunResult(
            workflow=request.workflow,
            question=request.question,
            thread_id=request.thread_id,
            persistent=request.persistent,
            answer_markdown=answer_markdown or "_No report markdown returned._",
            summary=str(summary) if isinstance(summary, str) and summary.strip() else None,
            confidence=final_answer.get("confidence"),
            citations=_normalize_citations(list(final_answer.get("citations") or [])),
            metadata={
                "evidence_count": final_answer.get("evidence_count"),
                "key_findings": final_answer.get("key_findings", []),
                "unresolved_questions": final_answer.get("unresolved_questions", []),
                "model": self.default_model_for_workflow(request.workflow),
            },
            raw_state=_to_jsonable(raw_state),
        )
