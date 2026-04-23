from __future__ import annotations

import asyncio

from perplexity_at_home.dashboard.models import DashboardRunRequest, SearchWorkflow
import perplexity_at_home.dashboard.service as dashboard_service
from perplexity_at_home.settings import AppSettings


def _settings() -> AppSettings:
    return AppSettings(
        _env_file=None,
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key",
        langsmith_api_key="test-langsmith-key",
    )


def test_dashboard_service_normalizes_quick_search(monkeypatch) -> None:
    async def fake_run_quick_search(*args, **kwargs):
        return {
            "structured_response": {
                "answer_markdown": "Quick answer",
                "confidence": 0.9,
                "used_search": True,
                "citations": [
                    {
                        "title": "Quick source",
                        "url": "https://example.com/quick",
                        "supports": "Supports the quick answer.",
                    }
                ],
            }
        }

    monkeypatch.setattr(dashboard_service, "run_quick_search", fake_run_quick_search)

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.QUICK,
                question="What is Tavily?",
                thread_id="quick-thread",
            )
        )
    )

    assert result.answer_markdown == "Quick answer"
    assert result.thread_id == "quick-thread"
    assert result.metadata["used_search"] is True
    assert result.citations[0].title == "Quick source"


def test_dashboard_service_normalizes_pro_search(monkeypatch) -> None:
    async def fake_run_pro_search(*args, **kwargs):
        return {
            "final_answer": {
                "answer_markdown": "Pro answer",
                "confidence": 0.75,
                "evidence_count": 4,
                "citations": [
                    {
                        "title": "Pro source",
                        "url": "https://example.com/pro",
                    }
                ],
                "unresolved_questions": ["Need another source"],
            }
        }

    monkeypatch.setattr(dashboard_service, "run_pro_search", fake_run_pro_search)

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.PRO,
                question="What changed?",
                thread_id="pro-thread",
                persistent=True,
            )
        )
    )

    assert result.answer_markdown == "Pro answer"
    assert result.persistent is True
    assert result.metadata["evidence_count"] == 4
    assert result.metadata["unresolved_questions"] == ["Need another source"]


def test_dashboard_service_normalizes_deep_research(monkeypatch) -> None:
    async def fake_run_deep_research(*args, **kwargs):
        return {
            "final_answer": {
                "report_markdown": "# Report",
                "executive_summary": "Deep summary",
                "confidence": 0.82,
                "evidence_count": 6,
                "key_findings": ["Finding A"],
                "citations": [
                    {
                        "title": "Deep source",
                        "url": "https://example.com/deep",
                        "supports": "Supports the report.",
                    }
                ],
            }
        }

    monkeypatch.setattr(dashboard_service, "run_deep_research", fake_run_deep_research)

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.DEEP,
                question="Research this",
                thread_id="deep-thread",
                persistent=True,
            )
        )
    )

    assert result.answer_markdown == "# Report"
    assert result.summary == "Deep summary"
    assert result.metadata["key_findings"] == ["Finding A"]
    assert result.citations[0].url == "https://example.com/deep"


def test_dashboard_service_normalizes_model_backed_payloads(monkeypatch) -> None:
    class FakePayload:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {
                "answer_markdown": "Quick model answer",
                "confidence": 0.5,
                "used_search": False,
                "citations": [
                    {
                        "title": "Valid source",
                        "url": "https://example.com/valid",
                    },
                    {
                        "title": "",
                        "url": "",
                    },
                ],
            }

    class FakeStateObject:
        def model_dump(self, *, mode: str) -> dict[str, object]:
            assert mode == "json"
            return {"serialized": True}

    async def fake_run_quick_search(*args, **kwargs):
        return {
            "structured_response": FakePayload(),
            "raw_object": FakeStateObject(),
            "fallback_repr": object(),
        }

    monkeypatch.setattr(dashboard_service, "run_quick_search", fake_run_quick_search)

    service = dashboard_service.DashboardService(settings=_settings())
    result = asyncio.run(
        service.run(
            DashboardRunRequest(
                workflow=SearchWorkflow.QUICK,
                question="Normalize this",
                thread_id="quick-model-thread",
            )
        )
    )

    assert result.answer_markdown == "Quick model answer"
    assert len(result.citations) == 1
    assert result.raw_state["raw_object"] == {"serialized": True}
    assert isinstance(result.raw_state["fallback_repr"], str)
