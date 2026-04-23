from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from perplexity_at_home.agents.deep_research.context import DeepResearchContext
from perplexity_at_home.agents.deep_research.runtime import run_deep_research
from perplexity_at_home.agents.pro_search.context import ProSearchContext
from perplexity_at_home.agents.pro_search.runtime import run_pro_search
from perplexity_at_home.agents.quick_search.runtime import run_quick_search
from perplexity_at_home.core.persistence import setup_persistence
from perplexity_at_home.settings import AppSettings


def _live_e2e_ready() -> tuple[bool, str]:
    settings = AppSettings()
    if settings.env != "development" and settings.env != "test":
        return False, "Live E2E is only enabled for local or CI development-style runs."

    run_e2e = False
    try:
        import os

        run_e2e = os.getenv("PERPLEXITY_AT_HOME_RUN_E2E", "").strip().lower() == "true"
    except Exception:
        run_e2e = False

    if not run_e2e:
        return False, "Set PERPLEXITY_AT_HOME_RUN_E2E=true to enable live external-service tests."

    missing: list[str] = []
    if settings.openai_api_key is None:
        missing.append("OPENAI_API_KEY")
    if settings.tavily_api_key is None:
        missing.append("TAVILY_API_KEY")

    if missing:
        return False, f"Missing required live-test credentials: {', '.join(missing)}."

    return True, ""


_LIVE_E2E_ENABLED, _LIVE_E2E_REASON = _live_e2e_ready()
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(300),
    pytest.mark.skipif(not _LIVE_E2E_ENABLED, reason=_LIVE_E2E_REASON),
]

QUICK_QUESTION = "What changed recently in Tavily's LangChain integration? Use current sources."
RESEARCH_QUESTION = "What is Tavily?"


def _random_thread_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


def _quick_payload(result: dict[str, object]) -> dict[str, object]:
    structured = result.get("structured_response", {})
    model_dump = getattr(structured, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(mode="json")
        if isinstance(payload, dict):
            return payload
    return structured if isinstance(structured, dict) else {}


def _assert_markdown_contains_tavily(markdown: str) -> None:
    assert markdown.strip()
    assert "tavily" in markdown.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True], ids=["in-memory", "persistent"])
async def test_live_quick_search(persistent: bool) -> None:
    if persistent:
        await setup_persistence()

    result = await run_quick_search(
        QUICK_QUESTION,
        thread_id=_random_thread_id("quick-e2e"),
        persistent=persistent,
        setup_persistence=persistent,
    )

    payload = _quick_payload(result)
    citations = payload.get("citations", [])
    answer_markdown = str(payload.get("answer_markdown") or "")

    _assert_markdown_contains_tavily(answer_markdown)
    assert isinstance(citations, list)
    assert citations


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True], ids=["in-memory", "persistent"])
async def test_live_pro_search(persistent: bool) -> None:
    if persistent:
        await setup_persistence()

    context = ProSearchContext(thread_id=_random_thread_id("pro-e2e"))
    result = await run_pro_search(
        RESEARCH_QUESTION,
        persistent=persistent,
        setup_persistence=persistent,
        context=context,
    )

    final_answer = result.get("final_answer", {})
    answer_markdown = str(final_answer.get("answer_markdown") or "")
    citations = final_answer.get("citations", [])

    _assert_markdown_contains_tavily(answer_markdown)
    assert isinstance(citations, list)
    assert citations
    assert int(final_answer.get("evidence_count", 0) or 0) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True], ids=["in-memory", "persistent"])
async def test_live_deep_research(persistent: bool) -> None:
    if persistent:
        await setup_persistence()

    context = replace(
        DeepResearchContext(thread_id=_random_thread_id("deep-e2e")),
        max_subquestions=3,
        max_iterations=2,
        max_parallel_retrieval_branches=2,
    )
    result = await run_deep_research(
        RESEARCH_QUESTION,
        persistent=persistent,
        setup_persistence=persistent,
        context=context,
    )

    final_answer = result.get("final_answer", {})
    report_markdown = str(final_answer.get("report_markdown") or "")
    citations = final_answer.get("citations", [])

    _assert_markdown_contains_tavily(report_markdown)
    assert isinstance(citations, list)
    assert citations
    assert int(final_answer.get("evidence_count", 0) or 0) > 0
