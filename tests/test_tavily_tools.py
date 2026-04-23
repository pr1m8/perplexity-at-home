from __future__ import annotations

from typing import Any

from perplexity_at_home.settings import AppSettings
from perplexity_at_home.tools.tavily import bundles, factories, normalize


def test_normalize_search_payload_and_answer_extraction() -> None:
    payload = {
        "answer": "A concise answer.",
        "results": [
            {
                "url": "https://example.com",
                "title": "Example",
                "content": "Snippet",
                "score": 0.9,
                "raw_content": "Raw",
            }
        ],
    }

    normalized_hits = normalize.normalize_search_payload(payload)

    assert normalized_hits == [
        {
            "url": "https://example.com",
            "title": "Example",
            "content": "Snippet",
            "score": 0.9,
            "raw_content": "Raw",
        }
    ]
    assert normalize.extract_answer(payload) == "A concise answer."
    assert normalize.extract_answer({"answer": "   "}) is None


def test_factories_build_tools_from_presets(monkeypatch) -> None:
    captured: dict[str, Any] = {}
    settings = AppSettings(_env_file=None, tavily_api_key="test-tavily-key")

    class DummySearchWrapper:
        def __init__(self, *, tavily_api_key) -> None:
            captured["search_wrapper"] = tavily_api_key

    class DummyExtractWrapper:
        def __init__(self, *, tavily_api_key) -> None:
            captured["extract_wrapper"] = tavily_api_key

    class DummyMapWrapper:
        def __init__(self, *, tavily_api_key) -> None:
            captured["map_wrapper"] = tavily_api_key

    class DummyCrawlWrapper:
        def __init__(self, *, tavily_api_key) -> None:
            captured["crawl_wrapper"] = tavily_api_key

    class DummyResearchWrapper:
        def __init__(self, *, tavily_api_key) -> None:
            captured["research_wrapper"] = tavily_api_key

    class DummySearch:
        def __init__(self, **kwargs: Any) -> None:
            captured["search"] = kwargs

    class DummyExtract:
        def __init__(self, **kwargs: Any) -> None:
            captured["extract"] = kwargs

    class DummyResearch:
        def __init__(self, **kwargs: Any) -> None:
            captured["research"] = kwargs

    class DummyMap:
        def __init__(self, **kwargs: Any) -> None:
            captured["map"] = kwargs

    class DummyCrawl:
        def __init__(self, **kwargs: Any) -> None:
            captured["crawl"] = kwargs

    class DummyGetResearch:
        def __init__(self, **kwargs: Any) -> None:
            captured["get_research"] = kwargs

    monkeypatch.setattr(factories, "TavilySearch", DummySearch)
    monkeypatch.setattr(factories, "TavilyExtract", DummyExtract)
    monkeypatch.setattr(factories, "TavilyResearch", DummyResearch)
    monkeypatch.setattr(factories, "TavilyMap", DummyMap)
    monkeypatch.setattr(factories, "TavilyCrawl", DummyCrawl)
    monkeypatch.setattr(factories, "TavilyGetResearch", DummyGetResearch)
    monkeypatch.setattr(factories, "TavilySearchAPIWrapper", DummySearchWrapper)
    monkeypatch.setattr(factories, "TavilyExtractAPIWrapper", DummyExtractWrapper)
    monkeypatch.setattr(factories, "TavilyMapAPIWrapper", DummyMapWrapper)
    monkeypatch.setattr(factories, "TavilyCrawlAPIWrapper", DummyCrawlWrapper)
    monkeypatch.setattr(factories, "TavilyResearchAPIWrapper", DummyResearchWrapper)
    monkeypatch.setattr(factories, "get_settings", lambda: settings)

    factories.build_search_tool()
    factories.build_pro_search_tool()
    factories.build_extract_tool()
    factories.build_pro_extract_tool()
    factories.build_map_tool()
    factories.build_crawl_tool()
    factories.build_research_tool()
    factories.build_get_research_tool()

    assert captured["search"]["max_results"] == factories.PRO_SEARCH_PRESET.max_results
    assert captured["search"]["api_wrapper"].__class__ is DummySearchWrapper
    assert captured["extract"]["extract_depth"] == factories.PRO_EXTRACT_PRESET.extract_depth
    assert captured["extract"]["apiwrapper"].__class__ is DummyExtractWrapper
    assert captured["research"]["model"] == factories.DEEP_RESEARCH_PRESET.model
    assert captured["research"]["api_wrapper"].__class__ is DummyResearchWrapper
    assert captured["map"]["api_wrapper"].__class__ is DummyMapWrapper
    assert captured["crawl"]["api_wrapper"].__class__ is DummyCrawlWrapper
    assert captured["get_research"]["api_wrapper"].__class__ is DummyResearchWrapper
    assert captured["search_wrapper"] == settings.require_tavily_api_key()
    assert captured["extract_wrapper"] == settings.require_tavily_api_key()
    assert captured["map_wrapper"] == settings.require_tavily_api_key()
    assert captured["crawl_wrapper"] == settings.require_tavily_api_key()
    assert captured["research_wrapper"] == settings.require_tavily_api_key()


def test_bundles_group_expected_tools(monkeypatch) -> None:
    monkeypatch.setattr(bundles, "build_search_tool", lambda: "quick-search")
    monkeypatch.setattr(bundles, "build_extract_tool", lambda: "quick-extract")
    monkeypatch.setattr(bundles, "build_pro_search_tool", lambda: "pro-search")
    monkeypatch.setattr(bundles, "build_pro_extract_tool", lambda: "pro-extract")
    monkeypatch.setattr(bundles, "build_map_tool", lambda: "map")
    monkeypatch.setattr(bundles, "build_crawl_tool", lambda: "crawl")
    monkeypatch.setattr(bundles, "build_research_tool", lambda: "research")
    monkeypatch.setattr(bundles, "build_get_research_tool", lambda: "get_research")

    assert bundles.build_quick_bundle() == {
        "search": "quick-search",
        "extract": "quick-extract",
    }
    assert bundles.build_pro_bundle() == {
        "search": "pro-search",
        "extract": "pro-extract",
        "map": "map",
        "crawl": "crawl",
    }
    assert bundles.build_deep_bundle() == {
        "search": "pro-search",
        "extract": "pro-extract",
        "map": "map",
        "crawl": "crawl",
        "research": "research",
        "get_research": "get_research",
    }
