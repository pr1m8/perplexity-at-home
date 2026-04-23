"""Normalization helpers for Tavily payloads.

Purpose:
    Normalize Tavily responses into stable internal dictionaries or lightweight
    models that higher layers can consume consistently.

Design:
    - Keep normalization separate from tool construction.
    - Be tolerant of partial Tavily payloads.
    - Avoid embedding agent-specific semantics here.

Examples:
    >>> payload = {"results": [{"url": "https://example.com", "title": "Example", "content": "Snippet"}]}
    >>> normalize_search_payload(payload)[0]["url"]
    'https://example.com'
"""

from __future__ import annotations

from typing import Any


def normalize_search_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single Tavily search hit.

    Args:
        hit: Raw Tavily result hit.

    Returns:
        A normalized hit dictionary.
    """
    return {
        "url": hit.get("url"),
        "title": hit.get("title", ""),
        "content": hit.get("content", ""),
        "score": hit.get("score"),
        "raw_content": hit.get("raw_content"),
    }


def normalize_search_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize a Tavily search payload.

    Args:
        payload: Raw Tavily search response.

    Returns:
        A list of normalized hit dictionaries.
    """
    raw_results = payload.get("results", [])
    return [normalize_search_hit(hit) for hit in raw_results]


def extract_answer(payload: dict[str, Any]) -> str | None:
    """Extract the top-level answer field from a Tavily payload.

    Args:
        payload: Raw Tavily response.

    Returns:
        The answer string if present, otherwise ``None``.
    """
    answer = payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer
    return None