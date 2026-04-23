from __future__ import annotations

import pytest


@pytest.fixture
def user_queries() -> list[str]:
    return [
        "What changed recently in Tavily's LangChain integration?",
        "How does the current migration guidance differ from older usage?",
    ]
