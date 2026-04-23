"""Prompt builders for the deep-research retrieval agent.

Purpose:
    Define the rich system prompt used by the deep-research retrieval agent.

Design:
    - Focuses only on dynamic retrieval, not final answer writing.
    - Exposes the full Tavily V2 tool surface.
    - Uses runtime context for temporal grounding, retrieval budgets, and
      downstream workflow constraints.
    - Exposes a dynamic prompt middleware so runtime context is injected at
      invocation time.
    - Makes strategy-following explicit so the parent graph can trust the result.

Attributes:
    build_retrieval_agent_system_prompt:
        Build the system prompt for the deep-research retrieval agent.
    retrieval_agent_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_retrieval_agent_system_prompt(
            current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
            timezone_name="America/Toronto",
            max_results_per_query=6,
            max_extract_urls_per_pass=5,
            allow_map=True,
            allow_crawl=True,
            allow_tavily_research=True,
            prefer_freshness=True,
            prefer_primary_sources=True,
        )
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_retrieval_agent_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    max_results_per_query: int,
    max_extract_urls_per_pass: int,
    allow_map: bool,
    allow_crawl: bool,
    allow_tavily_research: bool,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
) -> str:
    """Build the system prompt for the deep-research retrieval agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal grounding.
        timezone_name: Canonical timezone name for the current run.
        max_results_per_query: Maximum number of retained results per query.
        max_extract_urls_per_pass: Maximum number of URLs allowed in one extract pass.
        allow_map: Whether Tavily map is available.
        allow_crawl: Whether Tavily crawl is available.
        allow_tavily_research: Whether Tavily research/get-research is available.
        prefer_freshness: Whether the workflow should favor recent/current information.
        prefer_primary_sources: Whether the workflow should prefer primary or
            authoritative sources when available.

    Returns:
        A rich plain-text system prompt for the retrieval agent.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.

    Examples:
        >>> prompt = build_retrieval_agent_system_prompt(
        ...     current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
        ...     timezone_name="America/Toronto",
        ...     max_results_per_query=6,
        ...     max_extract_urls_per_pass=5,
        ...     allow_map=True,
        ...     allow_crawl=True,
        ...     allow_tavily_research=True,
        ...     prefer_freshness=True,
        ...     prefer_primary_sources=True,
        ... )
        >>> "retrieval agent" in prompt.lower()
        True
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the deep-research retrieval agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Execute the current retrieval objective using the available Tavily tools.
- Gather strong evidence for the provided question, brief, subquestions, and query plans.
- Do not write the final answer.
- Do not fabricate evidence, citations, tool usage, dates, or URLs.
- Return a structured evidence bundle for reflection and later synthesis.

Available Tavily tool surface in this V2 retrieval environment:
- search
- extract
- map
- crawl
- research
- get_research

Retrieval constraints:
- Maximum retained results per query: {max_results_per_query}
- Maximum extract URLs per pass: {max_extract_urls_per_pass}
- Map available: {allow_map}
- Crawl available: {allow_crawl}
- Tavily research available: {allow_tavily_research}
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary or authoritative sources when available: {prefer_primary_sources}

Primary objective:
Use the available Tavily tools to gather the best possible evidence bundle for
the current deep-research step.

Core behavior:
1. Read the current payload carefully.
2. Use the provided query plans and retrieval recommendations as guidance.
3. Prefer the recommended strategy unless there is a strong reason not to.
4. If you deviate from the recommended strategy, explain why clearly.
5. Use as few tool calls as necessary, but enough to gather strong evidence.
6. Prefer focused retrieval over noisy broad retrieval.

Strategy rules:
- Use ``search`` for first-pass discovery or when the task starts from a question.
- Use ``extract`` when strong candidate URLs are already known and deeper page content is needed.
- Use ``map`` when site structure matters before extraction and the task is centered on a documentation or official site.
- Use ``crawl`` when one domain is central and multiple pages likely matter.
- Use ``research`` and ``get_research`` only when a broader asynchronous research worker is clearly appropriate.
- Do not default to search-plus-extract automatically if the recommended strategy is map, crawl, or research.

Evidence rules:
- Prefer official, primary, or authoritative sources when possible.
- Prefer more recent evidence when the question or subquestion is freshness-sensitive.
- Retain evidence that is likely useful for reflection or final synthesis.
- Do not overstate what the evidence proves.
- If evidence is weak or incomplete, say so in unresolved gaps.

Reporting rules:
- Normalize tool names to the canonical labels: search, extract, map, crawl, research, get_research.
- Record the recommended strategy from the input.
- Record the strategy you actually applied.
- Record whether you followed the recommendation.
- Keep the retrieval summary operational rather than answer-like.
- Recommend the best next action for the parent graph.

Output rules:
- Return structured output only.
- Include structured evidence items.
- Include structured tool-usage records.
- Include unresolved gaps when material evidence is still missing.
- Be conservative with confidence.

Quality bar:
- Efficient
- Evidence-focused
- Honest about gaps
- Strategy-aware
- Conservative about overclaiming
""".strip()


@dynamic_prompt
def retrieval_agent_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the deep-research retrieval agent.

    Args:
        request: Current model-call request.

    Returns:
        Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_retrieval_agent_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        max_results_per_query=context.max_results_per_query,
        max_extract_urls_per_pass=context.max_extract_urls_per_pass,
        allow_map=context.allow_map,
        allow_crawl=context.allow_crawl,
        allow_tavily_research=context.allow_tavily_research,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
    )