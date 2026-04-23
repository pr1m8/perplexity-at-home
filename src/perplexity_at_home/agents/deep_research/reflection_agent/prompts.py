"""Prompt builders for the deep-research reflection agent.

Purpose:
    Define the rich system prompt used by the deep-research reflection agent.

Design:
    - Focuses only on reflection over already-collected evidence.
    - Produces an action-oriented decision for the parent graph.
    - Uses runtime context for temporal grounding and workflow budgets.
    - Exposes a dynamic prompt middleware so runtime context is injected at
      invocation time.

Attributes:
    build_reflection_system_prompt:
        Build the system prompt for the deep-research reflection agent.
    reflection_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_reflection_system_prompt(
            current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
            timezone_name="America/Toronto",
            max_iterations=3,
            allow_map=True,
            allow_crawl=True,
            allow_tavily_research=True,
            prefer_freshness=True,
            prefer_primary_sources=True,
        )
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_reflection_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    max_iterations: int,
    allow_map: bool,
    allow_crawl: bool,
    allow_tavily_research: bool,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
) -> str:
    """Build the system prompt for the deep-research reflection agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal grounding.
        timezone_name: Canonical timezone name for the current run.
        max_iterations: Maximum number of reflection/requery iterations allowed.
        allow_map: Whether Tavily map is available to downstream retrieval.
        allow_crawl: Whether Tavily crawl is available to downstream retrieval.
        allow_tavily_research: Whether Tavily research/get-research is available
            to downstream retrieval.
        prefer_freshness: Whether the workflow should favor recent/current
            information when relevant.
        prefer_primary_sources: Whether the workflow should prefer primary or
            authoritative sources when available.

    Returns:
        A rich plain-text system prompt for the reflection agent.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.

    Examples:
        >>> prompt = build_reflection_system_prompt(
        ...     current_datetime="The current date and time is: 2026-04-23 00:15:57 EDT",
        ...     timezone_name="America/Toronto",
        ...     max_iterations=3,
        ...     allow_map=True,
        ...     allow_crawl=True,
        ...     allow_tavily_research=True,
        ...     prefer_freshness=True,
        ...     prefer_primary_sources=True,
        ... )
        >>> "reflection agent" in prompt.lower()
        True
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the deep-research reflection agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Evaluate the current evidence bundle and decide what the workflow should do next.
- Determine whether the evidence is sufficient for final synthesis.
- If not sufficient, identify the most important gaps or conflicts.
- Recommend the best next action for the parent graph.
- Do not call tools.
- Do not write the final answer.

Workflow constraints:
- Maximum total reflection/requery iterations allowed: {max_iterations}
- Map available downstream: {allow_map}
- Crawl available downstream: {allow_crawl}
- Tavily research available downstream: {allow_tavily_research}
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary sources when available: {prefer_primary_sources}

Primary objective:
Produce the best possible structured reflection decision so the parent
deep-research graph can either:
- synthesize,
- requery,
- deepen retrieval with extract/map/crawl/research,
- or clarify with the user.

Reflection responsibilities:
1. Judge whether the available evidence is sufficient to answer the original question well.
2. Identify high-value missing facts or unresolved facets.
3. Identify meaningful conflicts or contradictions in the evidence.
4. Recommend the best next action for the graph.
5. Suggest follow-up queries only when they would materially improve the workflow.
6. Prefer focused next steps over vague “more research” recommendations.

Decision rules:
- Recommend ``synthesize`` only when the evidence is strong enough to answer
  the question honestly and usefully.
- Recommend ``requery`` when the evidence is incomplete but targeted additional
  searching is likely to fix the problem.
- Recommend ``extract`` when the graph already has strong candidate URLs but
  needs deeper page content.
- Recommend ``map`` when the target is a structured site and internal discovery
  is needed.
- Recommend ``crawl`` when a single domain is clearly central and multiple
  pages likely matter.
- Recommend ``research`` only when a broader asynchronous research worker is
  likely to add value.
- Recommend ``clarify`` only when the remaining ambiguity is fundamentally about
  user intent rather than missing evidence.

Evidence-quality rules:
- Prefer stronger and more authoritative evidence when there is a conflict.
- Prefer more recent evidence when the question is freshness-sensitive.
- Do not treat weak, tangential, or repetitive evidence as sufficient.
- Be honest when the evidence is thin, fragmented, or contradictory.

Follow-up query rules:
- Suggest follow-up queries only when they are likely to materially improve the result.
- Keep them concise and executable.
- Prioritize coverage of the most important gaps first.
- Avoid redundant query suggestions.

Output rules:
- Return structured output only.
- Always provide a rationale.
- Use open gaps only when real gaps remain.
- Use conflicting claims only when the conflict is meaningful.
- Use notes only when they help the parent graph route or debug the next step.

Quality bar:
- Honest
- Actionable
- Specific
- Graph-friendly
- Conservative about sufficiency
""".strip()


@dynamic_prompt
def reflection_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the deep-research reflection agent.

    Args:
        request: Current model-call request.

    Returns:
        Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_reflection_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        max_iterations=context.max_iterations,
        allow_map=context.allow_map,
        allow_crawl=context.allow_crawl,
        allow_tavily_research=context.allow_tavily_research,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
    )