"""Prompt builders for the pro-search answer-synthesis agent.

Purpose:
    Define the rich system prompt used by the pro-search answer agent.

Design:
    - Focuses only on synthesis from provided evidence.
    - Does not allow tool use or fresh retrieval.
    - Uses runtime context for temporal grounding and response policy.
    - Exposes a dynamic prompt middleware so runtime context is injected at
      invocation time.

Attributes:
    build_answer_agent_system_prompt:
        Build the system prompt for the answer-synthesis agent.
    answer_agent_prompt:
        Dynamic prompt middleware that injects runtime context into the prompt.

Examples:
    .. code-block:: python

        prompt = build_answer_agent_system_prompt(
            current_datetime="The current date and time is: 2026-04-22 23:15:57 EDT",
            timezone_name="America/Toronto",
            prefer_freshness=True,
            prefer_primary_sources=True,
        )
"""

from __future__ import annotations

from langchain.agents.middleware import ModelRequest, dynamic_prompt


def build_answer_agent_system_prompt(
    *,
    current_datetime: str | None,
    timezone_name: str,
    prefer_freshness: bool,
    prefer_primary_sources: bool,
) -> str:
    """Build the system prompt for the pro-search answer agent.

    Args:
        current_datetime: Human-readable current datetime string for temporal
            grounding.
        timezone_name: Canonical timezone name for the current run.
        prefer_freshness: Whether recent/current evidence should be treated as
            especially important when relevant.
        prefer_primary_sources: Whether primary or authoritative evidence should
            be weighted more heavily when available.

    Returns:
        A rich plain-text system prompt for answer synthesis.

    Raises:
        ValueError: Raised if incompatible prompt-construction inputs are passed.

    Examples:
        >>> prompt = build_answer_agent_system_prompt(
        ...     current_datetime="The current date and time is: 2026-04-22 23:15:57 EDT",
        ...     timezone_name="America/Toronto",
        ...     prefer_freshness=True,
        ...     prefer_primary_sources=True,
        ... )
        >>> "answer-synthesis agent" in prompt.lower()
        True
    """
    resolved_datetime = current_datetime or "Current datetime unavailable."

    return f"""
You are the pro-search answer-synthesis agent.

Current datetime: {resolved_datetime}
Current timezone: {timezone_name}

Your role:
- Synthesize a final markdown answer from the evidence provided to you.
- Answer the user's question directly, clearly, and with strong grounding.
- Do not call tools.
- Do not invent facts, sources, or citations.
- Do not claim you verified anything beyond the evidence you were given.

Execution constraints:
- Prefer freshness when relevant: {prefer_freshness}
- Prefer primary or authoritative sources when available: {prefer_primary_sources}

Primary goal:
Produce the best possible markdown answer from the provided evidence bundle.

Evidence-use rules:
1. Use only the evidence you are given.
2. Prefer stronger and more authoritative sources when there is a conflict.
3. Prefer more recent sources when the topic is time-sensitive.
4. If multiple sources support a key claim, cite the strongest relevant ones.
5. If the evidence is weak, conflicting, or incomplete, say so clearly.
6. Do not fabricate citations or cite sources that are not represented in the evidence.

Answer requirements:
- Answer the user's question directly.
- Use markdown.
- Be well organized but not bloated.
- Include citations for load-bearing claims.
- Preserve important nuance.
- Note uncertainty when appropriate.
- Include unresolved questions only when they materially affect confidence.

Citation requirements:
- Use the provided evidence to construct citation objects.
- Keep citations aligned with the actual sources represented in the evidence.
- Do not over-cite trivial claims.
- Always cite current, non-obvious, or load-bearing factual claims.

Quality bar:
- Direct
- Grounded
- Clear
- Honest about uncertainty
- Helpful without filler
""".strip()


@dynamic_prompt
def answer_agent_prompt(request: ModelRequest) -> str:
    """Dynamic prompt middleware for the pro-search answer agent.

    Args:
        request: Current model-call request.

    Returns:
        Fully rendered system prompt using runtime context.
    """
    context = request.runtime.context

    return build_answer_agent_system_prompt(
        current_datetime=context.current_datetime,
        timezone_name=context.timezone_name,
        prefer_freshness=context.prefer_freshness,
        prefer_primary_sources=context.prefer_primary_sources,
    )