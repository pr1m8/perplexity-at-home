# Deep Research Notes

## Current direction

- Use a parent `StateGraph` for the full research workflow instead of exposing only child agents.
- Keep child agents specialized:
  - planner
  - query generator
  - retrieval
  - reflection
  - answer synthesis
- Route with conditional edges after planning and reflection.

## What changed

- Added `pydantic-settings` backed model configuration in `src/perplexity_at_home/settings.py`.
- Default model is now `openai:gpt-5.4`, with workflow-specific overrides available via env vars.
- Added Postgres-backed settings exposure as `settings.postgres.uri`, with compatibility for both legacy `POSTGRES_*` vars and nested `PERPLEXITY_AT_HOME_POSTGRES__*` overrides.
- Added optional LangSmith / LangChain tracing settings to the config surface.
- Added a top-level deep-research graph and wrapper agent.
- Added compile-time hooks so your new async checkpointer/store can be passed into the graph cleanly.
- Added `langgraph.json` so the repository can be loaded as a LangGraph application.
- Added offline tests for end-to-end routing and clarification exits.

## Verification

- `pdm run ruff check src tests` passes.
- `pdm run pytest` passes.
- Coverage gate passes at 85.26%.
- Reports are written to `reports/junit.xml`, `reports/coverage.xml`, and `reports/htmlcov/`.

## Pending follow-ups

- Wire infra-managed services once Postgres, Docker Compose, and Make targets land.
- Add real end-to-end tests around live Tavily/OpenAI credentials after infra is in place.
- Add runtime wiring that opens your `core.checkpoint` and `core.store` contexts around graph compilation where the app lifespan is managed.
- Consider parallel branch fan-out per subquestion if the current single retrieval node becomes a bottleneck.
