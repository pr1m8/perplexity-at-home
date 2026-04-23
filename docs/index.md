# perplexity-at-home

`perplexity-at-home` packages a set of LangGraph research workflows behind a
clean Python package and CLI. The repo currently includes a durable
deep-research path, a tighter pro-search loop, and a lightweight quick-search
builder for simpler question-answer flows.

## What the package gives you

- Pydantic Settings for OpenAI, Tavily, LangSmith, and Postgres configuration
- a packaged CLI entrypoint: `perplexity-at-home`
- a packaged Streamlit launcher: `perplexity-at-home-dashboard`
- optional Postgres-backed LangGraph persistence for checkpoints and store state
- packaged builders for quick-search, pro-search, and deep-research graphs
- a dashboard surface for switching between all three workflows
- runnable demos under `examples/`
- unit and integration coverage around settings, graph construction, persistence,
  the CLI surface, plus gated live E2E coverage for real external services

## Core commands

```bash
pdm run perplexity-at-home --help
pdm run perplexity-at-home deep-research "What is Tavily?"
make infra-up
make infra-setup
pdm run perplexity-at-home deep-research --persistent "What is Tavily?"
make test-e2e
```

## Documentation map

- [Getting Started](getting-started.md): local setup, environment, and run flow
- [Architecture](architecture.md): graph layout, persistence, and package boundaries
- [Reference](reference/cli.md): API and runtime entrypoints pulled from the package
