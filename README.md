# perplexity-at-home

LangGraph-based research agents for local and Postgres-backed runs. The package
currently ships three main workflows:

- `deep-research` for multi-step planning, retrieval, reflection, and synthesis
- `pro-search` for a tighter search-answer loop
- `quick-search` for a focused single-pass answer path

The runtime is configured with Pydantic Settings, uses Tavily for retrieval,
supports GPT-5.4 as the default OpenAI model, and can persist LangGraph state
through Postgres for resumable sessions.

## Quickstart

```bash
pdm install -G test -G docs
cp .env.example .env
pdm run perplexity-at-home deep-research "What changed recently in Tavily's LangChain integration?"
```

To enable durable persistence locally:

```bash
make infra-up
make infra-setup
pdm run perplexity-at-home deep-research --persistent "What is Tavily?"
```

## Package Surface

- `src/perplexity_at_home/settings.py`: environment-backed settings and model selection
- `src/perplexity_at_home/core/`: Postgres store and checkpointer helpers
- `src/perplexity_at_home/agents/deep_research/`: top-level deep-research graph and child agents
- `src/perplexity_at_home/agents/pro_search/`: compact research workflow
- `src/perplexity_at_home/agents/quick_search/`: lightweight question-answer path
- `examples/`: runnable demos for the agent components

## Development

```bash
make lint
make test
make docs-build
pdm build
```

Documentation is being formalized under `docs/` with MkDocs Material and Read
the Docs configuration at the repository root.
