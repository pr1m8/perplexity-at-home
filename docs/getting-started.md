# Getting Started

## Install dependencies

```bash
make setup
```

You can inspect the common local commands with:

```bash
make help
```

If you want local durable state, start Postgres with the included infra:

```bash
make up
make db-setup
```

## Configure environment

Create `.env` from `.env.example` and set, at minimum:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY`
- `PERPLEXITY_AT_HOME_DEFAULT_MODEL` if you want to override the default `openai:gpt-5.4`
- `PERPLEXITY_AT_HOME_POSTGRES__*` values for persistent runs

LangSmith tracing is optional. If present, `LANGSMITH_API_KEY` and
`LANGCHAIN_TRACING_V2=true` are picked up from settings.

## Run the package

In-memory execution:

```bash
make quick QUESTION="What is Tavily?"
make pro QUESTION="What changed recently in Tavily's LangChain integration?"
make deep QUESTION="What changed recently in Tavily's LangChain integration?"
```

Persistent execution:

```bash
make deep-persistent QUESTION="What changed recently in Tavily's LangChain integration?"
```

Dashboard execution:

```bash
make dashboard
```

Recommended dashboard path:

1. Run `make dashboard` for a quick in-memory smoke test.
2. Turn persistence on in the sidebar only after `make up` and `make db-setup`.
3. Start with `quick-search`, then move to `pro-search` or `deep-research`.

## Validate locally

```bash
make lint
make test
make test-e2e
make docs-build
pdm build
```

`make test-e2e` is opt-in and expects real `OPENAI_API_KEY`, `TAVILY_API_KEY`,
and a reachable Postgres instance.
