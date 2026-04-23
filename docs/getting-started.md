# Getting Started

## Install dependencies

```bash
pdm install -G test -G docs
```

If you want local durable state, start Postgres with the included infra:

```bash
make infra-up
make infra-setup
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
pdm run perplexity-at-home deep-research "What changed recently in Tavily's LangChain integration?"
```

Persistent execution:

```bash
pdm run perplexity-at-home deep-research --persistent --setup-persistence "What changed recently in Tavily's LangChain integration?"
```

## Validate locally

```bash
make lint
make test
make docs-build
pdm build
```
