SHELL := /bin/bash
.DEFAULT_GOAL := help

PERPLEXITY_AT_HOME_POSTGRES__HOST ?= $(if $(POSTGRES_HOST),$(POSTGRES_HOST),localhost)
PERPLEXITY_AT_HOME_POSTGRES__PORT ?= $(if $(POSTGRES_PORT),$(POSTGRES_PORT),5442)
PERPLEXITY_AT_HOME_POSTGRES__USER ?= $(if $(POSTGRES_USER),$(POSTGRES_USER),postgres)
PERPLEXITY_AT_HOME_POSTGRES__PASSWORD ?= $(if $(POSTGRES_PASSWORD),$(POSTGRES_PASSWORD),postgres)
PERPLEXITY_AT_HOME_POSTGRES__DATABASE ?= $(if $(POSTGRES_DB),$(POSTGRES_DB),perplexity_at_home)
export PERPLEXITY_AT_HOME_POSTGRES__HOST
export PERPLEXITY_AT_HOME_POSTGRES__PORT
export PERPLEXITY_AT_HOME_POSTGRES__USER
export PERPLEXITY_AT_HOME_POSTGRES__PASSWORD
export PERPLEXITY_AT_HOME_POSTGRES__DATABASE

COMPOSE := docker compose --env-file .env -f infra/compose.yaml

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: \
	help env install dashboard-install setup up down restart logs psql status destroy db-setup \
	infra-up infra-down infra-restart infra-logs infra-psql infra-status infra-destroy infra-setup \
	lint test test-e2e docs-build docs-serve build dashboard dash quick pro deep deep-persistent release-check

QUESTION ?= What is Tavily?
RUNNER := pdm run perplexity-at-home

help: ## Show the common local commands
	@awk 'BEGIN {FS = ":.*##"; printf "\nTargets:\n"} /^[a-zA-Z0-9_.-]+:.*##/ { printf "  %-18s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

env: ## Create .env from .env.example if it does not exist
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env from .env.example"; else echo ".env already exists"; fi

install: ## Install core development dependencies
	pdm install -G test -G docs

dashboard-install: ## Install the dashboard dependency group
	pdm install -G dashboard

setup: ## Install local dependencies and bootstrap a dashboard-ready .env
	@$(MAKE) install
	@$(MAKE) dashboard-install
	@$(MAKE) env

up: infra-up ## Start local Postgres infra

down: infra-down ## Stop local Postgres infra

restart: infra-restart ## Restart local Postgres infra

logs: infra-logs ## Tail Postgres logs

psql: infra-psql ## Open psql inside the local Postgres container

status: infra-status ## Show local infra container status

destroy: infra-destroy ## Remove local infra containers and volumes

db-setup: infra-setup ## Initialize LangGraph persistence tables

infra-up: ## Start local Postgres infra
	$(COMPOSE) up -d

infra-down: ## Stop local Postgres infra
	$(COMPOSE) down

infra-restart: ## Restart local Postgres infra
	$(COMPOSE) restart

infra-logs: ## Tail Postgres logs
	$(COMPOSE) logs -f postgres

infra-psql: ## Open psql inside the local Postgres container
	$(COMPOSE) exec postgres psql -U "$(PERPLEXITY_AT_HOME_POSTGRES__USER)" -d "$(PERPLEXITY_AT_HOME_POSTGRES__DATABASE)"

infra-status: ## Show local infra container status
	$(COMPOSE) ps

infra-destroy: ## Remove local infra containers and volumes
	$(COMPOSE) down -v --remove-orphans

infra-setup: ## Initialize LangGraph persistence tables
	pdm run perplexity-at-home persistence setup

lint: ## Run Ruff
	pdm run ruff check src tests

test: ## Run the unit and integration suite
	pdm run pytest

test-e2e: ## Run the manual-only live E2E suite
	PERPLEXITY_AT_HOME_RUN_E2E=true pdm run pytest --no-cov -p no:rerunfailures -m e2e tests/test_live_e2e.py

docs-build: ## Build docs locally
	pdm run mkdocs build --strict

docs-serve: ## Serve docs locally
	pdm run mkdocs serve

build: ## Build the package distributions
	pdm build

dashboard: ## Launch the packaged Streamlit dashboard
	$(RUNNER) dashboard

dash: dashboard ## Alias for dashboard

quick: ## Run quick-search with QUESTION="..."
	$(RUNNER) quick-search "$(QUESTION)"

pro: ## Run pro-search with QUESTION="..."
	$(RUNNER) pro-search "$(QUESTION)"

deep: ## Run deep-research with QUESTION="..."
	$(RUNNER) deep-research "$(QUESTION)"

deep-persistent: ## Run deep-research with persistence enabled
	$(RUNNER) deep-research --persistent --setup-persistence "$(QUESTION)"

release-check: lint test docs-build build ## Run the main local release gates
