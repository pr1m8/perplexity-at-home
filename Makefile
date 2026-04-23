SHELL := /bin/bash

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
	infra-up infra-down infra-restart infra-logs infra-psql infra-status infra-destroy infra-setup \
	lint test test-e2e docs-build docs-serve build dashboard release-check

infra-up:
	$(COMPOSE) up -d

infra-down:
	$(COMPOSE) down

infra-restart:
	$(COMPOSE) restart

infra-logs:
	$(COMPOSE) logs -f postgres

infra-psql:
	$(COMPOSE) exec postgres psql -U "$(PERPLEXITY_AT_HOME_POSTGRES__USER)" -d "$(PERPLEXITY_AT_HOME_POSTGRES__DATABASE)"

infra-status:
	$(COMPOSE) ps

infra-destroy:
	$(COMPOSE) down -v --remove-orphans

infra-setup:
	pdm run perplexity-at-home persistence setup

lint:
	pdm run ruff check src tests

test:
	pdm run pytest

test-e2e:
	PERPLEXITY_AT_HOME_RUN_E2E=true pdm run pytest --no-cov -p no:rerunfailures -m e2e tests/test_live_e2e.py

docs-build:
	pdm run mkdocs build --strict

docs-serve:
	pdm run mkdocs serve

build:
	pdm build

dashboard:
	pdm run perplexity-at-home dashboard

release-check: lint test docs-build build
