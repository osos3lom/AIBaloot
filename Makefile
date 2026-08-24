.PHONY: help install sync lock lint format type test test-fast cov check pre-commit clean

UV ?= uv

help:  ## Show available targets
	@awk 'BEGIN{FS=":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install runtime + dev deps (resolves lockfile)
	$(UV) sync --all-extras

sync: install  ## Alias for install

lock:  ## Refresh uv.lock
	$(UV) lock

lint:  ## Ruff check
	$(UV) run ruff check .

format:  ## Ruff format (writes)
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

type:  ## Mypy strict
	$(UV) run mypy

test:  ## Pytest with coverage
	$(UV) run pytest

test-fast:  ## Pytest, no coverage, fail fast
	$(UV) run pytest -x --no-cov -q

cov: test  ## Alias for test

check: lint type test  ## Full local CI gate

pre-commit:  ## Install + run all pre-commit hooks
	$(UV) run pre-commit install
	$(UV) run pre-commit run --all-files

clean:  ## Remove caches and build artifacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache build dist *.egg-info coverage.xml htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
