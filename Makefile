.PHONY: help install install-dev test test-cov lint format clean build publish docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in development mode
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -e ".[dev,test,docs]"

test: ## Run tests
	export LD_LIBRARY_PATH=$$(python -c "import torch; print(torch.__file__)")/lib:$$LD_LIBRARY_PATH && \
	python -m pytest test/ -v



lint: ## Run linting checks
	ruff check .

format: ## Format code
	ruff format .
	ruff check --fix .

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf floating_point/build/
	rm -rf floating_point/*.so
	rm -rf floating_point/floating_point.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

build: ## Build the package
	cd floating_point && python setup.py build_ext --inplace
	python -m build

build-wheel: ## Build wheel only
	python -m build --wheel

build-sdist: ## Build source distribution only
	python -m build --sdist

check: ## Check the built package
	twine check dist/*

publish-test: ## Publish to TestPyPI
	twine upload --repository testpypi dist/*

publish: ## Publish to PyPI
	twine upload dist/*



pre-commit: ## Install pre-commit hooks
	pre-commit install

pre-commit-run: ## Run pre-commit on all files
	pre-commit run --all-files

setup-dev: ## Set up development environment
	pip install -e ".[dev]"
	pip install pytest parameterized
	pre-commit install

setup-ci: ## Set up CI environment
	pip install pytest parameterized
	export LD_LIBRARY_PATH=$$(python -c "import torch; print(torch.__file__)")/lib:$$LD_LIBRARY_PATH
	cd floating_point && python setup.py build_ext --inplace

version: ## Show current version
	@python -c "import floating_point; print(floating_point.__version__)"

check-deps: ## Check for outdated dependencies
	pip list --outdated

update-deps: ## Update dependencies
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -e ".[dev]"
	pip install --upgrade pytest parameterized



full-check: ## Run all checks (lint, test)
	make lint
	make test
