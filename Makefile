# Makefile
.PHONY: help up down logs migrate ingest train test test-unit test-integration \
        lint format build clean dashboard

help:
	@echo "Loan Default Pipeline — common commands"
	@echo ""
	@echo "  make up                 Start full stack (postgres, mlflow, airflow, api)"
	@echo "  make down               Stop all containers"
	@echo "  make logs               Tail logs from all services"
	@echo "  make migrate            Apply database migrations"
	@echo "  make ingest FILE=path   Ingest a CSV file into the raw data table"
	@echo "  make train              Trigger the training DAG manually"
	@echo "  make test               Run unit + integration tests"
	@echo "  make test-unit          Run unit tests only"
	@echo "  make test-integration   Run integration tests (needs live infra)"
	@echo "  make lint               Run ruff + black + isort checks"
	@echo "  make format             Auto-format code with black + isort"
	@echo "  make build              Build all Docker images"
	@echo "  make dashboard          Open the Streamlit monitoring dashboard"
	@echo "  make clean              Remove caches, __pycache__, build artefacts"

up:
	docker compose up -d --build
	@echo "Airflow:    http://localhost:8080"
	@echo "MLflow:     http://localhost:5000"
	@echo "API docs:   http://localhost:8000/docs"
	@echo "Monitoring: http://localhost:8001"

down:
	docker compose down

logs:
	docker compose logs -f

migrate:
	python scripts/run_migrations.py

ingest:
	python scripts/ingest_initial_data.py --file $(FILE) --apply-migrations

train:
	python scripts/trigger_dag.py --dag loan_default_training

test: test-unit test-integration

test-unit:
	pytest tests/unit -v --cov=src --cov-report=term-missing

test-integration:
	RUN_INTEGRATION_TESTS=1 pytest tests/integration --integration -v

lint:
	ruff check src/ tests/ dags/
	black --check src/ tests/ dags/
	isort --check-only src/ tests/ dags/

format:
	black src/ tests/ dags/
	isort src/ tests/ dags/

build:
	docker compose build

dashboard:
	@echo "Opening monitoring dashboard at http://localhost:8001"
	@python -m webbrowser http://localhost:8001 2>/dev/null || true

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov coverage.xml .coverage
