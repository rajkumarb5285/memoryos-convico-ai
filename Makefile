.PHONY: help install migrate run test eval benchmark quality coverage clean zip

PYTHON := .venv/Scripts/python.exe
PYTEST := .venv/Scripts/pytest.exe
RUFF   := .venv/Scripts/ruff.exe
BANDIT := .venv/Scripts/bandit.exe

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Create venv and install dependencies
	uv venv .venv --python 3.11
	uv pip install -r requirements.txt

migrate:  ## Apply Alembic migrations
	$(PYTHON) -m alembic upgrade head

run:  ## Start FastAPI server (port 8000)
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:  ## Run all 31 tests
	$(PYTEST) tests/ -v

eval:  ## Run 105-case evaluation suite
	PYTHONPATH=. $(PYTHON) evaluation/run_evaluation.py

benchmark:  ## Run performance benchmark (50 req, concurrency=5)
	PYTHONPATH=. $(PYTHON) evaluation/run_benchmark.py

quality:  ## Run ruff lint + format check + bandit
	$(RUFF) check app/ tests/
	$(RUFF) format --check app/ tests/
	$(BANDIT) -r app/ -ll

coverage:  ## Run tests with HTML coverage report
	$(PYTEST) tests/ --cov=app --cov-report=html:RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT/06_TEST_COVERAGE_REPORT --cov-report=term-missing

zip:  ## Package submission folder
	powershell Compress-Archive -Path RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT -DestinationPath RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT.zip -Force
	@echo "Created RAJKUMAR_PANDEY_CONVICO_AI_ASSIGNMENT.zip"

clean:  ## Remove __pycache__, .pytest_cache, .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .coverage htmlcov

