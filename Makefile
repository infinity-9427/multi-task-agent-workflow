# Simple Makefile for RAG system testing

.PHONY: help test test-simple test-minimal test-docker test-docker-simple test-docker-minimal install clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "🧪 LOCAL TESTING:"
	@echo "  test           - Run all FastAPI client tests locally"
	@echo "  test-simple    - Run simple RAG functionality tests locally"  
	@echo "  test-minimal   - Run minimal viable tests locally"
	@echo ""
	@echo "🐳 DOCKER TESTING (recommended):"
	@echo "  test-docker          - Run all FastAPI client tests in Docker"
	@echo "  test-docker-simple   - Run simple RAG tests in Docker"
	@echo "  test-docker-minimal  - Run minimal tests in Docker"
	@echo ""
	@echo "🛠️  UTILITIES:"
	@echo "  install        - Install dependencies"
	@echo "  clean          - Clean test artifacts"

# Install dependencies
install:
	uv sync --dev

# LOCAL TESTING
test:
	uv run pytest tests/ -v --tb=short

test-simple:
	uv run pytest tests/test_rag_simple.py -v -s --tb=short

test-minimal:
	uv run pytest tests/test_rag_minimal.py -v -s --tb=short

# DOCKER TESTING (recommended - uses real API)
test-docker:
	docker compose exec api python -m pytest tests/ -v --tb=short

test-docker-simple:
	docker compose exec api python -m pytest tests/test_rag_simple.py -v -s --tb=short

test-docker-minimal:
	docker compose exec api python -m pytest tests/test_rag_minimal.py -v -s --tb=short

# Clean up
clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ || true