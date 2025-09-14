# Simple Makefile for RAG system testing

.PHONY: help test test-simple install clean

help:
	@echo "Available commands:"
	@echo "  test-simple  - Run simple RAG functionality test"
	@echo "  test         - Run all tests"
	@echo "  install      - Install dependencies"
	@echo "  clean        - Clean test artifacts"

# Install dependencies
install:
	uv sync --dev

# Run simple test
test-simple:
	uv run pytest tests/test_rag_simple.py -v -s

# Run all tests
test:
	uv run pytest tests/ -v

# Clean up
clean:
	find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache/ || true