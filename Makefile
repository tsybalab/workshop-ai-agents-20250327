# --- AI Agents Workshop Makefile ---
# Requires: Python >= 3.11
# This Makefile helps you setup and clean virtual environments for each assignment
# Compatible: Linux / macOS (for Windows see README for alternative)

.PHONY: setup clean help
.DEFAULT_GOAL := help

# --- Setup Environments ---

# Generic setup for all assignments (except playwright-based)
setup-%:
	python3 --version | grep -E "3\.11" >/dev/null || (echo "❌ Python >= 3.11 is required" && exit 1)
	python3 -m venv .venv-$*
	@echo "Virtual environment .venv-$* created."
	. .venv-$*/bin/activate && \
	echo "Installing requirements-$*.txt..." && \
	pip install --upgrade pip && \
	pip install -r requirements-$*.txt && \
	echo "✅ Environment .venv-$* is ready." && \
	echo "To activate: source .venv-$*/bin/activate"

# Special setup for browser task (Playwright)
setup-browseruse:
	python3 --version | grep -E "3\.11" >/dev/null || (echo "❌ Python >= 3.11 is required" && exit 1)
	python3 -m venv .venv-browseruse
	@echo "Virtual environment .venv-browseruse created."
	. .venv-browseruse/bin/activate && \
	echo "Installing requirements-browseruse.txt and Playwright..." && \
	pip install --upgrade pip && \
	pip install -r requirements-browseruse.txt && \
	pip install playwright && \
	python -m playwright install && \
	echo "✅ Environment .venv-browseruse (with Playwright) is ready." && \
	echo "To activate: source .venv-browseruse/bin/activate"

# --- Dev ---

dev:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Please activate a virtual environment before running 'make dev'"; \
		exit 1; \
	fi
	pip install -r requirements-dev.txt
	@echo "✅ Development tools installed into $$VIRTUAL_ENV"

# --- Cleanup ---

clean:
	rm -rf .venv-*
	@echo "🧹 All .venv-* environments have been removed."

# --- Help ---

help:
	@echo ""
	@echo "AI Agents Workshop Makefile"
	@echo "==========================="
	@echo "⚠ Requires Python >= 3.11"
	@echo ""
	@echo "Usage:"
	@echo "  make setup-langgraph     # Setup for LangGraph task"
	@echo "  make setup-autogen       # Setup for Autogen task"
	@echo "  make setup-crewai        # Setup for CrewAI task"
	@echo "  make setup-rag           # Setup for RAG task"
	@echo "  make setup-browseruse    # Setup for Browser task (Playwright)"
	@echo "  make setup-voice         # Setup for Voice assistant task"
	@echo "  make dev                 # Install dev tools into current environment"
	@echo "  make clean               # Remove all virtual environments"
	@echo ""
