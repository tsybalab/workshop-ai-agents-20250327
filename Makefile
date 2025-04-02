# --- AI Agents Workshop Makefile ---
# Requires: Python >= 3.11
# This Makefile helps you setup and clean virtual environments for each assignment
# Compatible: Linux / macOS (for Windows see README for alternative)

.PHONY: setup clean help

# --- Setup Environments ---

# Generic setup for Assignments 1-4 and 6
setup-%:
	@if [ "$*" != "05" ]; then \
		python3 --version | grep -E "3\.11" >/dev/null || (echo "❌ Python >= 3.11 is required" && exit 1); \
		python3 -m venv .venv-$* && \
		echo "Virtual environment .venv-$* created." && \
		. .venv-$*/bin/activate && \
		echo "Installing requirements-$*.txt..." && \
		pip install --upgrade pip && \
		pip install -r requirements-$*.txt && \
		echo "✅ Environment .venv-$* is ready." && \
		echo "To activate: source .venv-$*/bin/activate"; \
	fi

# Special setup for Assignment 5 (requires Playwright install)
setup-05:
	python3 --version | grep -E "3\.11" >/dev/null || (echo "❌ Python >= 3.11 is required" && exit 1)
	python3 -m venv .venv-05
	@echo "Virtual environment .venv-05 created."
	. .venv-05/bin/activate && \
	echo "Installing requirements-05.txt and Playwright..." && \
	pip install --upgrade pip && \
	pip install -r requirements-05.txt && \
	pip install playwright && \
	python -m playwright install && \
	echo "✅ Environment .venv-05 (with Playwright) is ready." && \
	echo "To activate: source .venv-05/bin/activate"

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
	@echo "  make setup-01     # Setup environment for Assignment 1"
	@echo "  make setup-02     # Setup environment for Assignment 2"
	@echo "  make setup-03     # Setup environment for Assignment 3"
	@echo "  make setup-04     # Setup environment for Assignment 4"
	@echo "  make setup-05     # Setup environment for Assignment 5 (includes playwright install)"
	@echo "  make setup-06     # Setup environment for Assignment 6"
	@echo "  make clean        # Remove all virtual environments"
	@echo ""