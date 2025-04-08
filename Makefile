# --- AI Agents Workshop Makefile ---
# Requires: Python >= 3.11
# This Makefile helps you setup and clean virtual environments for each assignment
# Compatible: Linux / macOS (for Windows see README for alternative)

.PHONY: setup clean help
.DEFAULT_GOAL := help

# --- Setup Environments ---

# Generic setup for all assignments (except playwright-based)
setup-%:
	@bash -c '\
		python3 -c "import sys; \
		exit(not (sys.version_info >= (3, 11)))" || \
		(echo "❌ Python >= 3.11 is required" && exit 1) \
	'
	python3 -m venv .venv-$*
	@echo "Virtual environment .venv-$* created." && \
	. .venv-$*/bin/activate && \
	echo "Installing $*/requirements.txt..." && \
	pip install --upgrade pip && \
	pip install -r $*/requirements.txt && \
	[ -f $*/postinstall.sh ] && bash $*/postinstall.sh || true && \
	echo "✅ Environment .venv-$* is ready." && \
	echo "🔹 To activate:" && \
	echo "source .venv-$*/bin/activate"

# --- Dev ---

dev:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Please activate a virtual environment before running 'make dev'"; \
		exit 1; \
	fi
	@echo "Installing development tools..."
	@pip install -r requirements-dev.txt
	@echo "✅ Development tools installed into $$VIRTUAL_ENV"

# --- Cleanup ---

clean:
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "❌ You're currently in a virtualenv ($$VIRTUAL_ENV) — run 'deactivate' first."; \
		exit 1; \
	fi
	@echo "Removing all virtual environments..." && \
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
	@for dir in autogen browseruse crewai langgraph rag voice; do \
		printf "  make setup-%-12s  # Setup for %s\n" $$dir $$dir; \
	done
	@echo "  make dev                 # Install dev tools into current environment"
	@echo "  make clean               # Remove all virtual environments"
	@echo ""
