# Makefile for the BTC Sentiment-Driven LSTM Trading Pipeline.
#
# Common targets:
#   make setup   — create a virtualenv and install all dependencies
#   make run     — execute the Phase 1 master pipeline locally (fast path)
#   make phase2  — execute the Phase 2 advanced pipeline locally (Optuna + Risk + SHAP)
#   make test    — run the pytest suite
#   make lint    — quick syntax check on every Python file in src/ and scripts/
#   make clean   — remove the venv and interim artifacts
#   make help    — print this help
#
# Usage:
#   make setup
#   make run

PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: help setup run phase2 test lint clean

help:
	@echo "BTC Sentiment-Driven LSTM Trading Pipeline — Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make setup   — create venv at $(VENV) and install requirements.txt"
	@echo "  make run     — run the Phase 1 master pipeline (TextBlob sentiment, fast)"
	@echo "  make phase2  — run the Phase 2 advanced pipeline (Optuna + Risk + SHAP)"
	@echo "  make test    — run the pytest suite in tests/"
	@echo "  make lint    — syntax-check all Python files in src/ and scripts/"
	@echo "  make clean   — remove venv and interim artifacts"
	@echo ""
	@echo "Typical workflow:"
	@echo "  make setup && make run"

setup: $(VENV)/bin/activate
	@echo "✅ Virtualenv ready at $(VENV)"

$(VENV)/bin/activate:
	@echo "📦 Creating virtualenv at $(VENV) ..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install pytest
	@echo "✅ Installed all dependencies from requirements.txt"

run: setup
	@echo "🚀 Running Phase 1 master pipeline (--use-precomputed for fast path) ..."
	$(PY) scripts/run_pipeline.py --use-precomputed

phase2: setup
	@echo "🚀 Running Phase 2 advanced pipeline ..."
	$(PY) scripts/generate_interim_features.py
	$(PY) scripts/run_optuna_search.py --n-trials 5 --epochs 10
	$(PY) scripts/run_risk_managed_backtest.py
	$(PY) scripts/run_shap_analysis.py

test: setup
	@echo "🧪 Running pytest suite ..."
	$(VENV)/bin/pytest tests/ -v

lint:
	@echo "🔍 Syntax-checking Python files ..."
	@find src scripts -name "*.py" -print0 | xargs -0 -n1 $(PYTHON) -m py_compile
	@echo "✅ All files compile cleanly"

clean:
	@echo "🧹 Cleaning up ..."
	rm -rf $(VENV)
	rm -rf notebooks/interim
	rm -rf .pytest_cache __pycache__
	@echo "✅ Cleaned."
