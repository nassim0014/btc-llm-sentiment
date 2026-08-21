# btc-llm-sentiment — Agent Guidance

## What this is

BTC sentiment-driven LSTM trading pipeline. Trains an LSTM on Bitcoin
price + news sentiment data, backtests trading strategies, and provides
an interactive Streamlit dashboard. Public repo under `nassim0014`.

## Rules (inherited from the loop engine)

1. **Never push directly to `main`** — open a PR, squash-merge.
2. **Squash-merge only** — so a revert is one command.
3. **Never create GitHub issues** — owner wants 0%.
4. **Never change repository visibility.**
5. **Never rewrite git history or force-push.**
6. **Never delete a branch other than one created this run.**

## Architecture

```
src/
  backtest/       Backtest engine + simulator (risk-managed)
  cv/             Cross-validation (walk-forward, Optuna search)
  inference/      FinBERT sentiment inference
  interpretability/  SHAP feature importance
  models/         LSTM model factory
pages/            Streamlit dashboard pages (4 pages)
  1_ Phase 1 Deep-Dive
  2_ Phase 2 Deep-Dive
  3_ Live Predictions
  4_ Backtest Simulator
tests/            pytest suite (82 tests)
notebooks/        Jupyter notebooks (Colab pipeline)
```

## Development workflow

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn pytest ruff

# Tests (82 tests, ~1.5s)
pytest tests/ -v

# Lint
ruff check src/ scripts/ tests/

# Full local CI gate
make check
```

## CI

`.github/workflows/ci.yml` — runs on push + pull_request to main.
- Test job (Python 3.11 + 3.12): pytest + ruff + bandit SAST
- ML smoke job: installs tensorflow + torch CPU, import-checks ML modules
- Security scans: pip-audit + gitleaks
- Docker build smoke test

## Known traps

- **Docker build fails in CI**: pre-existing, unrelated to code changes.
  The Dockerfile references files that need the full pipeline output.
- **Notebook ruff errors**: 113 pre-existing errors in `notebooks/` —
  excluded from the ruff CI gate (only `src/ scripts/ tests/` are checked).
- **requirements.txt has duplicate pins**: items 1-3 in the backlog were
  about fixing duplicate ipykernel/nbconvert pins. Already resolved.
