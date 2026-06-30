# Contributing to the BTC Sentiment-Driven LSTM Pipeline

Thanks for your interest in contributing! This is a solo-maintained project, but high-quality PRs are welcome.

> **Maintainer:** Nassim K. — nassim@kinzoils.com
> **Response SLA:** Issues reviewed within 5 business days; PRs within 10.

## Code of Conduct

By participating you agree to abide by the [Code of Conduct](./CODE_OF_CONDUCT.md). Report issues to nassim@kinzoils.com.

## How to Contribute

- **Reporting bugs** — open an issue using the Bug Report template.
- **Suggesting features** — open an issue using the Feature Request template.
- **Reporting security vulnerabilities** — DO NOT open a public issue. See [SECURITY.md](./SECURITY.md).
- **Submitting code** — fork, branch, PR.

## Development Setup

```bash
git clone https://github.com/nassim0014/btc-llm-sentiment.git
cd btc-llm-sentiment

# Option 1: Makefile (recommended)
make setup    # creates .venv, installs pinned requirements.txt
make test     # runs pytest suite
make lint     # syntax-checks all Python files

# Option 2: Manual
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Git Workflow

1. **Branch from `main`**: `feat/<desc>`, `fix/<desc>`, `chore/<desc>`, `docs/<desc>`, `security/<desc>`.
2. **Conventional Commits**: `feat(lstm): add bidirectional variant`, `fix(backtest): correct Kelly cap edge case`.
3. **Keep PRs small** — one logical change, < 400 lines diff.
4. **Tests required** — every code PR must keep `pytest tests/ -v` green.
5. **Lint must pass** — `ruff check src/ scripts/ tests/`.

## Pre-commit Hooks (recommended)

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

Runs `ruff`, `bandit`, `gitleaks`, `pytest` before every commit.

## Code Style

- **Python:** `ruff format` (Black-compatible, 100-char line). Type hints on public functions.
- **Tests:** Pytest in `tests/`. Use the fixtures in `tests/test_backtest.py` as a pattern.
- **Notebooks:** Exempt from linting (see `ruff.toml` per-file-ignores).

## ML-Specific Guidelines

- **Reproducibility:** All deps are pinned in `requirements.txt`. Bump versions deliberately via PR, not by accident.
- **Random seeds:** Use `RANDOM_STATE = 42` (defined in `scripts/run_pipeline.py`) for all stochastic operations.
- **Model artifacts:** If you retrain and commit a new `.pkl` or `.keras`, you MUST update `BUNDLE_SHA256` / `MODEL_SHA256` in `src/config.py`. The safe loader will refuse to load a mismatched file.
- **Pickle safety:** Never use `pickle.load()` on committed files directly — always go through `src/utils/safe_pickle.py::safe_load_bundle()`.

## Security-Sensitive Changes

If your PR touches any of the following, expect extra review:

- `src/utils/safe_pickle.py` — the pickle security layer
- `src/config.py` — artifact hashes, URLs
- `.github/workflows/` — CI/CD pipeline
- `Dockerfile` — container security
- `SECURITY.md` or `MODEL_CARD.md`

## License

By contributing, you agree your contributions will be licensed under the MIT License.
