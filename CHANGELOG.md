# Changelog

All notable changes to the **BTC Sentiment-Driven LSTM Trading Pipeline** are documented here.

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### 🛡️ Security
- **Pickle deserialization hardening.** All `pickle.load()` calls on committed
  model artifacts now go through `src/utils/safe_pickle.py::safe_load_bundle()`,
  which (1) verifies a SHA256 integrity hash pinned in `src/config.py`, and (2)
  uses a restricted unpickler that only allows numpy arrays, sklearn
  `StandardScaler`, and built-in collections. Any other class import is blocked.
  Defense against a compromised repo shipping a malicious `.pkl` that would
  execute arbitrary code on Streamlit Cloud users.
- **Supply-chain hardening.** `sentiment_alert.yml` no longer auto-commits to
  `main` — alert logs are uploaded as workflow artifacts instead. Was a
  supply-chain risk: any compromised secret could push directly to main.
- **Dependency pinning.** All deps in `requirements.txt` are now pinned to
  exact versions (was `>=`) for ML reproducibility.
- **CI permissions.** Workflows use `contents: read` by default.
  `security-events: write` only on the security scan job.

### ✨ Added
- `src/config.py` — centralized config (NEWS_URL, model paths, SHA256 hashes,
  request timeouts, alert thresholds). Was hardcoded in 3+ files.
- `src/utils/safe_pickle.py` — safe pickle loader with SHA256 + restricted
  unpickler.
- `Dockerfile` — multi-stage build for the Streamlit dashboard (builder +
  runtime, non-root user `btc`, slim base).
- `.dockerignore` — prevents `.git`, `node_modules`, `.env` from entering
  the build context.
- `.env.example` — documents all env vars (SMTP, Slack, Discord, HF cache).
- `SECURITY.md` — full security policy with pickle risk explanation.
- `MODEL_CARD.md` — ML model card (intended use, training data, evaluation,
  limitations, ethical considerations, reproducibility).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `.github/CODEOWNERS`.
- Issue templates: `bug_report.yml`, `feature_request.yml`.
- `.github/PULL_REQUEST_TEMPLATE.md` with ML-retrain checklist.
- `.github/dependabot.yml` — weekly pip + GitHub Actions + Docker updates.
- `.pre-commit-config.yaml` — ruff, bandit, gitleaks, pytest.
- `.github/workflows/ci.yml` — replaces `test.yml`; adds ruff, bandit,
  pip-audit, gitleaks, trivy container scan, Codecov, Docker build test.
- `ruff.toml` — config with per-file ignores for notebooks and tests.

### 🔧 Changed
- Deprecated `torch.cuda.amp.autocast()` → `torch.amp.autocast("cuda")`
  (PyTorch 2.4+ API change).
- Deprecated `tf.keras.optimizers.Adam(lr)` → `learning_rate=lr`.
- `warnings.filterwarnings("ignore")` → selective filters per module
  (was hiding security and correctness warnings).
- `requests.get(NEWS_URL)` now sends a User-Agent header and retries with
  exponential backoff (was failing silently on transient 503s).
- `sentiment_alert.py` reads config from `src/config.py` instead of
  duplicating constants.

### 🗑️ Removed
- `.github/workflows/test.yml` — replaced by `.github/workflows/ci.yml`.
- Direct `pickle.load()` calls in `run_optuna_search.py`,
  `run_risk_managed_backtest.py`, `run_shap_analysis.py`, and the Live
  Predictions Streamlit page. All go through `safe_load_bundle()` now.

## [1.0.0] — 2025-06-23

Initial public release: FastAPI + LSTM + FinBERT + Optuna + risk-managed
backtester + SHAP + Streamlit dashboard.
