# Security Policy

## Supported Versions

The BTC Sentiment-Driven LSTM pipeline is under active development. Security fixes are applied to the latest `main` branch.

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability, please report it responsibly.

- **Email:** nassim@kinzoils.com
- **Response SLA:** Acknowledgement within 48 hours, assessment within 5 business days.

**Do not open a public GitHub issue** for security vulnerabilities.

## Security Architecture

This is a **machine-learning pipeline** with an interactive Streamlit dashboard. The threat model differs from a typical web app — the highest-risk surface is the **pickle deserialization of committed model artifacts**.

| Layer | Control |
|-------|---------|
| Model artifact integrity | `src/utils/safe_pickle.py` verifies SHA256 of `features_for_lstm.pkl` before loading. Hash is pinned in `src/config.py::BUNDLE_SHA256`. |
| Pickle deserialization | Restricted unpickler allowlists only numpy arrays, sklearn `StandardScaler`, and built-in collections. Any other class import is blocked. |
| Dependency pinning | All deps pinned to exact versions in `requirements.txt` for ML reproducibility. |
| Dependency scanning | `pip-audit` (blocking on main) + `gitleaks` + `bandit` SAST + `trivy` container scan in CI. SARIF uploaded to GitHub Security tab. |
| Container | Multi-stage Docker build; non-root user (`btc`, uid 1001); slim base. |
| Supply chain | `sentiment_alert.yml` no longer auto-commits to main. Alert logs uploaded as workflow artifacts. Dependabot + pre-commit hooks. |
| Network | All outbound HTTP uses `requests` with explicit timeout + User-Agent. Retry with exponential backoff on the news fetch. |
| Secrets | `.env` git-ignored. Alert SMTP/Slack/Discord credentials read from environment. `.env.example` documents all vars. |
| CI permissions | Workflows use `contents: read` by default. `security-events: write` only on the security scan job (for SARIF upload). |

## The Pickle Risk (Important)

The pipeline commits a `features_for_lstm.pkl` file to the repo so the Streamlit "Live Predictions" page can run without retraining. **Pickle deserialization is arbitrary code execution.** If an attacker gains write access to the repo (compromised PAT, merged malicious PR), they could ship a tampered `.pkl` that runs any code on every user who opens Live Predictions on Streamlit Cloud.

### Mitigation (current)

Two layers of defense, both in `src/utils/safe_pickle.py`:

1. **SHA256 integrity verification** — the expected hash is pinned in `src/config.py::BUNDLE_SHA256`. If the on-disk file's hash doesn't match, load is refused. Updating the hash requires a code change that goes through PR review.

2. **Restricted unpickler** — `find_class()` is overridden to only allow a strict allowlist: numpy arrays/dtypes, sklearn `StandardScaler`, and built-in collections. Any pickle that tries to import anything else (e.g. `os.system`, `subprocess.Popen`) is rejected.

### Roadmap (truly correct fix)

Replace the pickle bundle entirely with:
- **safetensors** for the numpy arrays (non-executable format)
- **JSON** for the `StandardScaler` params (mean_, scale_, var_, n_features_in_)
- Reconstruct the scaler in code from the JSON at load time

This is deferred to the roadmap because it requires retraining + regenerating the bundle + updating all consumers. Tracked as a P1 item in the audit report.

## Contact

Maintainer: **Nassim K.** — nassim@kinzoils.com
