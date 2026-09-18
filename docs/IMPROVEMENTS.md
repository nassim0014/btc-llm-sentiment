# Improvement backlog

The queue the closed-loop improvement cycle works from. One item per run,
highest value first. Items are added whenever something is noticed but is
too far out of scope to fix on the spot.

**Rules**

- Work the top unblocked item. Don't cherry-pick easy ones.
- One PR per item. If an item turns out to be three things, split it and
  re-rank.
- Move finished items to *Done* with the PR number. Don't delete them —
  the history is how the next session learns what has already been tried.
- If an item turns out to be wrong or no longer applies, move it to
  *Dropped* with the reason. That is a legitimate outcome.

This file is the first cycle's output — there was no backlog before, so
this PR (creating it) is itself item 0.

---

## Now

### 11. 🔴 `Test (Python 3.11/3.12)` red since 2026-09-10 — `tests/test_sentiment_alert.py` imports `requests`, missing from the lightweight CI install list   `source: ci-red`

_First observed 2026-09-10, still failing on `main` as of the latest CI run
(`34692181052`, 2026-09-12) — over the 7-day bar._

`tests/test_sentiment_alert.py` (added 2026-09-05) collects
`scripts/sentiment_alert.py`, which does a module-level `import requests`
(`scripts/sentiment_alert.py:41`). The lightweight CI test job's install
list (`pip install numpy pandas scikit-learn pytest pytest-cov ruff bandit`
— see `.github/workflows/ci.yml`) never included `requests`, so both
`Test (Python 3.11)` and `Test (Python 3.12)` fail at collection:

```
ERROR collecting tests/test_sentiment_alert.py
ImportError while importing test module '.../tests/test_sentiment_alert.py'.
scripts/sentiment_alert.py:41: in <module>
    import requests
E   ModuleNotFoundError: No module named 'requests'
```

This is separate from item 10's torch/transformers/Trivy failures (same CI
run, different jobs) — fixing item 10 will not clear this one. Fix is a
one-line addition to the lightweight test job's install list
(`.github/workflows/ci.yml`), which is a forbidden path for this loop —
flagging for the owner or a loop with workflow-file permission.

Loop-Agent: repo-review-loop / claude / laptop (2026-09-18)

### 10. 🔴 `main` CI red on two jobs — both now trace to CVE-laden ML pins (OWNER DECISION)   `source: ci-red`

_Consolidates former items 9, the un-numbered Docker item, and item 7 —
all three described a `numpy==2.5.0` / `pandas==3.0.3` resolution deadlock
that **no longer exists**. PR #48 rolled `requirements.txt` back to
`numpy==1.26.4` / `pandas==2.2.2`; `pip install -r requirements.txt` now
resolves cleanly on 3.11 and 3.12. Re-verified 2026-09-10 against the
current file and the latest `main` CI run (`33899902889`, commit
`d468600`). Both jobs are still red, but for one shared new reason._

**Both red jobs come down to: `torch==2.4.1` and `transformers==4.44.2`
carry many published CVEs, and the fix versions are major upgrades.**

`Security scans` — `pip-audit -r requirements.txt --strict` runs to
completion and reports:
```
Found 82 known vulnerabilities in 7 packages
transformers 4.44.2  — ~30 advisories; earliest fix 4.48.0, several need 5.x, CVE-2026-9856 needs 5.10.0
torch        2.4.1   — ~15 advisories; fixes spread across 2.5.0 → 2.9.0
requests     2.32.3  — 2 advisories, fix 2.32.4 / 2.33.0  (minor, dependabot-sized)
```
`--strict` fails on any advisory, so the `transformers`/`torch` set alone
guarantees red. `transformers 4.44.2 → 5.x` and `torch 2.4.1 → 2.9.x` are
breaking-change upgrades on the two libraries that do FinBERT sentiment
scoring and LSTM training — **an owner call on migration risk vs. residual
CVE risk**, not something the closed loop or a routine dependency PR should
force through. Options for the owner:
  - schedule the `transformers` + `torch` major upgrade deliberately (own
    branch, re-validate FinBERT scores + LSTM training numerically), or
  - accept the residual risk and relax `--strict` for exactly these two
    packages with an explicit, commented `--ignore-vuln` list (never a
    blanket `|| true`).

`Docker build` — the build succeeds; the job fails at the Trivy step with
`unable to find the specified image "btc-llm-sentiment:ci"`.
`docker/build-push-action@v7` runs `push: false` with no `load`, so buildx
keeps the image in its own cache ("WARNING: No output specified with
docker-container driver") and never hands it to the engine Trivy queries.
A one-line workflow fix (`load: true` on the build step) clears *that*
error — **but the image bundles the same `torch`/`transformers`, so Trivy
(`severity: HIGH,CRITICAL`, `exit-code: 1`) then fails on the same CVEs.**
So the Docker job cannot go green ahead of the ML-pin decision either; the
`load: true` fix is only worth doing bundled with that upgrade.

**Consequence:** merge-contract rule 5 (base branch must be green) blocks
auto-merge for *every* PR in this repo until the owner acts. The closed
loop cannot clear this itself — `requirements*.txt` and `.github/workflows/**`
are both forbidden paths, and the upgrade needs numeric re-validation.

Loop-Agent: closed-loop / claude / laptop  (2026-09-10 consolidation)

## Next

### 4. ~~ML-heavy modules at 0% coverage — needs an owner decision~~ ✅
**Decision #4: CPU install.** Added a new `ml-smoke` CI job that installs
`tensorflow-cpu` + `torch` (CPU-only, from the PyTorch CPU wheel index)
+ `transformers` + `shap`, then import-checks all four ML modules
(`src/cv/optuna_search.py`, `src/inference/finbert.py`,
`src/interpretability/shap_explainer.py`, `src/models/lstm.py`). Adds
~3-5 min to CI run time. Catches API mismatches that the lightweight
test job (which deliberately skips ML deps) can't see.

### 5. ~~`pages/` Streamlit pages — 1,464 lines of UI with zero tests~~ ✅
Four Streamlit page files (`1_🔬_Phase_1_Deep_Dive.py` through
`4_🎛️_Backtest_Simulator.py`) total 1,464 lines and had no tests.

**All four pages extracted across 4 PRs:**
- **PR #28**: `4_🎛️_Backtest_Simulator.py` → `src/backtest/simulator.py`
  (15 tests: backtest math, vol-targeting, circuit breaker, edge cases)
- **PR #29**: `1_🔬_Phase_1_Deep_Dive.py` → `src/phase1_display.py`
  (12 tests: metric config, label formatting, column display, has_data)
- **PR #30**: `2_🚀_Phase_2_Deep_Dive.py` → `src/phase2_display.py`
  (15 tests: Optuna stats, SHAP sorting, top-N features, has_data)
- **PR #31** (this PR): `3_🎯_Live_Predictions.py` → `src/live_features.py`
  (8 tests: feature engineering, RSI bounds, no-mutation, exception→None)

Total: 50 new tests across 4 extracted modules. Page files now contain
only `st.*` UI glue + inline imports of the extracted functions.

### 6. ~~`pytest.ini` vs `pyproject.toml`~~ ✅
Consolidated `pytest.ini` + `ruff.toml` into `pyproject.toml` (single
config file). Both tools natively read `pyproject.toml`. Verified:
31 tests pass (1 skip), ruff finds the config (113 pre-existing
notebook errors unchanged — not from this migration).

---

## Done

- **PR (2026-09-10)** — Backlog consolidation only (docs). Verified item 3
  landed on `main` as **PR #46** (commit `4c7dd61`, merged 2026-09-01) and
  moved it here. Collapsed former items 9 + un-numbered Docker + 7 into a
  single item 10: their shared `numpy==2.5.0` resolution premise was fixed
  by PR #48, and both red CI jobs now trace to CVE-laden `torch==2.4.1` /
  `transformers==4.44.2` pins — re-verified against live CI run
  `33899902889` and the current `requirements.txt`. No code touched;
  local suite still 83 passed, `ruff` clean.

- **PR #46 (merged 2026-09-01)** — Item 3 remainder: made the
  signal/returns alignment in `evaluate_oof_metrics` symmetric so
  `len(y_prob) < len(close) - 1` no longer raises a broadcast
  `ValueError`. Added `tests/test_backtest.py::TestEvaluateOofMetrics::
  test_fewer_probs_than_returns_does_not_crash` (proven to fail without
  the guard by reverting it). This was the same gap the loop registry
  tracked as "item 8" — closed by this PR. Earlier `evaluate_oof_metrics`
  empty-input fix shipped in PR #34 (commit `9299692`).

- **PR #23 / #24 (verified 2026-08-29, back-filled here)**
  - **Item 1** — Duplicate `ipykernel` / `nbconvert` pins in
    `requirements.txt`. Fixed in PR #23; current file has a single pin
    each (`ipykernel==7.3.0`, `nbconvert==7.17.1`). No duplicates remain.
  - **Item 2** — `audit/` directory never created before
    `sentiment_alert.py` writes `sentiment_alerts.csv`. Fixed in PR #24
    (commit `bf57c4e`): `ALERT_LOG_PATH.parent.mkdir(parents=True,
    exist_ok=True)` now runs before the first write.

- **PR #1 (this PR)** — Created `docs/IMPROVEMENTS.md` as the first
  cycle's deliverable. Read the codebase end-to-end: 27 tests pass,
  `ruff check` clean, 37% coverage overall. Five concrete items
  identified and ranked by value/risk. Items needing owner decisions
  (ML deps in CI, notebook dependency dedup version choice) are flagged
  explicitly and not picked.

  Findings that didn't make it into ranked items but are worth knowing:
  - `requirements.txt` is otherwise well-pinned (every dep has an exact
    version), which is the right call for an ML reproducibility project.
    Only the two duplicate lines are wrong.
  - `src/config.py` is well-organised — single source of truth for
    paths, URLs, timeouts. The `AUDIT_DIR` bug is a missing side-effect,
    not a structural problem.
  - The CI matrix (Python 3.11 + 3.12) is correct and matches the
    README badge. `pre-commit` is configured. `bandit` SAST runs in CI.
    This is a well-maintained repo; the backlog is short on purpose.

## Dropped

(none yet)
