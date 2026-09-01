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

### 🔴 `main` CI's Docker build has been red for 7+ days — `numpy==2.5.0` needs Python 3.12, builder image is 3.11
Filed by the repo-review loop, 2026-08-25. The `Docker build` job on `main`
has failed on every run since at least 2026-08-18T17:45 (checked back
through 20+ runs, all same root cause) — confirmed still red as of this run.

`requirements.txt` pins:
```
pandas==3.0.3
numpy==2.5.0
```
`numpy==2.5.0` requires Python ≥3.12 (`Requires-Python >=3.12`), but the
Dockerfile's builder stage installs into a Python 3.11 image. The build
log is unambiguous:
```
ERROR: Could not find a version that satisfies the requirement numpy==2.5.0
  (from versions: ... 2.4.6)   ← 2.5.0 not resolvable on cp311
ERROR: No matching distribution found for numpy==2.5.0
buildx failed with: ... pip install --no-cache-dir -r requirements.txt did not complete successfully: exit code: 1
```
Every other CI job (Test 3.11/3.12, ML import smoke, Security scans) is
green — this is Docker-build-specific because only that job's base image
is pinned to 3.11 while `numpy==2.5.0` needs 3.12+.

This is currently silent in practice because the lightweight test jobs
don't build the Docker image, so nothing outside `gh pr checks` surfaces
it — but it means **no dependency PR touching pandas/numpy-adjacent code
can ever go green on Docker build**, and it blocks auto-merge (contract
rule 5) for every PR in this repo, not just ones that touch these pins.

Fix options (ranked):
- (a) Bump the Dockerfile's builder-stage base image to Python 3.12,
  matching what `numpy==2.5.0` requires. Smallest diff if 3.12 is
  otherwise fine for the runtime image too (CI already tests against
  3.11 *and* 3.12, so 3.12 is a supported target).
  ⚠️ **Not a pure-dependency change** — do not let a dependency-bump PR
  smuggle this in per the auto-merge policy's own rule (a code change
  should not ship inside a dependency PR). This needs its own PR.
- (b) Pin `numpy` back to a 3.11-compatible version (`<2.5.0`, e.g.
  `2.4.6`) instead of moving the builder image. Zero-risk to the runtime
  image but reintroduces the exact version mismatch the next time
  dependabot bumps numpy again — treat as a stopgap, not the fix.

Take (a) unless there's a reason the runtime image must stay on 3.11.

> **2026-08-29 — root cause is deeper than the py3.11/3.12 split, and
> option (a) will NOT fix it.** Ran `pip install --dry-run -r
> requirements.txt` on Python 3.12: it is `ResolutionImpossible` on 3.12
> too. The binding conflict is `shap==0.52.0` (requires `numpy>=2`) vs
> `tensorflow-cpu==2.17.0` (requires `numpy<2.0.0` on py≥3.12) — **no
> single `numpy` version can satisfy both**, so neither moving the
> builder image (a) nor pinning `numpy` down (b) is sufficient on its
> own. One of `shap` / `tensorflow-cpu` has to move. Dependabot PR #40
> bumps `tensorflow-cpu` 2.17→2.21, which is the likely unblocker but is
> a real ML-stack version bump on a file whose header says "bump versions
> deliberately via a PR" — owner call. This item and item 7 (Security
> scans red) are the **same** root cause; work them as one PR, not two.

### 3. `src/cv/walk_forward.py` — `evaluate_oof_metrics` degenerate-input guards
_Items 1 and 2 were already merged (PR #23, #24) but never ticked off
here — verified and moved to Done on 2026-08-29. Item 3 as originally
written asked for four branch tests; three of them (empty input,
single-element input, zero-std) already shipped in PR #33 and are in
`tests/test_backtest.py::TestEvaluateOofMetrics`. This is what remained._

The "mismatched lengths" branch was only half-covered:
`test_mismatched_lengths_fees_skipped` exercises `len(signal) > len(rets)`
(signal gets truncated), but `len(y_prob) < len(close) - 1` fell straight
into `strat_rets = signal * rets` and raised
`ValueError: operands could not be broadcast together`. Not reachable from
`run_walk_forward` (there `len(y_prob) == len(close_va)` always), so it is
a robustness gap rather than a live crash — but the function already
defends the other direction, so the asymmetry was almost certainly
unintended.

**Done in PR (this cycle):** alignment step now trims signal *or* returns
to their common prefix; added
`test_fewer_probs_than_returns_does_not_crash` (proven to fail without the
guard). `evaluate_oof_metrics` branch coverage is now complete.

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

- **PR (2026-08-29)** — Item 3 remainder: made the signal/returns
  alignment in `evaluate_oof_metrics` symmetric so `len(y_prob) <
  len(close) - 1` no longer raises a broadcast `ValueError`. Added
  `test_fewer_probs_than_returns_does_not_crash` (proven to fail without
  the guard by reverting it). 83 tests pass (was 82), `ruff` clean,
  `bandit` clean on the changed file. Also reconciled backlog bookkeeping:
  items 1 & 2 verified already-fixed and moved here; the 🔴 Docker item
  re-diagnosed (see its note — real cause is a `shap`/`tensorflow-cpu`
  numpy deadlock, same root as item 7).

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
