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

### 7. `ruff check` fails on `main` right now — unused `pytest` import
`tests/test_phase1_display.py:5` imports `pytest` but never uses it
(`F401`). Confirmed via `git stash` against a clean `main` checkout —
this is not caused by any pending change, it's already broken.
One-line, `ruff --fix`-able. Small enough that whoever takes the next
item here should just delete the import as a 30-second prelude rather
than dedicating a full cycle to it — but flagging it explicitly since
"lint is currently red on main" is exactly the kind of thing a backlog
should never let sit silently.

### 8. `evaluate_oof_metrics` crashes when predictions are shorter than the close-price window
Separate from the empty-input bug fixed this cycle (see *Done*): when
`len(y_prob) < len(rets)` — i.e. `close` has more points than
`len(y_prob) + 1` — `signal * rets` at
`src/cv/walk_forward.py:179` raises
`ValueError: operands could not be broadcast together with shapes
(N,) (M,)`. The existing truncation (`if len(signal) > len(rets):
signal = signal[:len(rets)]`) only handles the signal-longer case; there
is no symmetric truncation of `rets` (or padding of `signal`) for the
signal-shorter case.

Reproduce:
```python
evaluate_oof_metrics(
    y_true=np.array([1, 0, 1]), y_prob=np.array([0.9, 0.1, 0.8]),
    close=np.linspace(100, 110, 10),  # rets has 9 elements, signal has 3
)
# ValueError: operands could not be broadcast together with shapes (3,) (9,)
```
Whether this can happen from `run_walk_forward`'s own fold slicing is
unverified — `X_va`/`close_va` are sliced from the same `val_idx` so
they should stay aligned in the normal pipeline — but the function is
also called directly from `optuna_search.py` and is public API, so a
caller passing mismatched arrays currently gets an opaque numpy error
instead of a clear one. Fix should mirror the existing truncation
direction (truncate `rets` to `len(signal)` too) plus a test that
reproduces this exact shape mismatch first, per the loop's own
verification rule.

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

- **PR #23** — Removed the duplicate `ipykernel`/`nbconvert` pins in
  `requirements.txt` (item 1). Merged 2026-08-19.
- **PR #24** — Added `AUDIT_DIR.mkdir(parents=True, exist_ok=True)`
  before `sentiment_alert.py`'s first write (item 2, option (a) as
  recommended). Merged, predates this entry — opened as an "open-loop
  experiment" rather than through this backlog, which is why it wasn't
  ticked off here until now. Bookkeeping fix, not new work.
- **Item 3, closed-loop cycle 2026-08-21** — The four branches item 3
  asked for (empty input, single-element, mismatched lengths, zero-std)
  *already had tests* in `tests/test_backtest.py::TestEvaluateOofMetrics`
  by the time this cycle picked it up — same story as items 1 and 2,
  written by an earlier pass but never ticked off here. Investigating
  why anyway turned up a real, separate bug the existing tests were
  masking: `test_empty_input_returns_zero_metrics` was **self-skipping**
  via `pytest.skip()` whenever sklearn raised on empty input — which it
  does, every time, on the installed sklearn (1.9.0). So the "empty
  input" branch had a test in name only; production `evaluate_oof_metrics`
  actually crashed with `ValueError: Found empty input array...` on
  `y_true=[]`, never reaching its own `n_days == 0` graceful-degradation
  path. Fixed by returning the zeroed-metrics dict immediately when
  `len(y_true) == 0`, before any sklearn call. Verified by reverting the
  fix and confirming the (now non-skipping) test fails with that exact
  `ValueError`, then restoring it and confirming all 82 tests pass.
  `walk_forward.py` coverage moved 68% → 69%; the remaining gap is
  mostly `run_walk_forward`'s Keras-training loop (needs the ML-heavy
  install) plus two dataclass `__repr__`/`summary` methods — neither is
  the "silent error path" item 3 was actually chasing, so not picked up
  here. One new bug found in the same function while verifying this fix
  is filed separately as item 8 (different failure mode, different fix
  location, out of scope for this PR).
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
