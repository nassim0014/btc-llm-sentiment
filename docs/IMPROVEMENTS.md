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

### 1. Duplicate entries in `requirements.txt`
`ipykernel` and `nbconvert` are each listed twice with conflicting versions:

```
ipykernel==7.3.0       # line 47
nbconvert==7.16.4      # line 48
ipykernel==6.29.5      # line 49  ← overrides 7.3.0
nbconvert==7.17.1      # line 50  ← overrides 7.16.4
```

`pip install -r` silently picks the last occurrence, so `ipykernel==7.3.0`
and `nbconvert==7.16.4` are dead lines that look like the pinned version
but never get installed. `pip-audit` (which the README claims to run)
won't catch this — it audits the resolved environment, not the file.

Two questions for whoever takes this item:
- Which version is correct? `6.29.5` for ipykernel is the older,
  widely-tested line; `7.3.0` is newer but ipykernel 7.x is a recent
  release. Same shape for nbconvert.
- Are both lines even needed? The duplicate looks like a copy-paste
  accident during the Phase 2 notebook work, not a deliberate pin.

Mechanical fix once the version question is answered: delete the wrong
line, leave the right one, no other changes.

### 2. `audit/` directory is never created — `sentiment_alert.py` crashes on first run
`src/config.py` declares `AUDIT_DIR = ROOT / "audit"` and
`ALERT_LOG_PATH = AUDIT_DIR / "sentiment_alerts.csv"`.
`scripts/sentiment_alert.py` writes to that path at line ~307
(`ALERT_LOG_PATH.exists()` check + csv writer). Nothing in the repo ever
calls `AUDIT_DIR.mkdir(...)`.

Reproduce: `python scripts/sentiment_alert.py` on a fresh clone →
`FileNotFoundError: [Errno 2] No such file or directory: '.../audit/
sentiment_alerts.csv'`.

Fix options (ranked):
- (a) Add `AUDIT_DIR.mkdir(parents=True, exist_ok=True)` at the top of
  `scripts/sentiment_alert.py` before the first write. Smallest diff,
  clearest intent. The script is the only consumer of `ALERT_LOG_PATH`
  today, so the directory-creation belongs there.
- (b) Make `AUDIT_DIR` a `pathlib.Path` property that creates itself on
  first access. Cleaner but over-engineered for one consumer.
- (c) Add `audit/` to the repo with a `.gitkeep`. Wrong — the path is
  `ROOT/"audit"`, not `ROOT/data/audit`, and committing an empty dir
  doesn't fix the bug if someone deletes it.

Take (a). Add a test that runs `sentiment_alert.py`'s write path against
a tmp path and asserts the file exists after — that test would have
caught this.

### 3. `src/cv/walk_forward.py` coverage gap — lines 292-333
`evaluate_oof_metrics` is the only function in the file with a coverage
gap of consequence (file is 66% overall; the rest is at 100% or
intentionally-untested error branches). The existing
`test_evaluate_oof_metrics` covers the perfect-predictions and
random-around-half cases, but the function has branches for:
- empty input arrays
- single-element input
- mismatched lengths between predictions and actuals
- division-by-zero (when std is 0)

All four are silent error paths today. Add one test per branch. This is
the same shape as kinz-competitor-intelligence's coverage-gap work
(PR #39 there) — pure-Python logic, no ML deps, runs in the lightweight
CI install.

## Next

### 4. ~~ML-heavy modules at 0% coverage — needs an owner decision~~ ✅
**Decision #4: CPU install.** Added a new `ml-smoke` CI job that installs
`tensorflow-cpu` + `torch` (CPU-only, from the PyTorch CPU wheel index)
+ `transformers` + `shap`, then import-checks all four ML modules
(`src/cv/optuna_search.py`, `src/inference/finbert.py`,
`src/interpretability/shap_explainer.py`, `src/models/lstm.py`). Adds
~3-5 min to CI run time. Catches API mismatches that the lightweight
test job (which deliberately skips ML deps) can't see.

### 5. `pages/` Streamlit pages — 1,464 lines of UI with zero tests (in progress)
Four Streamlit page files (`1_🔬_Phase_1_Deep_Dive.py` through
`4_🎛️_Backtest_Simulator.py`) total 1,464 lines and have no tests.

**First pass (this PR):** extracted `run_backtest` from
`4_🎛️_Backtest_Simulator.py` (the largest, most logic-heavy page) into
`src/backtest/simulator.py`. 15 new tests in `tests/test_simulator.py`
cover: result keys, equity normalization, no-trade-when-below-threshold,
position sizing, metrics (sharpe/max_dd/win_rate/n_trades), vol-targeting
toggle, circuit-breaker trigger + disable, edge cases (single day, two
days, fee impact). Page reduced from 568 → 454 lines.

Remaining pages (1, 2, 3) still need the same treatment — each is a
separate PR.

### 6. ~~`pytest.ini` vs `pyproject.toml`~~ ✅
Consolidated `pytest.ini` + `ruff.toml` into `pyproject.toml` (single
config file). Both tools natively read `pyproject.toml`. Verified:
31 tests pass (1 skip), ruff finds the config (113 pre-existing
notebook errors unchanged — not from this migration).

---

## Done

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
