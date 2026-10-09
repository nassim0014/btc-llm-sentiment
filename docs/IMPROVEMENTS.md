# Improvements backlog

Re-filed 2026-10-09 after `CLAUDE.md`/`docs/IMPROVEMENTS.md` were deleted in the
"Docs and formatting cleanup" commit (`dd0580d`, 2026-10-02) on `main`. Every
item below was verified against the code as it stands on `main` at that
commit, not carried over from memory or an older copy of this file.

Known and explicitly **out of scope** (already diagnosed in prior runs, do
not re-attempt without a dedicated session): the Dockerfile pins Python 3.11
while `requirements.txt` pins `shap==0.52.0` (needs Python>=3.12), and
`pip-audit --strict` hits a `ResolutionImpossible` across the
numpy/shap/tensorflow-cpu/pandas/scikit-learn/optuna/torch/yfinance pins on
push to `main`. This needs a coordinated full-stack re-pin verified against
the real heavy ML stack.

Also verified **not actual findings**, so they are not listed below:
- The second `evaluate_oof_metrics` shape-mismatch bug flagged in earlier
  session notes (crash when `len(y_prob) < len(rets)`) is no longer present.
  The function already trims both arrays to `common = min(len(signal),
  len(rets))` before the element-wise product, with a comment explaining
  why. Confirmed with a direct repro (`y_prob` length 5, `close` length 11)
  - no crash.
- The `ruff` F401 unused-`pytest`-import failure noted on 2026-08-21 in
  `tests/test_phase1_display.py` is also gone; `ruff check src/ scripts/
  tests/` passes clean on current `main`.

## Items

- [x] **`risk_managed_backtest()`'s all-NaN volatility fallback ignores
  `target_annual_vol`, silently mis-sizing every position on a short
  series (done 2026-10-09)**

  `src/backtest/risk_managed.py::risk_managed_backtest()` computes
  `realized_vol` from a `vol_lookback`-day rolling std of returns. When the
  input series is shorter than `vol_lookback + 1`, every value in that
  rolling window is `NaN`, and the code fell back to the **literal**
  `realized_vol[:] = 0.20` - not the caller's `target_annual_vol` parameter
  (which also defaults to 0.20, which is why this went unnoticed). Any
  caller who passed a different `target_annual_vol` on a short series (a
  live-prediction window still warming up, an early walk-forward fold, a
  user-picked short range in the Backtest Simulator page) got a
  `vol_target_factor` computed against the wrong denominator - e.g.
  `target_annual_vol=0.10` produced `factor = min(0.10/0.20, 1.0) = 0.5`
  (half-sized positions) instead of the intended `1.0` ("nothing to scale
  against yet, trade at full Kelly size"). The sibling implementation,
  `src/backtest/simulator.py::run_backtest()`, already does this correctly
  (`realized_vol_safe = realized_vol.fillna(target_annual_vol)`) - the two
  near-identical backtesters had already drifted apart on this exact point.
  **How to check the fix:** `pytest tests/test_backtest.py -k
  test_vol_fallback_respects_target_annual_vol -v` - the test builds a
  10-day series against the default `vol_lookback=20` (guaranteeing an
  all-NaN rolling window), passes `target_annual_vol=0.10`, and asserts
  `vol_target_factor == 1.0` throughout. Confirmed failing (`got [0.5, 0.5,
  ..., 1.0]`) against the pre-fix code, passing after.

- [ ] **`src/utils/safe_pickle.py::safe_load_pickle()` has zero test
  coverage.**

  This is the generic sibling of `safe_load_bundle()` - same SHA256 +
  restricted-unpickler defense against a tampered pickle running arbitrary
  code on load - meant for "pickle files other than the feature bundle"
  per its own docstring. `safe_load_bundle()` has 7+ tests in
  `tests/test_safe_pickle.py` covering the missing-file, tampered-hash and
  disallowed-class paths; grepping the same file for `safe_load_pickle`
  turns up nothing - it is never imported or called anywhere in `tests/`.
  It is currently correct (manually verified: loads a legitimate pickle,
  rejects one with the wrong hash), but a security-relevant code path with
  no regression test is one bad edit away from silently stopping doing its
  job. **How to check:** `grep -n safe_load_pickle tests/test_safe_pickle.py`
  returns nothing today; after fixing, it should cover at least a
  successful load, a hash-mismatch rejection, and a disallowed-class
  rejection, mirroring the `safe_load_bundle` tests already in that file.

- [ ] **`engineer_features()` swallows every exception, hiding real bugs as
  "no features".**

  `src/live_features.py::engineer_features()` wraps its entire body - news
  aggregation *and* every technical indicator (RSI, MACD, Bollinger Bands,
  all the `llm_sent_lag*` columns) - in a bare `try/except Exception:
  return None`. This is deliberately tested
  (`tests/test_live_features.py::test_returns_none_on_exception`) as a
  "bad input degrades gracefully" guarantee for the Streamlit page, which
  is reasonable for a dashboard. But it means a genuine regression
  introduced anywhere in ~70 lines of indicator math (a typo in a rolling
  window, a wrong shift count) would also just return `None` instead of
  raising - in CI, in a notebook, anywhere this function is called outside
  the one page that expects a graceful `None`. **How to check:** narrow the
  `except` to the specific failure modes the docstring actually anticipates
  (missing columns via `KeyError`, empty input), and add a test that a
  deliberately-broken indicator calculation (e.g. a bad `.shift()` arg)
  raises instead of silently returning `None`.

- [ ] **`src/backtest/simulator.py` and `src/backtest/risk_managed.py`
  duplicate the entire Kelly + vol-targeting + circuit-breaker loop, and
  have already drifted once.**

  The item above (`risk_managed_backtest`'s vol fallback) is proof this
  isn't theoretical: the two files implement the same math with the same
  variable names and the same intent, and one of them silently went stale
  relative to the other. `src/backtest/simulator.py`'s own docstring says
  it is "the same logic as `risk_managed.risk_managed_backtest` but with
  user-adjustable parameters" - i.e. it was always meant to track the other
  file, with no shared function and no test asserting they agree.
  **How to check:** run both backtesters on the same `(prob, close)` input
  with matching parameters (`use_vol_target=True, use_circuit_breaker=True,
  kelly_cap=1.0`) and diff the resulting `positions`/`equity` arrays - they
  should be identical; a future fix to one that isn't mirrored in the
  other will show up as a divergence. Extracting the shared loop into one
  function both call would close this permanently, but is a larger,
  riskier refactor than this backlog item alone.
