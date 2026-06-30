<!--
Thanks for opening a PR! See CONTRIBUTING.md for full guidelines.
-->

## Summary

<!-- One paragraph: what & why. -->

## Type of change

- [ ] 🐛 Bug fix (non-breaking)
- [ ] ✨ New feature (non-breaking)
- [ ] 💥 Breaking change
- [ ] 📚 Documentation only
- [ ] 🔧 Refactor / chore
- [ ] 🛡️ Security hardening
- [ ] 🤖 ML model retrain (see checklist below)

## Related issue

<!-- "Closes #123" or "Refs #123". -->

## Changes

-
-
-

## Security considerations

<!-- If this PR touches safe_pickle.py, config.py, CI, Docker, or any
     security-sensitive code, explain what was done. If not applicable, write "N/A". -->

## How to test

```bash
# Example:
pytest tests/ -v
python scripts/run_pipeline.py --quick
```

## Checklist

- [ ] My code follows the project style (ruff passes)
- [ ] I added tests for any new behavior
- [ ] All new and existing tests pass locally (`pytest tests/ -v`)
- [ ] I updated documentation (README, SECURITY.md, MODEL_CARD.md) if behavior changed
- [ ] I did NOT commit any secrets, `.env` files, or private keys
- [ ] PR title follows Conventional Commits (`feat(lstm):`, `fix(backtest):`, etc.)

### ML model retrain checklist (only if you retrained and committed a new model)

- [ ] I updated `BUNDLE_SHA256` in `src/config.py` with the new hash of `features_for_lstm.pkl`
- [ ] I updated `MODEL_SHA256` in `src/config.py` with the new hash of `best_optuna_model.keras`
- [ ] I verified `safe_load_bundle()` succeeds with the new hashes
- [ ] I updated the Model Card if model behavior or metrics changed
