"""Tests for the safe pickle loader (src/utils/safe_pickle.py).

Verifies:
  - safe_load_bundle() loads the committed bundle successfully
  - SHA256 mismatch is rejected
  - Restricted unpickler blocks malicious pickles (os.system, subprocess)
  - Missing bundle raises a clear error
  - Centralized config exposes the correct constants
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

# Ensure repo root is on sys.path so `src.*` imports work
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    ALERT_LOG_PATH,
    ALERT_THRESHOLDS,
    BUNDLE_SHA256,
    MODEL_SHA256,
    NEWS_URL,
    REQUEST_TIMEOUT_SECONDS,
    USER_AGENT,
    find_bundle,
    find_model,
)
from src.utils.safe_pickle import SafePickleError, safe_load_bundle

# ────────────────────────────────────────────────────────────
# 1. Centralized config
# ────────────────────────────────────────────────────────────


def test_config_exposes_required_constants():
    """All config constants that consumers depend on must be present."""
    assert NEWS_URL.startswith("https://")
    assert "cryptonews.csv" in NEWS_URL
    assert REQUEST_TIMEOUT_SECONDS > 0
    assert USER_AGENT
    assert BUNDLE_SHA256  # non-empty
    assert MODEL_SHA256  # non-empty
    assert ALERT_THRESHOLDS["very_bullish"] > ALERT_THRESHOLDS["bullish"]
    assert ALERT_THRESHOLDS["bearish"] > ALERT_THRESHOLDS["very_bearish"]
    assert "sentiment_alerts.csv" in str(ALERT_LOG_PATH)


def test_find_model_returns_path():
    """find_model() must locate the committed .keras file."""
    p = find_model()
    assert p is not None, "Model file not found in any configured location"
    assert p.exists()
    assert p.suffix == ".keras"


def test_find_bundle_returns_path():
    """find_bundle() must locate the committed .pkl file."""
    p = find_bundle()
    assert p is not None, "Feature bundle not found in any configured location"
    assert p.exists()
    assert p.suffix == ".pkl"


# ────────────────────────────────────────────────────────────
# 2. Safe bundle loading
# ────────────────────────────────────────────────────────────


def test_safe_load_bundle_succeeds():
    """The committed bundle must load successfully with the pinned hash."""
    bundle = safe_load_bundle()
    assert isinstance(bundle, dict)
    assert "train_x" in bundle
    assert "train_y" in bundle
    assert "test_x" in bundle
    assert "scaler" in bundle
    assert "feature_cols" in bundle
    # train_x must be a numpy array
    assert hasattr(bundle["train_x"], "shape")
    assert bundle["train_x"].ndim == 3  # (samples, timesteps, features)


def test_safe_load_bundle_rejects_tampered_hash():
    """A wrong expected_sha256 must cause SafePickleError."""
    with pytest.raises(SafePickleError) as exc_info:
        safe_load_bundle(expected_sha256="deadbeef" * 8)
    assert "SHA256 mismatch" in str(exc_info.value)


def test_safe_load_bundle_rejects_missing_file(tmp_path):
    """A missing bundle file must raise SafePickleError, not FileNotFoundError."""
    nonexistent = tmp_path / "does_not_exist.pkl"
    with pytest.raises(SafePickleError) as exc_info:
        safe_load_bundle(path=nonexistent, skip_integrity_check=True)
    msg = str(exc_info.value).lower()
    assert "not found" in msg or "missing" in msg or "no such file" in msg


# ────────────────────────────────────────────────────────────
# 3. Restricted unpickler — malicious pickle rejection
# ────────────────────────────────────────────────────────────


def test_restricted_unpickler_blocks_os_system(tmp_path):
    """A pickle that tries to import os.system must be rejected."""
    # Construct a malicious pickle that would call os.system("echo pwned")
    import os

    class _Malicious:
        def __reduce__(self):
            return (os.system, ("echo pwned",))

    malicious_path = tmp_path / "malicious.pkl"
    with malicious_path.open("wb") as f:
        pickle.dump(_Malicious(), f)

    with pytest.raises(SafePickleError) as exc_info:
        safe_load_bundle(path=malicious_path, skip_integrity_check=True)
    assert "Blocked attempt to import" in str(exc_info.value)


def test_restricted_unpickler_blocks_subprocess(tmp_path):
    """A pickle that tries to import subprocess.Popen must be rejected."""
    import subprocess

    class _Malicious:
        def __reduce__(self):
            return (subprocess.Popen, (["echo", "pwned"],))

    malicious_path = tmp_path / "malicious_subproc.pkl"
    with malicious_path.open("wb") as f:
        pickle.dump(_Malicious(), f)

    with pytest.raises(SafePickleError) as exc_info:
        safe_load_bundle(path=malicious_path, skip_integrity_check=True)
    assert "Blocked attempt to import" in str(exc_info.value)


def test_restricted_unpickler_blocks_eval(tmp_path):
    """A pickle that tries to import builtins.eval must be rejected."""
    class _Malicious:
        def __reduce__(self):
            return (eval, ("__import__('os').system('echo pwned')",))

    malicious_path = tmp_path / "malicious_eval.pkl"
    with malicious_path.open("wb") as f:
        pickle.dump(_Malicious(), f)

    with pytest.raises(SafePickleError) as exc_info:
        safe_load_bundle(path=malicious_path, skip_integrity_check=True)
    assert "Blocked attempt to import" in str(exc_info.value)


def test_restricted_unpickler_allows_legitimate_bundle():
    """The legitimate bundle (dict + numpy + StandardScaler) must load."""
    bundle = safe_load_bundle()
    # Verify the scaler is the expected type
    from sklearn.preprocessing import StandardScaler

    assert isinstance(bundle["scaler"], StandardScaler)
    # Verify numpy arrays
    import numpy as np

    assert isinstance(bundle["train_x"], np.ndarray)
    assert isinstance(bundle["train_y"], np.ndarray)


# ────────────────────────────────────────────────────────────
# 4. SHA256 verification helper
# ────────────────────────────────────────────────────────────


def test_file_sha256_matches_pinned_value():
    """The on-disk bundle's SHA256 must match BUNDLE_SHA256 in config."""
    from src.utils.safe_pickle import file_sha256

    p = find_bundle()
    assert p is not None
    actual = file_sha256(p)
    assert actual == BUNDLE_SHA256, (
        f"Bundle SHA256 mismatch!\n"
        f"  config says: {BUNDLE_SHA256}\n"
        f"  on-disk is:  {actual}\n"
        f"If you just retrained, update BUNDLE_SHA256 in src/config.py."
    )


def test_model_sha256_matches_pinned_value():
    """The on-disk .keras model's SHA256 must match MODEL_SHA256 in config."""
    from src.utils.safe_pickle import file_sha256

    p = find_model()
    assert p is not None
    actual = file_sha256(p)
    assert actual == MODEL_SHA256, (
        f"Model SHA256 mismatch!\n"
        f"  config says: {MODEL_SHA256}\n"
        f"  on-disk is:  {actual}\n"
        f"If you just retrained, update MODEL_SHA256 in src/config.py."
    )
