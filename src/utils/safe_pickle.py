"""Safe loaders for committed model artifacts.

## Why this module exists

The pipeline commits a `features_for_lstm.pkl` file to the repo so the
Streamlit "Live Predictions" page can run without retraining. **Pickle
deserialization is arbitrary code execution** — if an attacker gains
write access to the repo (compromised PAT, merged malicious PR), they
can ship a tampered `.pkl` that runs any code on every user who opens
the Live Predictions page on Streamlit Cloud.

This module mitigates that risk with two layers:

1. **SHA256 integrity verification** — the expected hash is stored in
   `src/config.py::BUNDLE_SHA256`. If the on-disk file's hash does not
   match, we refuse to load. Updating the hash requires a code change
   that goes through PR review.

2. **Restricted unpickler** — we override `pickle.Unpickler.find_class`
   to only allow a strict allowlist of module/class pairs that the
   legitimate bundle uses (numpy arrays, sklearn StandardScaler, plain
   dict/list/str/int/float). Any pickle that tries to import anything
   else is rejected.

Together these make the pickle load safe-ish. The truly correct fix is
to retrain and save the scaler in a non-executable format (e.g.
safetensors for the arrays + JSON for the scaler params), but that's a
larger change deferred to the roadmap.

## Usage

    from src.utils.safe_pickle import safe_load_bundle
    bundle = safe_load_bundle()  # raises SafePickleError on tamper
"""
from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any

from src.config import BUNDLE_SHA256, find_bundle

logger = logging.getLogger(__name__)


class SafePickleError(RuntimeError):
    """Raised when a pickle file fails integrity or allowlist checks."""


# ────────────────────────────────────────────────────────────
# SHA256 verification
# ────────────────────────────────────────────────────────────


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA256 hex digest of a file (streaming, 1MB chunks)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    """Raise SafePickleError if the file's SHA256 does not match `expected`."""
    actual = file_sha256(path)
    if actual != expected:
        raise SafePickleError(
            f"SHA256 mismatch for {path}:\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"Refusing to load — file may have been tampered with. "
            f"If you just retrained the model, update BUNDLE_SHA256 in "
            f"src/config.py with the new hash."
        )


# ────────────────────────────────────────────────────────────
# Restricted unpickler
# ────────────────────────────────────────────────────────────

# Allowlist of (module, name) pairs that the legitimate bundle uses.
# Anything outside this set is rejected.
_ALLOWED_PICKLE_CLASSES = frozenset(
    {
        # Built-in collections
        ("builtins", "dict"),
        ("builtins", "list"),
        ("builtins", "tuple"),
        ("builtins", "set"),
        ("builtins", "frozenset"),
        ("builtins", "str"),
        ("builtins", "int"),
        ("builtins", "float"),
        ("builtins", "bool"),
        ("builtins", "complex"),
        ("builtins", "bytes"),
        ("builtins", "bytearray"),
        ("builtins", "NoneType"),
        ("collections", "OrderedDict"),
        ("collections", "defaultdict"),
        # numpy — arrays, dtypes, scalars (both old and new module paths)
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy", "float64"),
        ("numpy", "float32"),
        ("numpy", "int64"),
        ("numpy", "int32"),
        ("numpy", "float16"),
        ("numpy", "int16"),
        ("numpy", "int8"),
        ("numpy", "uint8"),
        ("numpy", "uint16"),
        ("numpy", "uint32"),
        ("numpy", "uint64"),
        # numpy 2.x renamed internal modules — both paths must be allowed
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        # sklearn — only StandardScaler (the only estimator in the bundle)
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("sklearn.preprocessing.data", "StandardScaler"),  # older sklearn path
        # pandas — Timestamps appear in date arrays
        ("pandas", "Timestamp"),
        ("pandas._libs.tslibs", "Timestamp"),
        ("pandas._libs.tslibs.timestamps", "Timestamp"),
    }
)


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that only allows classes in `_ALLOWED_PICKLE_CLASSES`."""

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in _ALLOWED_PICKLE_CLASSES:
            raise SafePickleError(
                f"Blocked attempt to import {module}.{name} during pickle load. "
                f"This pickle is not a legitimate model bundle. "
                f"If you just retrained with a new estimator type, add it to "
                f"_ALLOWED_PICKLE_CLASSES in src/utils/safe_pickle.py."
            )
        return super().find_class(module, name)


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────


def safe_load_bundle(
    path: Path | None = None,
    expected_sha256: str | None = BUNDLE_SHA256,
    skip_integrity_check: bool = False,
) -> dict:
    """Load the feature bundle with SHA256 + restricted-unpickler checks.

    Parameters
    ----------
    path : Path, optional
        Path to the .pkl file. If None, searches BUNDLE_PATHS.
    expected_sha256 : str, optional
        Expected SHA256 hex digest. Defaults to BUNDLE_SHA256 from config.
    skip_integrity_check : bool
        If True, skip the SHA256 check (e.g. for tests with a fresh bundle).
        Default False — never skip in production.

    Returns
    -------
    dict
        The bundle contents (train_x, train_y, scaler, etc.).

    Raises
    ------
    SafePickleError
        If the file is missing, the SHA256 doesn't match, or the pickle
        tries to import a class outside the allowlist.
    """
    if path is None:
        path = find_bundle()
    if path is None:
        raise SafePickleError(
            "Feature bundle not found in any of the configured locations. "
            "Run `python scripts/generate_interim_features.py` first."
        )

    # Layer 1: integrity check
    if not skip_integrity_check and expected_sha256:
        verify_sha256(path, expected_sha256)
        logger.info("Bundle integrity verified: %s (sha256=%s...)", path, expected_sha256[:12])

    # Layer 2: restricted unpickler
    try:
        with path.open("rb") as f:
            unpickler = _RestrictedUnpickler(f)
            bundle = unpickler.load()
    except SafePickleError:
        raise
    except Exception as e:
        raise SafePickleError(f"Failed to load bundle from {path}: {e}") from e

    if not isinstance(bundle, dict):
        raise SafePickleError(
            f"Expected bundle to be a dict, got {type(bundle).__name__}. "
            f"The pickle format may have changed."
        )

    return bundle


def safe_load_pickle(
    path: Path,
    expected_sha256: str,
    *,
    skip_integrity_check: bool = False,
) -> Any:
    """Generic safe pickle loader for arbitrary committed .pkl files.

    Use this for pickle files other than the feature bundle. Always
    provide an explicit expected_sha256.
    """
    if not skip_integrity_check:
        verify_sha256(path, expected_sha256)
    try:
        with path.open("rb") as f:
            return _RestrictedUnpickler(f).load()
    except SafePickleError:
        raise
    except Exception as e:
        raise SafePickleError(f"Failed to load {path}: {e}") from e
