"""Root-level pytest configuration — shared fixtures across all test modules.

Adds the repo root to sys.path so `src/` is importable without a
package install, and provides a seeded random fixture for reproducible
test data.
"""
import sys
from pathlib import Path

# Ensure src/ is importable from any test directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest


@pytest.fixture
def seeded_random():
    """Return a seeded numpy random generator for reproducible tests."""
    return np.random.default_rng(seed=42)
