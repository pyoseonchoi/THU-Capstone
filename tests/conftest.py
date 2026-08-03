"""Shared test fixtures and conftest for FULLSCAN-QA tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the app package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))
