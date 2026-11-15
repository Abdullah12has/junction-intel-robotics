"""Runs the loop policy unit tests. Usage under python3 -I: python3 -I run_tests.py APP_ROOT"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    sys.path.insert(0, str(root / "app"))
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern="test_*.py", top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
