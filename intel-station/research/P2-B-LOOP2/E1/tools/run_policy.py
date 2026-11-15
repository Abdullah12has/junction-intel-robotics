"""Driver for the loop policy command line (loop/policy_run.py). Run it under python3 -I:

  python3 -I run_policy.py APP_ROOT scenarios
  python3 -I run_policy.py APP_ROOT dry SCENARIO LOGS_ROOT

APP_ROOT is the code copy: the folder that holds app/loop. The driver puts its app folder first on sys.path.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    app = Path(sys.argv[1]).resolve() / "app"
    sys.path.insert(0, str(app))
    from loop.policy_run import main as policy_main

    return policy_main(sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
