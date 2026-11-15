"""Test set-up: put the app folder (which contains pick_v2) on sys.path.

Only in a sandbox without the motion environment, loguru, yaml or the physicalai SO-101 plugin are
stand-in modules installed, so so101_motion can be imported for the fake-bus tests. On the laptop
the real packages from R/envs/motion-env are used and nothing is stubbed.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import loguru  # noqa: F401
except ImportError:
    class _Logger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    _loguru = types.ModuleType("loguru")
    _loguru.logger = _Logger()
    sys.modules["loguru"] = _loguru

try:
    import yaml  # noqa: F401
except ImportError:
    sys.modules["yaml"] = types.ModuleType("yaml")

try:
    import physicalai_so101_plugin.constants  # noqa: F401
except ImportError:
    _pkg = types.ModuleType("physicalai_so101_plugin")
    _pkg.__path__ = []
    _const = types.ModuleType("physicalai_so101_plugin.constants")
    _const.SO101_JOINT_ORDER = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    sys.modules["physicalai_so101_plugin"] = _pkg
    sys.modules["physicalai_so101_plugin.constants"] = _const
