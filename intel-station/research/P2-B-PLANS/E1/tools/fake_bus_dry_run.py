"""Fake-bus dry run of run_sequence on placeholder poses. No serial port is opened.

Run from the E1 folder (paths below are relative to this file):
  P2R6_DRIVER_SRC=<core src>:<plugin src> <motion-env python> -I -B tools/fake_bus_dry_run.py
The fake Feetech SDK from tests/test_motion.py is the only scservo_sdk this process can load.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E1 = HERE.parent
POSES = E1 / "out" / "poses.synthetic.yaml"  # placeholder: true, built from the synthetic calibration fixture

sys.path.insert(0, str(E1 / "tests"))
import test_motion as tm  # noqa: E402  (installs the fake scservo_sdk before any driver import)
import run_sequence  # noqa: E402
import so101_motion  # noqa: E402

assert sys.modules["scservo_sdk"] is tm._fake_sdk, "the real scservo_sdk would be active"
assert so101_motion.is_placeholder(POSES), "the poses file must keep placeholder: true"

home = so101_motion.load_poses(POSES)["home"]
start = [f"{float(value):.2f}" for value in home]
print("poses sha256", hashlib.sha256(POSES.read_bytes()).hexdigest())
print("start (home)", start)

cases = [
    ("list", ["--poses", str(POSES), "--list"], 0),
    ("dry spot_to_blue", ["--poses", str(POSES), "--sequence", "spot_to_blue", "--start", *start], 0),
    ("dry spot_to_green", ["--poses", str(POSES), "--sequence", "spot_to_green", "--start", *start], 0),
    (
        "confirm spot_to_blue",
        ["--poses", str(POSES), "--sequence", "spot_to_blue", "--start", *start, "--confirm-motion"],
        4,
    ),
    (
        "confirm spot_to_green",
        ["--poses", str(POSES), "--sequence", "spot_to_green", "--start", *start, "--confirm-motion"],
        4,
    ),
]

failures = 0
for label, argv, expected in cases:
    bus = tm.install_bus()
    code = run_sequence.main(argv)
    ok = code == expected and not bus.events and not bus.open_ports
    failures += 0 if ok else 1
    print(
        f"CASE {label}: exit={code} expected={expected} bus_events={len(bus.events)} "
        f"open_ports={sorted(bus.open_ports)} {'OK' if ok else 'FAIL'}"
    )

print("SUMMARY", "PASS" if failures == 0 else f"FAIL ({failures})")
raise SystemExit(0 if failures == 0 else 1)
