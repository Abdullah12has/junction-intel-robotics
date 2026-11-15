"""Print the run-time paths the loop modules resolve, for the real cell (no env) and a simulated root."""
import os
import sys

sys.path.insert(0, "/home/intel/intel-junction-2026/app/tools")
import grasp_auto as ga  # noqa: E402
import handdet as hd  # noqa: E402
import sort_any as sa  # noqa: E402
import sort_loop as sl  # noqa: E402

print("JUNCTION_RUN_ROOT =", os.environ.get("JUNCTION_RUN_ROOT"))
for k, v in (("ga.RUN_ROOT", ga.RUN_ROOT), ("ga.LOG_ROOT", ga.LOG_ROOT), ("ga.SD", ga.SD), ("ga.SIM_ROOT", ga.SIM_ROOT),
             ("hd.HEARTBEAT", hd.HEARTBEAT), ("sa.SCAN_PAIRS", sa.SCAN_PAIRS), ("sl.RUNTIME_DIR", sl.RUNTIME_DIR),
             ("sl.HAND_LOG", sl.HAND_LOG)):
    print(f"  {k:16s} {v}")
