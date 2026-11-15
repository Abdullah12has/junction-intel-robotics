"""Run-time roots of the sorter: the real cell by default, a simulated cell when JUNCTION_RUN_ROOT is set."""
import os

R = "/home/intel/intel-junction-2026"
SIM_ROOT = os.environ.get("JUNCTION_RUN_ROOT") or None
RUN_ROOT = SIM_ROOT or f"{R}/run"  # live frames, flags, motion server state, heartbeat
LOG_ROOT = os.path.join(SIM_ROOT, "logs") if SIM_ROOT else f"{R}/logs"  # episodes, monitor logs, e-stop images
