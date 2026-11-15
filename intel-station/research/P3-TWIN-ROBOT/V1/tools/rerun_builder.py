#!/usr/bin/env python3
"""V1 re-run of the builder's gate scripts (E1/tools/g1_check.py or gripper_check.py), code unchanged. Only the output
folder is redirected to V1/outputs/rerun_<which>, so the builder's E1 files are not overwritten. Inputs are read from
the builder's E1 folder and the R/sim/twin folders as the builder wrote them.
Usage: rerun_builder.py g1 | gripper"""
import importlib.util
import os
import sys

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-ROBOT/E1"
V1_OUT = f"{R}/work/P3-TWIN-ROBOT/V1/outputs"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    if which == "g1":
        mod = load(f"{E1}/tools/g1_check.py", "builder_g1_check")
    elif which == "gripper":
        mod = load(f"{E1}/tools/gripper_check.py", "builder_gripper_check")
    else:
        raise SystemExit("usage: rerun_builder.py g1 | gripper")
    out = f"{V1_OUT}/rerun_{which}"
    os.makedirs(out, exist_ok=True)
    mod.E1 = out
    mod.main()


if __name__ == "__main__":
    main()
