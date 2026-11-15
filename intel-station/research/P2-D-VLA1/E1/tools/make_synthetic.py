import csv
import json
import math
import os
import random
import sys

ROOT = "/home/intel/intel-junction-2026"
BASE = os.path.join(ROOT, "work/P2-D-VLA1/E1/synthetic_runs")
MKV_START = 1794675831.767
DUR = 20.0
REST = [-6.6, -98.1, 99.4, 54.3, -53.1, 20.0]
AMP = [20.0, 15.0, 10.0, 8.0, 5.0, 10.0]


def write_run(run_name, offsets):
    run_dir = os.path.join(BASE, run_name)
    for k, off in enumerate(offsets, start=1):
        t0 = MKV_START + off
        ep_dir = os.path.join(run_dir, "ep_%04d" % k)
        os.makedirs(ep_dir, exist_ok=True)
        rnd = random.Random(7919 * k + len(run_name))
        n = int(round(DUR * 30))
        with open(os.path.join(ep_dir, "joints.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["t"] + ["goal%d" % j for j in range(6)] + ["pos%d" % j for j in range(6)])
            for i in range(n):
                t = t0 + i / 30.0 + rnd.uniform(-0.001, 0.001)
                ph = 2 * math.pi * (i / 30.0) / 6.0
                pos = [REST[j] + AMP[j] * math.sin(ph + j) for j in range(6)]
                goal = [p + 0.4 for p in pos]
                w.writerow(["%.6f" % t] + ["%.4f" % g for g in goal] + ["%.4f" % p for p in pos])
        meta = {
            "episode": k,
            "run": run_name,
            "t_start": t0,
            "t_end": t0 + n / 30.0,
            "task": "move the red brick to the green plate",
            "plate": "green",
            "colour": "red",
            "shape": "2x2",
            "label": "GOOD",
            "label_source": "SYNTHETIC",
            "success": True,
            "failure": None,
            "synthetic": True,
            "cams": {
                "top": "logs/monitor/top_20261114T170351Z.mkv",
                "close": "logs/monitor/close_20261114T170351Z.mkv",
                "wrist_dir": "data/wrist/raw",
            },
            "note": "SYNTHETIC: joints are fake, video is a 20 s window of existing MKVs. Never import.",
        }
        with open(os.path.join(ep_dir, "episode.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
    return len(offsets)


def main():
    a = write_run("SYNTH-A-3EP", [90.0, 140.0, 190.0])
    b = write_run("SYNTH-B-150EP", [30.0 + 3.8 * k for k in range(150)])
    print("wrote", a, "episodes to SYNTH-A-3EP and", b, "to SYNTH-B-150EP under", BASE)


if __name__ == "__main__":
    sys.exit(main())
