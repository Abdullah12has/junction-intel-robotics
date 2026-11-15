"""Self-check (R12): re-read the dataset parquet and compare every waypoint in the two YAML files to it."""

import json
import re
import sys
from pathlib import Path

import numpy as np

TOOLS = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1/tools")
sys.path.insert(0, str(TOOLS))
import analyze_releases as ar  # noqa: E402

E1 = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1")
GM = Path("/home/intel/intel-junction-2026/app/motion/gm")
POSE_RE = re.compile(r"^  (\w+_w\d\d): \[(.*)\]$")
STEP_RE = re.compile(r"^    - \{pose: (\w+), speed: ([0-9.]+)(?:, dwell_s: ([0-9.]+))?\}$")


def parse(path: Path):
    poses, steps, seqs = {}, {}, {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = POSE_RE.match(line)
        if m:
            poses[m.group(1)] = [float(x) for x in m.group(2).split(",")]
            continue
        if line.startswith("  ") and line.endswith(":") and not line.startswith("   "):
            current = line.strip()[:-1]
            seqs[current] = []
            continue
        m = STEP_RE.match(line)
        if m and current:
            seqs[current].append((m.group(1), float(m.group(2)), float(m.group(3) or 0.0)))
    return poses, seqs


def main() -> None:
    seg = json.loads((E1 / "segments.json").read_text(encoding="utf-8"))
    data = ar.read_all("data/chunk-000/*.parquet").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    ep_ids = data["episode_index"].to_numpy()
    episodes = {"blue": 11, "green": 9}
    report = {}
    for name, ep in episodes.items():
        s = states[ep_ids == ep]
        poses, seqs = parse(GM / f"{name}_plate.yaml")
        frames = seg[name]["waypoint_frames"]
        cap = seg[name]["speed_cap_units_s"]
        max_diff = 0.0
        for j, fr in enumerate(frames):
            vals = np.array(poses[f"{name}_w{j:02d}"])
            ref = s[fr, :5]
            max_diff = max(max_diff, float(np.max(np.abs(vals[:5] - ref))))
            assert vals[5] == 25.0, (name, j)
        n = len(frames)
        fwd = [x[0] for x in seqs[f"{name}_plate_to_home"]]
        rev = seqs[f"{name}_home_to_plate"]
        ok_fwd = fwd == [f"{name}_w{j:02d}" for j in range(n)]
        ok_rev = [x[0] for x in rev] == [f"{name}_w{j:02d}" for j in reversed(range(n))]
        ok_dwell = rev[-1][2] == 1.0 and all(x[2] == 0.0 for x in rev[:-1])
        ok_speed = all(abs(x[1] - cap) < 1e-9 for x in rev + seqs[f"{name}_plate_to_home"])
        last_frame_is_end = frames[-1] == seg[name]["end_frame"]
        first_frame_is_release = frames[0] == seg[name]["release_frame"]
        report[name] = {
            "episode": ep,
            "n_waypoints": n,
            "max_abs_diff_vs_dataset_units": max_diff,
            "fwd_order_ok": ok_fwd,
            "rev_order_ok": ok_rev,
            "dwell_only_last_step_ok": ok_dwell,
            "speed_cap_ok": ok_speed,
            "first_is_release_last_is_end": bool(last_frame_is_end and first_frame_is_release),
            "body_within_100": bool(all(abs(v) <= 100.0 for p in poses.values() for v in p[:5])),
        }
        print(name, report[name])
    (E1 / "verify_outputs.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
