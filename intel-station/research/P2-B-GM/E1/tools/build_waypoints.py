"""Build plate_to_home / home_to_plate waypoint sequences from recorded episodes (dataset read-only).

Writes R/app/motion/gm/<plate>_plate.yaml and copies them plus segments.json into E1/.
No serial port, no camera, no motion.
"""

import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1/tools")
sys.path.insert(0, str(TOOLS))
import analyze_releases as ar  # noqa: E402

E1 = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1")
GM = Path("/home/intel/intel-junction-2026/app/motion/gm")
GRIP_OPEN = 25.0
SPEED_DEFAULT = 25.0
STEP_FRAMES = 0.25 * ar.FPS
JOINTS = "[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]"


def strict_home(s: np.ndarray, r: int) -> int | None:
    q = s[:, :5]
    n = len(s)
    v = np.zeros(n)
    v[:-1] = np.linalg.norm(np.diff(q, axis=0), axis=1) * ar.FPS
    for k in range(r, n - ar.STILL_FRAMES + 1):
        if abs(q[k, 1] - ar.HOME_LIFT) <= ar.HOME_TOL and bool(np.all(v[k : k + ar.STILL_FRAMES] < ar.STILL_SPEED)):
            return k
    return None


def waypoint_index(r: int, end: int) -> list[int]:
    idx = []
    k = 0
    while True:
        i = r + int(np.floor(STEP_FRAMES * k + 0.5))
        if i >= end:
            break
        idx.append(i)
        k += 1
    idx.append(end)
    return idx


def fmt_pose(vals: list[float]) -> str:
    return "[" + ", ".join(f"{x:.4f}" for x in vals) + "]"


def build_plate(name: str, s: np.ndarray, r: int, end: int, note: str) -> dict:
    q = s[:, :5]
    idx = waypoint_index(r, end)
    seg = q[r : end + 1]
    peak = float(np.max(np.abs(np.diff(seg, axis=0)))) * ar.FPS
    cap = round(min(0.5 * peak, SPEED_DEFAULT), 3)
    names = []
    values = []
    for j, i in enumerate(idx):
        vals = [float(x) for x in q[i]]
        assert all(abs(x) <= 100.0 for x in vals), (name, i, vals)
        names.append(f"{name}_w{j:02d}")
        values.append((vals, i))
    head = [
        f"# Plate {name.upper()}: recorded arm path from dataset 6d6bdce3 (LeRobot v3), episode {note['episode']}.",
        f"# Release frame {r} = {r / ar.FPS:.3f} s (gripper crosses 6.0 upward after a brick hold).",
        f"# End frame {end} = {end / ar.FPS:.3f} s ({note['end_rule']}).",
        f"# Waypoints every 0.25 s (7.5 frames at 30 fps), w00 = release (plate) pose, last waypoint = home.",
        "# Arm joints are the recorded values (rounded to 4 decimals); gripper fixed at 25.0 (open, no brick).",
        f"# Speed cap {cap} units/s = min(0.5 x recorded peak {peak:.3f} units/s, executor default {SPEED_DEFAULT}).",
        "# No placeholder key: poses come from recorded data, not from teach_poses.py.",
        "unit: normalized",
        f"joint_order: {JOINTS}",
        "poses:",
    ]
    body = [f"  {n}: {fmt_pose(v + [GRIP_OPEN])}" for n, (v, _) in zip(names, values, strict=True)]
    seq_fwd = [f"    - {{pose: {n}, speed: {cap:.3f}}}" for n in names]
    seq_rev = [f"    - {{pose: {n}, speed: {cap:.3f}}}" for n in reversed(names)]
    seq_rev[-1] = f"    - {{pose: {names[0]}, speed: {cap:.3f}, dwell_s: 1.0}}"
    lines = head + body + ["sequences:", f"  {name}_plate_to_home:"] + seq_fwd + [f"  {name}_home_to_plate:"] + seq_rev
    text = "\n".join(lines) + "\n"
    GM.mkdir(parents=True, exist_ok=True)
    (GM / f"{name}_plate.yaml").write_text(text, encoding="utf-8")
    (E1 / "waypoints").mkdir(parents=True, exist_ok=True)
    (E1 / "waypoints" / f"{name}_plate.yaml").write_text(text, encoding="utf-8")
    return {
        "plate": name,
        "episode": note["episode"],
        "release_frame": r,
        "release_t_s": round(r / ar.FPS, 3),
        "end_frame": end,
        "end_t_s": round(end / ar.FPS, 3),
        "end_rule": note["end_rule"],
        "n_waypoints": len(names),
        "waypoint_frames": [i for _, i in values],
        "recorded_peak_joint_speed_units_s": round(peak, 3),
        "speed_cap_units_s": cap,
        "release_pose_arm": [round(x, 4) for x in values[0][0]],
        "home_pose_arm": [round(x, 4) for x in values[-1][0]],
        "yaml": f"R/app/motion/gm/{name}_plate.yaml",
    }


def main() -> None:
    data = ar.read_all("data/chunk-000/*.parquet").sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    ep_ids = data["episode_index"].to_numpy()
    out = {}

    s11 = states[ep_ids == 11]
    rel11 = ar.release_frames(s11[:, 5])
    r11 = rel11[0]
    h11 = strict_home(s11, r11)
    assert h11 is not None, "ep11 strict home not found"
    out["blue"] = build_plate(
        "blue",
        s11,
        r11,
        h11,
        {"episode": 11, "end_rule": "strict home: shoulder_lift within 5 of -98 and 15 frames speed < 1.0 units/s"},
    )

    s9 = states[ep_ids == 9]
    rel9 = ar.release_frames(s9[:, 5])
    r9 = rel9[0]
    assert strict_home(s9, r9) is None, "ep9 unexpectedly has a strict home"
    out["green"] = build_plate(
        "green",
        s9,
        r9,
        len(s9) - 1,
        {
            "episode": 9,
            "end_rule": "RELAXED: recording end, shoulder_lift %.2f (within 5 of -98); no 15-frame stationary dwell exists"
            % float(s9[-1, 1]),
        },
    )

    alternates = {}
    for e in (10, 14, 15):
        se = states[ep_ids == e]
        re_ = ar.release_frames(se[:, 5])
        alternates[e] = {
            "release_frames": re_,
            "release_t_s": [round(x / ar.FPS, 3) for x in re_],
            "strict_home": strict_home(se, re_[0]) if re_ else None,
            "lift_at_end": round(float(se[-1, 1]), 3),
        }
    out["green_alternates"] = alternates
    out["ep8_note"] = "release at 5.0 s over GREEN but recording ends with lift -90.1 (not within 5 of -98): excluded"
    out["ep7_note"] = "release at 40.367 s over BLUE but lift -9 at end: no return home: excluded"
    out["ep12_note"] = "release at 22.7 s over BLUE, lift in tolerance from 26.27 s, no 15-frame stationary dwell: alternate"
    out["ep13_note"] = "no brick hold (gripper min 11.1): no release: excluded"
    (E1 / "segments.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k in ("blue", "green")}, indent=1))
    print(json.dumps(alternates, indent=1))


if __name__ == "__main__":
    main()
