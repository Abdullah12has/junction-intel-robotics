"""Read-only analysis: release events and return-to-home segments of one LeRobot v3 dataset.

Writes analysis.json into the E1 folder. Never touches a serial port or camera.
"""

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/intel/intel-junction-2026/data/studio-datasets/6d6bdce3-b382-478d-9697-fb9e573b97af")
OUT = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1/analysis.json")
FPS = 30.0
GRIP_CLOSED = 6.0
BRICK_LO, BRICK_HI = 1.6, 4.5
HOME_LIFT = -98.0
HOME_TOL = 5.0
STILL_SPEED = 1.0
STILL_FRAMES = 15
PRE_HOLD = 15


def read_all(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(ROOT / pattern)))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def release_frames(g: np.ndarray) -> list[int]:
    """Release = first frame >= 6.0 after a brick-hold run (>= PRE_HOLD frames in [1.6, 4.5]), within 2 s."""
    n = len(g)
    in_hold = (g >= BRICK_LO) & (g <= BRICK_HI)
    out = []
    i = 0
    while i < n:
        if not in_hold[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and in_hold[j + 1]:
            j += 1
        if j - i + 1 >= PRE_HOLD:
            k = j + 1
            while k < n and k <= j + 60 and g[k] < GRIP_CLOSED:
                k += 1
            if k < n and k <= j + 60 and g[k] >= GRIP_CLOSED:
                out.append(k)
        i = j + 1
    return out


def analyse(s: np.ndarray) -> dict:
    g = s[:, 5]
    q = s[:, :5]
    n = len(s)
    v = np.zeros(n)
    v[:-1] = np.linalg.norm(np.diff(q, axis=0), axis=1) * FPS
    still = np.zeros(n, dtype=bool)
    for k in range(n - STILL_FRAMES + 1):
        still[k] = bool(np.all(v[k : k + STILL_FRAMES] < STILL_SPEED))
    releases = release_frames(g)
    events = []
    for t in releases:
        home = None
        for k in range(t, n):
            if abs(q[k, 1] - HOME_LIFT) <= HOME_TOL and still[k]:
                home = k
                break
        pan = float(q[t, 0])
        if pan > 0:
            side = "BLUE-side (pan>0)"
        elif pan < -30:
            side = "GREEN-side (pan<-30)"
        else:
            side = "UNCLEAR (pan -30..0)"
        seg = None
        if home is not None:
            seg = {
                "home_frame": int(home),
                "home_t_s": round(home / FPS, 3),
                "duration_s": round((home - t) / FPS, 3),
                "peak_joint_speed_units_s": round(float(np.max(np.abs(np.diff(q[t : home + 1], axis=0)))) * FPS, 3),
                "home_pose": [round(float(x), 3) for x in q[home]],
            }
        events.append(
            {
                "release_frame": int(t),
                "release_t_s": round(t / FPS, 3),
                "release_pose_arm": [round(float(x), 3) for x in q[t]],
                "release_gripper": round(float(g[t]), 3),
                "side": side,
                "return_home": seg,
            }
        )
    return {
        "n_frames": int(n),
        "length_s": round(n / FPS, 3),
        "start_pose_arm": [round(float(x), 3) for x in q[0]],
        "end_pose_arm": [round(float(x), 3) for x in q[-1]],
        "end_lift_gap": round(float(q[-1, 1] - HOME_LIFT), 3),
        "gripper_min": round(float(g.min()), 3),
        "gripper_max": round(float(g.max()), 3),
        "releases": events,
    }


def main() -> None:
    tasks = pd.read_parquet(ROOT / "meta/tasks.parquet")
    print("tasks columns:", tasks.columns.tolist(), "index:", tasks.index.name)
    meta = read_all("meta/episodes/chunk-000/*.parquet")
    print("meta columns:", meta.columns.tolist(), "rows:", len(meta))
    data = read_all("data/chunk-000/*.parquet")
    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    print("data columns:", data.columns.tolist(), "rows:", len(data))
    states = np.stack(data["observation.state"].to_numpy()).astype(np.float64)
    ep_ids = data["episode_index"].to_numpy()
    summary = {}
    for e in np.unique(ep_ids):
        summary[int(e)] = analyse(states[ep_ids == e])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    for e, info in summary.items():
        print(
            f"ep {e:>2} len_s={info['length_s']} start={info['start_pose_arm']} end_lift_gap={info['end_lift_gap']} "
            f"grip[{info['gripper_min']},{info['gripper_max']}] releases={len(info['releases'])}"
        )
        for ev in info["releases"]:
            rh = ev["return_home"]
            if rh is None:
                home_txt = "NO RETURN HOME"
            else:
                home_txt = f"home_t={rh['home_t_s']} dur={rh['duration_s']} peak={rh['peak_joint_speed_units_s']}"
            print(f"    release t={ev['release_t_s']} g={ev['release_gripper']} pose={ev['release_pose_arm']} {ev['side']} {home_txt}")


if __name__ == "__main__":
    main()
