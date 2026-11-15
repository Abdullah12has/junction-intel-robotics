"""Fake-bus dry run of the executor (app/motion/so101_motion.py MotionPlayer).

A simulated follower replies instantly with the goal and a virtual clock replaces time.
No serial port, no camera, no motion. Writes setpoint CSVs and summary.json into E1/fake_bus/.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

APP = Path("/home/intel/intel-junction-2026/app/motion")
E1 = Path("/home/intel/intel-junction-2026/work/P2-B-GM/E1")
GM = Path("/home/intel/intel-junction-2026/app/motion/gm")
sys.path.insert(0, str(APP))
import so101_motion as sm  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class VirtualClock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.t += seconds


class _Obs:
    def __init__(self, q: np.ndarray) -> None:
        self.joint_positions = q


class FakeFollower:
    max_speed = 279.0
    JOINT_ORDER = sm.JOINT_ORDER

    def __init__(self, start: np.ndarray, clock: VirtualClock) -> None:
        self.cur = np.asarray(start, dtype=np.float64).copy()
        self.clock = clock
        self.t_log: list[float] = []
        self.q_log: list[np.ndarray] = []
        self.step_log: list[int] = []
        self.step = 0

    def get_observation(self) -> _Obs:
        return _Obs(self.cur.copy())

    def send_action(self, setpoint) -> None:
        sp = np.asarray(setpoint, dtype=np.float64).copy()
        self.cur = sp
        self.t_log.append(self.clock.t)
        self.q_log.append(sp)
        self.step_log.append(self.step)

    def _read_joint_positions(self) -> np.ndarray:
        return self.cur.copy()

    def _write_joint_positions(self, raw) -> None:
        return None

    def set_torque(self, enabled: bool) -> None:
        return None

    def disconnect(self) -> None:
        return None


def seg_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    u = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    proj = a + u[:, None] * ab
    return np.linalg.norm(points - proj, axis=1)


def run_plate(name: str) -> dict:
    yaml_path = GM / f"{name}_plate.yaml"
    poses = sm.load_poses(yaml_path)
    seqs = sm.load_sequences(yaml_path)
    clock = VirtualClock()
    sm.time = clock  # so101_motion.py uses time.monotonic() and time.sleep(); both now virtual
    out = {}
    recorded = json.loads((E1 / "segments.json").read_text(encoding="utf-8"))[name]
    wp_names = [f"{name}_w{j:02d}" for j in range(recorded["n_waypoints"])]
    wp_path = np.array([poses[n][:5] for n in wp_names])
    seg_rec = np.load(E1 / f"{name}_recorded_segment.npy") if (E1 / f"{name}_recorded_segment.npy").exists() else None
    for seq_name in (f"{name}_plate_to_home", f"{name}_home_to_plate"):
        steps = seqs[seq_name]
        start = poses[steps[0].pose]
        fake = FakeFollower(start, clock)
        player = sm.MotionPlayer(fake, max_speed=sm.DEFAULT_MAX_SPEED, rate_hz=sm.DEFAULT_RATE_HZ, tolerance=sm.DEFAULT_TOLERANCE)
        step_rows = []
        for index, step in enumerate(steps, start=1):
            fake.step = index
            t0 = clock.t
            res = player.move_to(poses[step.pose], max_speed=step.speed)
            step_rows.append({"step": index, "pose": step.pose, "t_start_s": round(t0, 3), "t_end_s": round(clock.t, 3),
                              "arrived": bool(res.arrived), "max_error": round(res.max_error, 4),
                              "setpoints": res.setpoints})
            if not res.arrived and not step.allow_stall:
                raise sm.MotionStalledError(f"{seq_name} step {index} not reached", [])
            if step.dwell_s > 0:
                clock.sleep(step.dwell_s)
        t = np.asarray(fake.t_log)
        q = np.asarray(fake.q_log)
        steps_idx = np.asarray(fake.step_log)
        dt = 1.0 / sm.DEFAULT_RATE_HZ
        rate = np.max(np.abs(np.diff(q[:, :5], axis=0))) / dt if len(q) > 1 else 0.0
        second = np.diff(q[:, :5], n=2, axis=0)
        accel = np.max(np.abs(second)) / dt**2 if len(second) else 0.0
        chord = None
        if seg_rec is not None and name:
            chord = None
        exec_path = q[:, :5]
        ref = np.asarray(seg_rec) if seg_rec is not None else None
        csv_path = E1 / "fake_bus" / f"{seq_name}_setpoints.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s"] + JOINTS + ["step"])
            for ti, qi, si in zip(t, q, steps_idx, strict=True):
                writer.writerow([f"{ti:.4f}"] + [f"{x:.4f}" for x in qi] + [int(si)])
        gripper_ok = bool(np.all(np.abs(q[:, 5] - 25.0) < 1e-9))
        out[seq_name] = {
            "n_steps": len(steps),
            "n_setpoints": int(len(q)),
            "virtual_duration_s": round(float(t[-1] - t[0]) + dt, 3),
            "all_steps_arrived": all(r["arrived"] for r in step_rows),
            "max_step_error_units": max(r["max_error"] for r in step_rows),
            "peak_joint_rate_units_s": round(float(rate), 3),
            "peak_joint_accel_units_s2": round(float(accel), 3),
            "rate_within_cap": bool(rate <= 25.0 + 1e-6),
            "accel_within_cap": bool(accel <= sm.DEFAULT_MAX_ACCEL + 1e-6),
            "body_within_100": bool(np.all(np.abs(q[:, :5]) <= 100.0)),
            "gripper_fixed_25": gripper_ok,
            "final_setpoint": [round(float(x), 4) for x in q[-1]],
            "final_equals_last_waypoint": bool(np.allclose(q[-1], poses[steps[-1].pose], atol=1e-9)),
            "first_setpoint_equals_first_pose": bool(np.allclose(q[0], poses[steps[0].pose], atol=1e-9)),
            "steps": step_rows,
            "csv": str(csv_path),
        }
        if ref is not None and name:
            pass
        out[seq_name]["exec_path_points"] = int(len(exec_path))
    wp_chord = {}
    if seg_rec is not None:
        dists = np.full(len(seg_rec), np.inf)
        for i in range(len(wp_path) - 1):
            dists = np.minimum(dists, seg_dist(seg_rec[:, :5], wp_path[i], wp_path[i + 1]))
        wp_chord = {"max_recorded_frame_to_waypoint_polyline_units": round(float(np.max(dists)), 4),
                    "n_recorded_frames": int(len(seg_rec))}
    out["waypoint_polyline_check"] = wp_chord
    return out


def main() -> None:
    results = {}
    for name in ("blue", "green"):
        results[name] = run_plate(name)
    (E1 / "fake_bus").mkdir(parents=True, exist_ok=True)
    (E1 / "fake_bus" / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name, r in results.items():
        for key, val in r.items():
            if key == "waypoint_polyline_check":
                print(name, key, val)
                continue
            print(name, key, {k: v for k, v in val.items() if k not in ("steps", "csv")})


if __name__ == "__main__":
    main()
