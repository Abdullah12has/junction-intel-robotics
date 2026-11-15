"""Policy runner for Studio-exported policies (P2-B-OVRUN, builder E1). Dry-run in this build.

Modes
  --mode fake (default, executed in this build): replays one recorded episode of dataset a4ba6ede through
               the same PolicySource path as replay.py. No camera, no robot, no Studio.
  --mode live (NOT EXECUTED here): opens the follower by its by-id port through app/motion
               (open_follower_held: goal = present pose before torque), captures both RealSense views
               (physicalai RealSenseCamera, 640x480 @ 30, RGB, as Studio's camera_factory does), runs the
               policy at 30 Hz through PolicySource, sends each capped action, stops on control.json stop.
               Requires --live, --pause-checked, a typed GO, and R/run/HASSAN_PRESENT.

Outputs (LOOP-CONTRACT section 8): <RUN_ROOT>/<run_id>/events.jsonl with run_start, policy_run
{phase: start|grasp|release|return|done|timeout|error, model_id, task_text, plate, t}, run_end;
latency.csv (one row per tick). control.json in the run dir: {"cmd": "stop|quit", "id": <int>}.
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/home/intel/intel-junction-2026/app/ovrun")
import replay as rp  # noqa: E402  (StudioWorkerPath, Obs, decode_camera, paths, FPS, MODEL_ID)
from physicalai.inference import InferenceModel  # noqa: E402
from physicalai.runtime import ChunkedActionQueue, LerpSmoother, PolicySource, SyncExecution  # noqa: E402

R = "/home/intel/intel-junction-2026"
RUN_ROOT = f"{R}/logs/ovrun"
HASSAN_PRESENT = f"{R}/run/HASSAN_PRESENT"
SERIALS = ["243722064316", "243622061421"]  # close low view, top view (Studio dataset keys)
MAX_STEP = 25.0 / 30.0  # normalized units per tick; app/motion DEFAULT_MAX_SPEED = 25 units/s at 30 Hz
HOME_TOL = 5.0  # max abs joint error (joints 0-4) to count as home
GRIP_DROP = 5.0  # gripper units below its running max that count as a grasp
GRIP_RISE = 8.0  # gripper units above its running min that count as a release
STILL_DELTA = 0.2  # max abs change per tick that still counts as stationary
TICK_S = 1.0 / 30.0


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def clean(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    return x


class RunLog:
    def __init__(self, run_id):
        self.dir = os.path.join(RUN_ROOT, run_id)
        os.makedirs(self.dir, exist_ok=True)
        cur = os.path.join(RUN_ROOT, "current")
        if os.path.islink(cur):
            os.unlink(cur)
        os.symlink(self.dir, cur)
        self.events = open(os.path.join(self.dir, "events.jsonl"), "a", buffering=1)
        self.lat_fh = open(os.path.join(self.dir, "latency.csv"), "w", newline="")
        self.lat = csv.writer(self.lat_fh)
        self.lat.writerow(["tick", "update_ms", "queried"])

    def emit(self, typ, **kw):
        rec = {"t": utc(), "mono": time.monotonic(), "type": typ}
        rec.update(kw)
        self.events.write(json.dumps(clean(rec)) + "\n")

    def close(self):
        self.events.close()
        self.lat_fh.close()


def read_control(run_dir, last_id):
    try:
        with open(os.path.join(run_dir, "control.json")) as fh:
            cmd = json.load(fh)
    except (OSError, ValueError):
        return None, last_id
    if int(cmd.get("id", -1)) > last_id:
        return cmd.get("cmd"), int(cmd["id"])
    return None, last_id


def hands_in_scene_hook(frames):
    """Refusal hook for hands in the workspace. Stub in this build: returns None (no detector yet)."""
    return None


class Episode:
    """Phases from the gripper and the home distance: start, grasp, release, return, done, timeout."""

    def __init__(self, max_s, home):
        self.max_ticks = int(max_s * rp.FPS)
        self.home = np.asarray(home, dtype=float)[:5]  # the cycle starts at home (OD-12): home = pose at run start
        self.phase = "start"
        self.gmax = self.gmin = None
        self.still = 0
        self.prev = None

    def step(self, state, t):
        out = []
        g = float(state[5])
        self.gmax = g if self.gmax is None else max(self.gmax, g)
        self.gmin = g if self.gmin is None else min(self.gmin, g)
        if self.phase == "start":
            self.phase = "run"
            out.append("start")
        if self.phase == "run" and g < self.gmax - GRIP_DROP:
            self.phase, self.gmin = "grasp", g
            out.append("grasp")
        elif self.phase == "grasp" and g > self.gmin + GRIP_RISE:
            self.phase = "release"
            out.append("release")
        elif self.phase == "release" and float(np.max(np.abs(np.asarray(state, float)[:5] - self.home))) < HOME_TOL:
            self.phase = "return"
            out.append("return")
        if self.phase == "return":
            moving = self.prev is not None and float(np.max(np.abs(state - self.prev))) > STILL_DELTA
            self.still = 0 if moving else self.still + 1
            if self.still >= rp.FPS:
                self.phase = "done"
                out.append("done")
        self.prev = np.array(state, dtype=float)
        if self.phase not in ("done", "timeout") and t >= self.max_ticks:
            self.phase = "timeout"
            out.append("timeout")
        return out


class LiveFrames:
    """Lets StudioWorkerPath index frames[name][tick] while the frame is the latest camera read."""

    def __init__(self, getter):
        self.getter = getter

    def __getitem__(self, name):
        return _Latest(self.getter, name)


class _Latest:
    def __init__(self, getter, name):
        self.getter, self.name = getter, name

    def __getitem__(self, _tick):
        return self.getter(self.name)


def make_source(path, task_text):
    src = PolicySource(path, SyncExecution(request_threshold=0.5),
                       ChunkedActionQueue(smoother=LerpSmoother()), task=task_text)
    src.connect(bus=None, session_id="")
    return src


def run_fake(args, log):
    prep = json.load(open(f"{rp.W}/data/prep.json"))
    arr = np.load(f"{rp.W}/data/a4ba6ede_arrays.npz")
    ep_meta = next(e for e in prep["episodes"] if e["episode_index"] == args.episode)
    rows = np.where(arr["episode_index"] == args.episode)[0]
    S = arr["state"][rows]
    cams = [k for k in ["realsense 243722064316", "realsense 243622061421"]]
    frames = {f"images.{c}": rp.decode_camera(c) for c in cams}
    g0 = int(round(float(ep_meta[f"videos/observation.images.{cams[0]}/from_timestamp"]) * rp.FPS))
    task_text = prep["task_text"][str(int(arr["task_index"][rows[0]]))]
    model = InferenceModel(args.model_dir, backend="openvino", device=args.device)
    path = rp.StudioWorkerPath(model, frames, list(frames.keys()))
    src = make_source(path, task_text)
    episode = Episode(args.max_s, S[0])
    log.emit("run_start", run_id=os.path.basename(log.dir), mode="fake", config={
        "device": args.device, "episode": args.episode, "model_id": rp.MODEL_ID, "task_text": task_text,
        "max_s": args.max_s, "frame_offset": g0})
    last, reason = 0, "episode_end"
    for t in range(len(rows)):
        cmd, last = read_control(log.dir, last)
        if cmd in ("stop", "quit"):
            reason = "stop_requested"
            break
        path.tick = g0 + t
        t0 = time.perf_counter()
        src.update(rp.Obs(S[t], t * TICK_S), {}, t)
        dt = time.perf_counter() - t0
        log.lat.writerow([t, round(dt * 1000.0, 4), int(len(path.call_ticks) and path.call_ticks[-1] == g0 + t)])
        for ph in episode.step(S[t], t):
            log.emit("policy_run", phase=ph, model_id=rp.MODEL_ID, task_text=task_text, plate=None, t=t)
        if episode.phase in ("done", "timeout"):
            reason = episode.phase
            break
    log.emit("run_end", reason=reason, ticks=t + 1, inferences=len(path.call_ticks))
    return reason


def run_live(args, log):
    if not args.live or not args.pause_checked:
        raise SystemExit("live mode needs --live and --pause-checked (Opus checks PAUSE.txt before launch)")
    if not os.path.exists(HASSAN_PRESENT) or time.time() - os.path.getmtime(HASSAN_PRESENT) > 900:
        raise SystemExit(f"refused: {HASSAN_PRESENT} is absent or older than 15 min (Hassan must be present now)")
    if input("Type GO to start the live run: ").strip() != "GO":
        raise SystemExit("refused: GO not typed")
    sys.path.insert(0, f"{R}/app/motion")
    import so101_motion as sm  # validated executor and port rules (app/motion)
    from physicalai.capture import ColorMode, RealSenseCamera  # Studio's camera_factory uses this class

    port = sm.follower_serial_port(None)
    if sm.port_holder_pid(port) is not None:
        raise SystemExit("refused: the follower port is held by another process")
    robot = sm.load_studio_robot(args.robots_json, args.robot_name)
    cams = {f"images.realsense {s}": RealSenseCamera(serial_number=s, fps=30, width=640, height=480,
                                                     color_mode=ColorMode.RGB) for s in SERIALS}
    for cam in cams.values():
        cam.connect()
    log.emit("run_start", run_id=os.path.basename(log.dir), mode="live", config={
        "device": args.device, "port": port, "model_id": rp.MODEL_ID, "task_text": args.task,
        "max_s": args.max_s, "max_step": MAX_STEP})
    driver = None
    reason, last = "episode_end", 0
    try:
        frames_now = {k: c.read_latest().data for k, c in cams.items()}
        refusal = hands_in_scene_hook(frames_now)
        if refusal:
            raise SystemExit(f"refused by hands-in-scene hook: {refusal}")
        driver = sm.open_follower_held(port, robot.calibration, unit=robot.unit)
        model = InferenceModel(args.model_dir, backend="openvino", device=args.device)
        path = rp.StudioWorkerPath(model, LiveFrames(lambda name: cams[name].read_latest().data), list(cams.keys()))
        src = make_source(path, args.task)
        state0 = np.asarray(driver.get_observation().joint_positions, dtype=np.float64)
        episode, prev, t = Episode(args.max_s, state0), None, 0
        while True:
            cmd, last = read_control(log.dir, last)
            if cmd in ("stop", "quit"):
                reason = "stop_requested"
                break
            if t % 5 == 0 and hands_in_scene_hook({k: c.read_latest().data for k, c in cams.items()}):
                reason = "hands_in_scene"
                break
            state = np.asarray(driver.get_observation().joint_positions, dtype=np.float64)
            t0 = time.perf_counter()
            act = np.asarray(src.update(rp.Obs(state, time.monotonic()), {}, t), dtype=np.float64)
            dt = time.perf_counter() - t0
            target = act if prev is None else prev + np.clip(act - prev, -MAX_STEP, MAX_STEP)
            driver.send_action(target.astype(np.float32))
            prev = target
            log.lat.writerow([t, round(dt * 1000.0, 4), int(len(path.call_ticks) and path.call_ticks[-1] == t)])
            for ph in episode.step(state, t):
                log.emit("policy_run", phase=ph, model_id=rp.MODEL_ID, task_text=args.task, plate=None, t=t)
            if episode.phase in ("done", "timeout"):
                reason = episode.phase
                break
            t += 1
            time.sleep(TICK_S)
        src.disconnect()
    except BaseException as exc:
        log.emit("policy_run", phase="error", model_id=rp.MODEL_ID, task_text=args.task, plate=None,
                 message=f"{type(exc).__name__}: {str(exc)[:300]}")
        reason = "error"
        raise
    finally:
        held = reason not in ("done", "timeout", "episode_end")
        if driver is not None:
            if held:
                sm.MotionPlayer(driver).hold()  # stop or error: hold the present pose, torque stays on
            else:
                driver.set_torque(enabled=False)  # torque off at the end of a finished episode
            driver.disconnect()
        for cam in cams.values():
            cam.disconnect()
        log.emit("run_end", reason=reason, torque_left_on=bool(held and driver is not None))
    return reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fake", "live"], default="fake")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--model-dir", default=rp.EXPORT)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--task", default="move the brick to the inspection spot")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--robots-json", default=f"{R}/app/motion/project_robots.json")
    ap.add_argument("--robot-name", default="follower")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--pause-checked", action="store_true")
    args = ap.parse_args()
    run_id = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    log = RunLog(run_id)
    try:
        reason = run_fake(args, log) if args.mode == "fake" else run_live(args, log)
        print(utc(), "run", run_id, "mode", args.mode, "reason", reason, flush=True)
    finally:
        log.close()


if __name__ == "__main__":
    main()
