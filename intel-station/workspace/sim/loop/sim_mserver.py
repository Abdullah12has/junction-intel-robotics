import argparse
import bisect
import json
import math
import os
import random
import signal
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

TWIN = "/home/intel/intel-junction-2026/sim/twin"
ROBOT_XML = f"{TWIN}/robot/so101_twin.xml"
SCENE_XML = f"{TWIN}/scene/scene_twin_v1.xml"
DYN_JSON = f"{TWIN}/dynamics/twin_dynamics_v1.json"
ROBOT_MESHES = f"{TWIN}/robot/meshes"
SIM_RUN = "/home/intel/intel-junction-2026/sim/run"
REAL_RUN = "/home/intel/intel-junction-2026/run"
RUN_ROOT = os.environ.get("JUNCTION_RUN_ROOT") or SIM_RUN
UNIT_ROOT = f"{SIM_RUN}/unit"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
BRICKS = ("brick_blue", "brick_orange", "brick_yellow", "brick_white")
GRIP = 5
TICKS_PER_REV = 4095.0
CALIBRATION = {
    "shoulder_pan": (891, 3212),
    "shoulder_lift": (822, 3090),
    "elbow_flex": (898, 3102),
    "wrist_flex": (812, 3171),
    "wrist_roll": (74, 3899),
    "gripper": (2046, 3577),
}
K = {j: (hi - lo) / (100.0 if j == "gripper" else 200.0) * 2.0 * math.pi / TICKS_PER_REV
     for j, (lo, hi) in CALIBRATION.items()}
GRIP_CLOSED_STOP = -0.174533
GRIP_EMPTY_NORM = 0.52
INIT_NORM = [-12.54, 12.79, -24.05, 89.74, -62.09, 24.62]
HARD_SPEED = 45.0
HARD_ACCEL = 200.0
OFFSET_CLAMP = 8.0
RATE_HZ = 30.0
DT = 1.0 / RATE_HZ
BODY_LIMIT = 100.0
GRIP_LOW = 0.0
GRIP_HIGH = 100.0
COMP_MAX_ITERS = 10
COMP_GAIN = 0.7
COMP_TOL = 0.5
COMP_STEP_S = 0.35
SETTLE_S = 0.5
PHYS_DT = 0.004
HELD_RANGE = (1.57, 2.42)
GRASP_RADIUS = 0.03
RELEASE_MARGIN = 2.0
SITE = "gripper_frame_link"
MEASURE_POSES = [
    [0.0, 30.0, -30.0, 40.0, 0.0, 25.0],
    [20.0, -40.0, 50.0, -30.0, 10.0, 25.0],
    [-30.0, 60.0, -20.0, 10.0, -20.0, 25.0],
    [10.0, 80.0, -60.0, 20.0, 0.0, 25.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 25.0],
]


class MotionError(ValueError):
    pass


def norm_to_joint_rad(joint, n):
    if joint == "gripper":
        return GRIP_CLOSED_STOP + K[joint] * (float(n) - GRIP_EMPTY_NORM)
    return K[joint] * float(n)


def joint_rad_to_norm(joint, q):
    if joint == "gripper":
        return GRIP_EMPTY_NORM + (float(q) - GRIP_CLOSED_STOP) / K[joint]
    return float(q) / K[joint]


def rounded(values):
    return [round(float(x), 2) for x in values]


def finite_positive(value):
    return math.isfinite(value) and value > 0.0


def validate_target(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (len(JOINTS),):
        raise MotionError(f"expected {len(JOINTS)} values in order {JOINTS}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise MotionError("joint values must be finite")
    if np.any(np.abs(arr[:GRIP]) > BODY_LIMIT):
        raise MotionError(f"body joints must lie in [-100, 100], got {arr[:GRIP].tolist()}")
    if not GRIP_LOW <= arr[GRIP] <= GRIP_HIGH:
        raise MotionError(f"gripper must lie in [0, 100], got {arr[GRIP]}")
    return arr


def plan_ramp(start, target, max_speed, dt, max_accel=50.0):
    if not (finite_positive(max_speed) and finite_positive(max_accel) and finite_positive(dt)):
        raise MotionError("max_speed, max_accel and dt must be finite and positive")
    s = np.asarray(start, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if s.shape != t.shape:
        raise MotionError(f"start shape {s.shape} differs from target shape {t.shape}")
    delta = float(np.max(np.abs(t - s)))
    if delta == 0.0:
        return t.reshape(1, -1).copy()
    duration = max(1.5 * delta / max_speed, math.sqrt(6.0 * delta / max_accel))
    steps = max(1, math.ceil(duration / dt))
    u = np.arange(1, steps + 1, dtype=np.float64) / steps
    ease = 3.0 * u ** 2 - 2.0 * u ** 3
    setpoints = s + np.outer(ease, t - s)
    setpoints[-1] = t
    return setpoints


def load_params():
    with open(DYN_JSON) as f:
        return json.load(f)["parameters"]


def build_xml(params, gravity=True):
    robot = ET.parse(ROBOT_XML).getroot()
    scene = ET.parse(SCENE_XML).getroot()
    root = ET.Element("mujoco", model="sim_so101_world")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true", meshdir=ROBOT_MESHES)
    ET.SubElement(root, "option", timestep=str(PHYS_DT), integrator="implicitfast",
                  gravity="0 0 -9.81" if gravity else "0 0 0")
    root.append(robot.find("asset"))
    world = ET.SubElement(root, "worldbody")
    for element in scene.find("worldbody"):
        if element.get("euler") is not None:
            values = [f"{math.radians(float(v)):.10g}" for v in element.get("euler").split()]
            element.set("euler", " ".join(values))
        world.append(element)
    for element in robot.find("worldbody"):
        world.append(element)
    for j in JOINTS:
        joint = world.find(f".//joint[@name='{j}']")
        joint.set("armature", f"{params[j]['armature']:.9g}")
        joint.set("frictionloss", f"{params[j]['frictionloss']:.9g}")
        joint.set("damping", "0")
    act = ET.SubElement(root, "actuator")
    for j in JOINTS:
        p = params[j]
        ET.SubElement(act, "position", name=f"act_{j}", joint=j, kp=f"{p['kp']:.9g}", kv=f"{p['kv']:.9g}",
                      forcelimited="true", forcerange=f"{-p['forcerange']:g} {p['forcerange']:g}")
    return ET.tostring(root, encoding="unicode")


class World:
    def __init__(self, params, dynamics=True, gravity=True, weld=True, seed=0, init_norm=None):
        self.model = mujoco.MjModel.from_xml_string(build_xml(params, gravity))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.h = m.opt.timestep
        self.dynamics = dynamics
        self.weld_enabled = weld
        self.t = 0.0
        self.qadr = {}
        self.dadr = {}
        self.aid = {}
        self.delay = {}
        for j in JOINTS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.qadr[j] = int(m.jnt_qposadr[jid])
            self.dadr[j] = int(m.jnt_dofadr[jid])
            self.aid[j] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{j}")
            self.delay[j] = float(params[j]["delay"])
        self.gain0 = m.actuator_gainprm.copy()
        self.bias0 = m.actuator_biasprm.copy()
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, SITE)
        rng = random.Random(seed)
        self.bricks = {}
        self.held = {}
        for b in BRICKS:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{b}_free")
            self.bricks[b] = (int(m.jnt_qposadr[jid]), int(m.jnt_dofadr[jid]))
            self.held[b] = rng.uniform(*HELD_RANGE)
        self.weld = None
        start = np.array(INIT_NORM if init_norm is None else init_norm, dtype=np.float64)
        self.data.qpos[:] = m.qpos0
        for i, j in enumerate(JOINTS):
            self.data.qpos[self.qadr[j]] = norm_to_joint_rad(j, start[i])
        self.goal = start.copy()
        self.hist_t = [0.0]
        self.hist_g = [start.copy()]
        mujoco.mj_forward(m, self.data)

    def reading(self):
        return np.array([joint_rad_to_norm(j, self.data.qpos[self.qadr[j]]) for j in JOINTS])

    def torque(self, enabled):
        if enabled:
            self.model.actuator_gainprm[:] = self.gain0
            self.model.actuator_biasprm[:] = self.bias0
        else:
            self.model.actuator_gainprm[:, 0] = 0.0
            self.model.actuator_biasprm[:, 1:3] = 0.0

    def goal_at(self, tau):
        idx = bisect.bisect_right(self.hist_t, tau) - 1
        return self.hist_g[max(idx, 0)]

    def send(self, goal):
        g = np.array(goal, dtype=np.float64)
        self.goal = g
        self.hist_t.append(self.t)
        self.hist_g.append(g)
        if len(self.hist_t) > 4000:
            cut = bisect.bisect_right(self.hist_t, self.t - 2.0) - 1
            if cut > 0:
                del self.hist_t[:cut]
                del self.hist_g[:cut]

    def hold(self):
        self.send(self.reading())

    def advance_to(self, t_target):
        while self.t < t_target - 0.5 * self.h:
            self.step()

    def step(self):
        d = self.data
        if self.dynamics:
            for i, j in enumerate(JOINTS):
                n = self.goal_at(self.t - self.delay[j])[i]
                d.ctrl[self.aid[j]] = norm_to_joint_rad(j, n)
        else:
            for i, j in enumerate(JOINTS):
                d.qpos[self.qadr[j]] = norm_to_joint_rad(j, self.goal[i])
                d.qvel[self.dadr[j]] = 0.0
        self.update_weld()
        if self.dynamics:
            mujoco.mj_step(self.model, d)
        else:
            mujoco.mj_forward(self.model, d)
        self.t += self.h

    def update_weld(self):
        if not self.weld_enabled:
            return
        d = self.data
        gq = self.qadr["gripper"]
        gad = self.dadr["gripper"]
        reading_g = joint_rad_to_norm("gripper", d.qpos[gq])
        goal_g = float(self.goal[GRIP])
        site_p = d.site_xpos[self.site].copy()
        site_r = d.site_xmat[self.site].reshape(3, 3).copy()
        if self.weld is None:
            if goal_g < reading_g - 0.2:
                for b, (qa, _) in self.bricks.items():
                    bp = d.qpos[qa:qa + 3].copy()
                    if np.linalg.norm(bp - site_p) < GRASP_RADIUS and reading_g <= self.held[b]:
                        brick_r = np.zeros(9)
                        mujoco.mju_quat2Mat(brick_r, d.qpos[qa + 3:qa + 7].copy())
                        self.weld = {"brick": b, "rel_p": site_r.T @ (bp - site_p),
                                     "rel_r": site_r.T @ brick_r.reshape(3, 3), "held": self.held[b]}
                        break
        elif goal_g > self.weld["held"] + RELEASE_MARGIN:
            self.weld = None
        if self.weld is None:
            return
        qa, da = self.bricks[self.weld["brick"]]
        pos = site_p + site_r @ self.weld["rel_p"]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (site_r @ self.weld["rel_r"]).reshape(9))
        d.qpos[qa:qa + 3] = pos
        d.qpos[qa + 3:qa + 7] = quat
        d.qvel[da:da + 6] = 0.0
        d.qpos[gq] = norm_to_joint_rad("gripper", self.weld["held"])
        d.qvel[gad] = 0.0


class VirtualClock:
    def __init__(self, world):
        self.world = world
        self.t = 0.0
        self.wall0 = time.time()

    def now(self):
        return self.t

    def wall(self):
        return self.wall0 + self.t

    def sleep(self, seconds):
        self.t += seconds
        self.world.advance_to(self.t)


class RealClock:
    def __init__(self, world):
        self.world = world
        self.t0 = time.monotonic()

    def now(self):
        return time.monotonic() - self.t0

    def wall(self):
        return time.time()

    def sleep(self, seconds):
        self.world.advance_to(self.now())
        time.sleep(seconds)
        self.world.advance_to(self.now())


class SimMotion:
    def __init__(self, world, clock, run_root):
        self.world = world
        self.clock = clock
        self.D = f"{run_root}/mserver"
        self.estop_path = f"{run_root}/estop"
        self.active_path = f"{run_root}/motion_active"
        os.makedirs(self.D, exist_ok=True)
        self.cmdf = f"{self.D}/cmd.jsonl"
        open(self.cmdf, "a").close()
        self.pos = os.path.getsize(self.cmdf)
        self.goal = world.reading()
        self.target = self.goal.copy()
        self.offset = np.zeros(6)
        self.status = "ready"
        self.rec = None
        self.rec_last = 0.0
        self.stop = False
        self.release = False

    def log(self, msg):
        with open(f"{self.D}/server.log", "a") as f:
            f.write(time.strftime("%H:%M:%SZ ", time.gmtime(self.clock.wall())) + msg + "\n")

    def estop(self):
        return os.path.exists(self.estop_path) or self.stop

    def rec_tick(self, goal, force=False):
        if self.rec is None:
            return
        now = self.clock.wall()
        if not force and now - self.rec_last < 1.0 / 30 - 0.002:
            return
        cur = self.world.reading()
        self.rec_last = now
        self.rec.write("%.4f," % now + ",".join("%.2f" % x for x in goal) + ","
                       + ",".join("%.2f" % x for x in cur) + "\n")

    def send_for(self, goal, seconds):
        n = max(1, int(round(seconds / DT)))
        return self.paced(np.repeat(np.asarray(goal, dtype=np.float64).reshape(1, -1), n, axis=0))

    def paced(self, setpoints):
        for sp in setpoints:
            if self.estop():
                self.world.hold()
                cur = self.world.reading()
                if self.target[GRIP] < 5:
                    cur[GRIP] = self.target[GRIP]
                    self.world.send(cur)
                self.goal = self.world.reading()
                if self.target[GRIP] < 5:
                    self.goal[GRIP] = self.target[GRIP]
                self.status = "estop"
                self.log("ESTOP during motion - held at present")
                return False
            self.world.send(sp)
            self.goal = np.array(sp, dtype=np.float64)
            self.rec_tick(self.goal)
            self.clock.sleep(DT)
        return True

    def move(self, q, speed, comp, keep_grip=False, accel=60.0, keep_offset=False):
        if self.estop():
            self.status = "estop"
            return "refused: estop present"
        q = np.asarray(q, dtype=np.float64)
        if keep_grip:
            q[GRIP] = self.target[GRIP]
        validate_target(q)
        speed = min(float(speed), HARD_SPEED)
        if not comp and not keep_offset:
            self.offset[:] = 0
        g = q.copy()
        g[:GRIP] += self.offset[:GRIP]
        try:
            validate_target(g)
        except MotionError:
            self.offset[:] = 0
            g = q.copy()
        open(self.active_path, "w").close()
        its = 0
        try:
            ok = self.paced(plan_ramp(self.goal, g, speed, DT, max_accel=min(float(accel), HARD_ACCEL)))
            ok = ok and self.send_for(g, SETTLE_S)
            while ok and comp and its < COMP_MAX_ITERS:
                cur = self.world.reading()
                err = q - cur
                err[GRIP] = 0.0
                if np.max(np.abs(err[:GRIP])) <= COMP_TOL:
                    break
                self.offset[:GRIP] = np.clip(self.offset[:GRIP] + COMP_GAIN * err[:GRIP],
                                             -OFFSET_CLAMP, OFFSET_CLAMP)
                g = q.copy()
                g[:GRIP] += self.offset[:GRIP]
                try:
                    validate_target(g)
                except MotionError:
                    break
                ok = self.send_for(g, COMP_STEP_S)
                its += 1
        finally:
            if os.path.exists(self.active_path):
                os.remove(self.active_path)
        self.target = q
        if ok:
            self.status = "ok"
        return f"comp_iters={its if comp else 0}"

    def state(self, cid, msg):
        cur = self.world.reading()
        st = {"last_id": cid, "status": self.status, "msg": msg,
              "t": time.strftime("%H:%M:%SZ", time.gmtime(self.clock.wall())),
              "target": rounded(self.target), "goal": rounded(self.goal), "present": rounded(cur),
              "err": rounded(self.target - cur), "offset": rounded(self.offset)}
        tmp = f"{self.D}/state.tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, f"{self.D}/state.json")
        self.log(json.dumps(st))

    def rec_start(self, path):
        if self.rec is not None:
            self.rec.close()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.rec = open(path, "w", buffering=1)
        self.rec.write("t," + ",".join(f"goal{i}" for i in range(6)) + ","
                       + ",".join(f"pos{i}" for i in range(6)) + "\n")
        self.rec_tick(self.goal, force=True)
        return "rec_start " + path

    def rec_stop(self):
        if self.rec is not None:
            self.rec_tick(self.goal, force=True)
            self.rec.close()
        self.rec = None
        return "rec_stop"

    def poll(self):
        with open(self.cmdf) as f:
            f.seek(self.pos)
            lines = f.readlines()
            self.pos = f.tell()
        if not lines:
            self.rec_tick(self.goal)
            return False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                c = json.loads(line)
                cid, op = c.get("id", -1), c.get("op")
                if op == "move":
                    msg = self.move(c["q"], c.get("speed", 5), c.get("comp", True), accel=c.get("accel", 60.0))
                elif op == "rec_start":
                    msg = self.rec_start(c["path"])
                elif op == "rec_stop":
                    msg = self.rec_stop()
                elif op == "rel":
                    msg = self.move(self.target + np.asarray(c["dq"], dtype=np.float64), c.get("speed", 5),
                                    c.get("comp", True), accel=c.get("accel", 60.0))
                elif op == "grip":
                    q = self.target.copy()
                    q[GRIP] = float(c["value"])
                    msg = self.move(q, c.get("speed", 20), False, accel=c.get("accel", 120.0), keep_offset=True)
                elif op == "state":
                    msg = "state"
                elif op in ("exit", "release"):
                    self.release = op == "release"
                    self.state(cid, op)
                    self.stop = True
                    return True
                else:
                    msg = f"unknown op {op}"
            except Exception as e:
                self.status = "error"
                msg = f"error: {e!r}"
            self.state(cid, msg)
        return True

    def write_world(self):
        d = self.world.data
        bricks = {b: d.qpos[qa:qa + 7].tolist() for b, (qa, _) in self.world.bricks.items()}
        st = {"t": round(self.world.t, 4), "qpos": d.qpos.tolist(), "reading": rounded(self.world.reading()),
              "weld": None if self.world.weld is None else self.world.weld["brick"], "bricks": bricks}
        tmp = f"{self.D}/sim_world.tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, f"{self.D}/sim_world.json")

    def run(self):
        self.state(0, "started")
        last = 0.0
        try:
            while not self.stop:
                if not self.poll():
                    self.clock.sleep(0.01 if self.rec is not None else 0.05)
                if time.monotonic() - last > 0.1:
                    self.write_world()
                    last = time.monotonic()
        finally:
            if self.release:
                self.world.torque(False)
                self.log("released: torque off")
            if self.rec is not None:
                self.rec.close()
            self.log("sim port closed (no re-hold)")


def new_server(params, root, dynamics=True, gravity=True, weld=True, seed=0, virtual=True):
    world = World(params, dynamics=dynamics, gravity=gravity, weld=weld, seed=seed)
    clock = VirtualClock(world) if virtual else RealClock(world)
    return SimMotion(world, clock, root)


def place_brick(world, brick, pos):
    qa, da = world.bricks[brick]
    d = world.data
    d.qpos[qa:qa + 3] = pos
    d.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[da:da + 6] = 0.0
    mujoco.mj_forward(world.model, d)


def residual(srv):
    return np.abs(srv.target - srv.world.reading())[:GRIP]


def measure_config(params, root, name, **kw):
    srv = new_server(params, root, **kw)
    rows = []
    for pose in MEASURE_POSES:
        t0 = srv.clock.now()
        srv.move(pose, 15.0, False)
        e_nc = residual(srv)
        t1 = srv.clock.now()
        msg = srv.move(pose, 15.0, True)
        e_c = residual(srv)
        rows.append({"pose": pose, "no_comp_abs_err": rounded(e_nc), "comp_abs_err": rounded(e_c),
                     "comp_msg": msg, "move_no_comp_sim_s": round(t1 - t0, 3),
                     "move_comp_sim_s": round(srv.clock.now() - t1, 3), "status": srv.status})
    return {"name": name, "rows": rows}


def measure_gripper(params, root):
    out = {"empty_close_norm": [], "held_norm": [], "held_value_drawn": [], "open_norm": []}
    for seed in range(3):
        srv = new_server(params, root, gravity=False, seed=seed)
        srv.move(MEASURE_POSES[0], 15.0, True)
        place_brick(srv.world, "brick_blue", srv.world.data.site_xpos[srv.world.site].copy())
        q = srv.target.copy()
        q[GRIP] = 0.0
        srv.move(q, 20.0, False, accel=120.0, keep_offset=True)
        out["held_norm"].append(round(float(srv.world.reading()[GRIP]), 3))
        out["held_value_drawn"].append(round(float(srv.world.held["brick_blue"]), 3))
        q[GRIP] = 25.0
        srv.move(q, 20.0, False, accel=120.0, keep_offset=True)
        out["open_norm"].append(round(float(srv.world.reading()[GRIP]), 3))
        srv = new_server(params, root, gravity=False, seed=seed, weld=False)
        srv.move(MEASURE_POSES[0], 15.0, True)
        q = srv.target.copy()
        q[GRIP] = 0.0
        srv.move(q, 20.0, False, accel=120.0, keep_offset=True)
        out["empty_close_norm"].append(round(float(srv.world.reading()[GRIP]), 3))
    return out


def measure(out_path):
    params = load_params()
    os.makedirs(UNIT_ROOT, exist_ok=True)
    root = tempfile.mkdtemp(prefix="measure_", dir=UNIT_ROOT)
    result = {"configs": [
        measure_config(params, root, "dyn_gravity"),
        measure_config(params, root, "dyn_no_gravity", gravity=False),
        measure_config(params, root, "ideal_no_dynamics", dynamics=False),
    ], "gripper": measure_gripper(params, root)}
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="SO-101 simulated motion server with the real server file protocol")
    ap.add_argument("--run-root", default=RUN_ROOT)
    ap.add_argument("--no-dynamics", action="store_true")
    ap.add_argument("--no-gravity", action="store_true")
    ap.add_argument("--no-weld", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measure", metavar="OUT_JSON")
    a = ap.parse_args(argv)
    if a.measure:
        print(json.dumps(measure(a.measure)["gripper"]))
        return 0
    if os.path.realpath(a.run_root) == REAL_RUN:
        sys.stderr.write("refusing: run root is the real cell run directory\n")
        return 2
    srv = new_server(load_params(), a.run_root, dynamics=not a.no_dynamics, gravity=not a.no_gravity,
                     weld=not a.no_weld, seed=a.seed, virtual=False)
    pid_path = f"{srv.D}/pid"
    with open(pid_path, "w") as f:
        f.write(str(os.getpid()))

    def on_signal(*_):
        srv.stop = True

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        srv.run()
    finally:
        if os.path.exists(pid_path):
            os.remove(pid_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
