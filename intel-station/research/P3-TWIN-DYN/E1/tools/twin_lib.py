#!/usr/bin/env python3
"""P3-TWIN-DYN E1 library: SO-101 joint dynamics as MuJoCo position actuators, fitted to joints.csv logs.

Model per joint j (hinge joint, no joint limits, gravity and link inertias from the SO-101 URDF):
  MuJoCo <position> actuator: force = clip(kp*(ctrl - q) - kv*qdot, -F, F), F fixed at 10 N.m (URDF effort limit)
  <joint armature=... frictionloss=...>
  ctrl = k_j * logged goal, delayed by D_j, zero-order hold between logged samples (k_j = rad per normalized unit)
Identified per joint: kp (N.m/rad), kv (N.m.s/rad), frictionloss (N.m), armature (kg.m^2), delay D (s).
Parameter array p has shape (6, 6) with columns [kp, kv, frictionloss, armature, forcerange, delay].
"""
import csv
import glob
import json
import math
import os
import re

import mujoco
import numpy as np
from mujoco import rollout as mjro
from scipy.optimize import least_squares

R = "/home/intel/intel-junction-2026"
URDF = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
        "robotics-ai/platforms/so101/urdf/so101.urdf")
NOGEOM_URDF = f"{R}/work/P3-TWIN-DYN/E1/urdf/so101_nogeom.urdf"
BASE_XML = f"{R}/work/P3-TWIN-DYN/E1/urdf/so101_base_mjcf.xml"
PROJECT_ROBOTS = f"{R}/app/motion/project_robots.json"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PARAM_NAMES = ["kp", "kv", "frictionloss", "armature", "forcerange", "delay"]
DT = 0.004  # fit and final model use 4 ms steps: nominal error changes < 0.003 units vs 2 ms (dtcheck), half the time
TICKS_PER_REV = 4095.0
NTHREAD = int(os.environ.get("TWIN_NTHREAD", "8"))
FORCE_FIXED = 10.0  # N.m, URDF effort limit (MuJoCo URDF import gives actuatorfrcrange -10 10); not identified
FREE = [0, 1, 2, 3, 5]  # identified columns: kp, kv, frictionloss, armature, delay
GRIP_CONTACT_POS = 3.0  # gripper rows with pos5 and goal5 both below this are brick-contact rows (masked)
INIT = np.array([
    [3.0, 0.05, 0.05, 0.0005, FORCE_FIXED, 0.12],
    [6.0, 0.20, 0.20, 0.002, FORCE_FIXED, 0.12],
    [5.0, 0.20, 0.20, 0.002, FORCE_FIXED, 0.12],
    [3.0, 0.05, 0.05, 0.0002, FORCE_FIXED, 0.12],
    [2.0, 0.02, 0.02, 0.00005, FORCE_FIXED, 0.12],
    [2.0, 0.05, 0.05, 0.00005, FORCE_FIXED, 0.12],
])
# per-joint bounds for the FREE columns [kp, kv, frictionloss, armature, delay]; kp caps keep the explicit position
# spring stable at DT (omega*DT < 2 for the small wrist and gripper inertia)
LO_MAT = np.array([[0.5, 1e-3, 1e-4, 1e-5, 0.002]] * 6)
HI_MAT = np.array([[20.0, 5.0, 2.0, 0.02, 0.3], [20.0, 5.0, 2.0, 0.02, 0.3], [20.0, 5.0, 2.0, 0.02, 0.3],
                   [10.0, 5.0, 2.0, 0.02, 0.3], [5.0, 5.0, 2.0, 0.02, 0.3], [5.0, 5.0, 2.0, 0.02, 0.3]])


def calib_k():
    """Radians per normalized unit for the six joints (kinematics.py: width/NORM_SPAN*2pi/TICKS_PER_REV)."""
    entries = json.load(open(PROJECT_ROBOTS))
    e = next(x for x in entries if x["type"] == "SO101_Follower")
    cal = e["payload"]["calibration"]
    k = []
    for name in JOINTS:
        lo, hi = float(cal[name]["range_min"]), float(cal[name]["range_max"])
        span = 100.0 if name == "gripper" else 200.0
        k.append((hi - lo) / span * 2.0 * math.pi / TICKS_PER_REV)
    return np.array(k)


def load_episodes():
    """All logs/sort/any_17*/ep_*/joints.csv (t, goal0..5, pos0..5); t is relative to the first row."""
    out = []
    for f in sorted(glob.glob(f"{R}/logs/sort/any_17*/ep_*/joints.csv")):
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            header = next(rd)
            rows = [r for r in rd if r and len(r) == len(header)]
        a = np.array(rows, dtype=float)
        out.append({"name": "/".join(f.split("/")[-3:-1]), "t": a[:, 0] - a[0, 0],
                    "goal": a[:, 1:7], "pos": a[:, 7:13]})
    return out


def base_xml():
    if not os.path.exists(BASE_XML):
        m0 = mujoco.MjModel.from_xml_path(NOGEOM_URDF)
        mujoco.mj_saveLastXML(BASE_XML, m0)
    return open(BASE_XML).read()


def make_xml(p):
    """MJCF text for parameter array p (6 x 6). The body tree is the URDF tree as saved by MuJoCo."""
    s = base_xml()
    s = s.replace('<compiler angle="radian"/>',
                  '<compiler angle="radian"/>\n  <option timestep="%g" integrator="implicitfast" '
                  'gravity="0 0 -9.81"/>' % DT, 1)
    for j, name in enumerate(JOINTS):
        arm, fr = p[j, 3], p[j, 2]
        new_joint = '<joint name="%s" limited="false" armature="%.9g" frictionloss="%.9g" damping="0"/>' % (
            name, arm, fr)
        s, n = re.subn(r'<joint name="%s"[^>]*/>' % name, new_joint, s)
        assert n == 1, name
    acts = "\n".join(
        '    <position name="act_%s" joint="%s" kp="%.9g" kv="%.9g" forcelimited="true" forcerange="%.9g %.9g"/>'
        % (name, name, p[j, 0], p[j, 1], -p[j, 4], p[j, 4]) for j, name in enumerate(JOINTS))
    s = s.replace("</mujoco>", "  <actuator>\n" + acts + "\n  </actuator>\n</mujoco>")
    return s


def nstep_for(eps):
    return int(math.ceil(max(ep["t"][-1] for ep in eps) / DT)) + 2


def build_inputs(eps, p, nstep):
    """Initial states [time, qpos(6), qvel(6)] (rad) and per-step ctrl (rad) for every episode in eps."""
    k = calib_k()
    tau = np.arange(nstep) * DT
    ne = len(eps)
    ctrl = np.zeros((ne, nstep, 6))
    init = np.zeros((ne, 13))
    for e, ep in enumerate(eps):
        for j in range(6):
            idx = np.searchsorted(ep["t"], tau - p[j, 5], side="right") - 1
            idx = np.clip(idx, 0, len(ep["t"]) - 1)
            ctrl[e, :, j] = k[j] * ep["goal"][idx, j]
        init[e, 1:7] = k * ep["pos"][0]
    return init, ctrl


def groups_by_length(eps, n_groups=3):
    order = np.argsort([ep["t"][-1] for ep in eps], kind="stable")
    return [[int(i) for i in c] for c in np.array_split(order, n_groups) if len(c)]


def simulate(eps, p, ro):
    """Roll out every episode with one parameter set. Episodes are batched by length (avoids padding all to the
    longest). Returns a list with one array per episode, shape (nstep_e, 13): [time, qpos(6), qvel(6)] after
    each step."""
    m = mujoco.MjModel.from_xml_string(make_xml(p))
    datas = [mujoco.MjData(m) for _ in range(NTHREAD)]
    out = [None] * len(eps)
    for g in groups_by_length(eps):
        sub = [eps[i] for i in g]
        ns = nstep_for(sub)
        init, ctrl = build_inputs(sub, p, ns)
        st, _ = ro.rollout(m, datas, init, ctrl, nstep=ns)
        for local, i in enumerate(g):
            out[i] = st[local]
    return out


def mask_for(ep, j):
    if j == 5:
        return ~((ep["pos"][:, 5] < GRIP_CONTACT_POS) & (ep["goal"][:, 5] < GRIP_CONTACT_POS))
    return np.ones(len(ep["t"]), dtype=bool)


def errors_per_episode(eps, states):
    """Normalized-unit errors (sim - logged) per episode and joint, at the logged sample times."""
    k = calib_k()
    out = []
    for e, ep in enumerate(eps):
        st = states[e]
        ts = st[:, 0]
        errs = []
        for j in range(6):
            q = np.interp(ep["t"], ts, st[:, 1 + j])
            errs.append((q - ep["pos"][:, j] * k[j]) / k[j])
        out.append(errs)
    return out


def residual_vector(eps, states):
    errs = errors_per_episode(eps, states)
    parts = []
    for e, ep in enumerate(eps):
        for j in range(6):
            parts.append(errs[e][j][mask_for(ep, j)])
    return np.concatenate(parts)


def unpack(z):
    p = np.zeros((6, 6))
    p[:, 4] = FORCE_FIXED
    p[:, FREE] = np.exp(np.asarray(z)).reshape(6, len(FREE))
    return p


def pack(p):
    return np.log(p[:, FREE]).ravel()


def fit(eps, max_nfev=25, z0=None, verbose=0):
    """Least-squares fit of the 30 log-parameters on the given episodes, from the fixed INIT unless z0 is given."""
    lo = np.log(LO_MAT).ravel()
    hi = np.log(HI_MAT).ravel()
    z0 = pack(INIT) if z0 is None else z0
    ro = mjro.Rollout(nthread=NTHREAD)

    def fun(z):
        return residual_vector(eps, simulate(eps, unpack(z), ro))

    res = least_squares(fun, z0, bounds=(lo, hi), method="trf", x_scale="jac", diff_step=1e-3,
                        max_nfev=max_nfev, verbose=verbose)
    return unpack(res.x), res
