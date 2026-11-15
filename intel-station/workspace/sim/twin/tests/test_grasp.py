#!/usr/bin/env python3
"""P3-TWIN-ROBOT E3: graspable twin tests on sim/twin/robot/so101_twin_v3.xml.

2x2 brick (31.7 x 31.7 x 19.2 mm; mass upper bound and friction from scene/scene_measurements.json) on the mat, the
sort_any.grasp sequence (hover, descend to 26 mm, descend to 8 mm above the mat, close, lift 45 mm), the empty-close
reading, 10-seed contact stability, normal contact force logging, the jaw-gap table, and G1 on the candidate (0.5 mm).
The arm uses integral correction of the position ctrl so that gravity sag does not leave the TCP off the target.
Plain test_* functions with a __main__ runner (pytest is not installed in sim-env).
Run: sim-env/bin/python -I -B test_grasp.py
"""
import json
import math
import os
import sys
import tempfile

import mujoco
import numpy as np

R = "/home/intel/intel-junction-2026"
ROBOT = f"{R}/sim/twin/robot"
MESHES = f"{ROBOT}/meshes"
SRC = f"{ROBOT}/so101_twin_v3.xml"     # tools may point this at a temporary candidate
SCENE = f"{R}/sim/twin/scene/scene_measurements.json"
G1 = f"{R}/sim/twin/tests/data/g1_fixed_sample.json"
OUT = f"{R}/work/P3-TWIN-ROBOT/E3/outputs/grasp_results.json"
sys.path.insert(0, ROBOT)
import tcp  # noqa: E402
import units  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LAW = "closed_at_empty_reading"
MAT_Z = -0.0383          # kin.MAT_Z (m): TCP z when touching the mat
HOVER_DZ = 0.0577        # kin.HOVER_DZ
TIP_TARGET = 0.026       # sort_any.grasp: first descent, 26 mm above the mat
TIP_FINAL = 0.008        # sort_any.grasp: final descent, 8 mm above the mat
LIFT = 0.045             # sort_any.grasp: lift 45 mm
BRICK_HALF = (0.01585, 0.01585, 0.0096)   # 31.7 x 31.7 x 19.2 mm
MAT_HALF = 0.6
BRICK_XY = (0.17, 0.0)
Z_DOWN = np.array([0.0, 0.0, -1.0])       # assumed approach axis = TCP site z
JAW_BODIES = {"gripper_link", "moving_jaw_so101_v1_link"}
JAW_FIXED_MESH = "wrist_roll_follower_so101_v1"
JAW_MOVING_MESH = "moving_jaw_so101_v1"


def _bricks():
    return json.load(open(SCENE))["bricks"]


def build_model(brick_xy=None, yaw=0.0):
    b = _bricks()
    mass = b["mass_upper_bound_g"] / 1000.0
    mu = b["sliding_friction"]
    txt = open(SRC).read().replace('meshdir="meshes"', f'meshdir="{MESHES}"')
    extra = (f'<geom name="mat" type="box" size="{MAT_HALF} {MAT_HALF} 0.005" pos="0 0 {MAT_Z - 0.005:.6f}" '
             f'friction="{mu} 0.005 0.0001" contype="1" conaffinity="1" rgba="0.3 0.3 0.3 1"/>')
    if brick_xy is not None:
        extra += (f'<body name="brick" pos="{brick_xy[0]} {brick_xy[1]} {MAT_Z + BRICK_HALF[2]:.6f}" '
                  f'euler="0 0 {yaw}"><freejoint name="brick_free"/>'
                  f'<geom name="brick_geom" type="box" size="{BRICK_HALF[0]} {BRICK_HALF[1]} {BRICK_HALF[2]}" '
                  f'mass="{mass}" friction="{mu} 0.005 0.0001" contype="1" conaffinity="1" rgba="0.9 0.2 0.2 1"/>'
                  f'</body>')
    txt = txt.replace("</worldbody>", extra + "</worldbody>")
    path = os.path.join(tempfile.mkdtemp(prefix="twin_grasp_"), "scene.xml")
    with open(path, "w") as fh:
        fh.write(txt)
    return mujoco.MjModel.from_xml_path(path)


class Rig:
    def __init__(self, m):
        self.m = m
        self.d = mujoco.MjData(m)
        self.dk = mujoco.MjData(m)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in JOINTS]
        self.qadr = [int(m.jnt_qposadr[j]) for j in jid]
        self.dofs = [int(m.jnt_dofadr[j]) for j in jid]
        self.lo = np.array([m.jnt_range[j][0] for j in jid[:5]])
        self.hi = np.array([m.jnt_range[j][1] for j in jid[:5]])
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripper_frame_link")
        self.brick = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "brick")
        mujoco.mj_forward(m, self.d)


def gripper_ctrl(n):
    return units.gripper_norm_to_rad(n, LAW)


def gripper_reading(rig):
    return float(units.gripper_rad_to_norm(rig.d.qpos[rig.qadr[5]], LAW))


def ik(rig, target, q0, iters=400):
    """Damped least squares on TCP position and approach axis (site z = Z_DOWN), starting from q0."""
    m, dk = rig.m, rig.dk
    q = np.array(q0, dtype=float)
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    ep = er = np.zeros(3)
    for _ in range(iters):
        for a, v in zip(rig.qadr[:5], q):
            dk.qpos[a] = v
        mujoco.mj_fwdPosition(m, dk)
        p = dk.site_xpos[rig.site]
        zc = dk.site_xmat[rig.site].reshape(3, 3)[:, 2]
        ep = target - p
        er = np.cross(zc, Z_DOWN)
        if np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3:
            break
        mujoco.mj_jacSite(m, dk, jp, jr, rig.site)
        J = np.vstack([jp[:, rig.dofs[:5]], jr[:, rig.dofs[:5]]])
        e = np.concatenate([ep, 0.5 * er])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
        q = np.clip(q + 0.5 * dq, rig.lo, rig.hi)
    return q, float(np.linalg.norm(ep) * 1000.0), float(np.linalg.norm(er))


def settle(rig, q, g, seconds):
    rig.d.ctrl[:5] = q
    rig.d.ctrl[5] = g
    for _ in range(int(round(seconds / rig.m.opt.timestep))):
        mujoco.mj_step(rig.m, rig.d)
    mujoco.mj_forward(rig.m, rig.d)


def goto(rig, target, q_seed, g, seconds=1.2, iters=5):
    """IK target, then integral correction of the ctrl until the simulated TCP is within 0.5 mm of the target."""
    qn, perr, oerr = ik(rig, np.asarray(target, dtype=float), q_seed)
    ctrl = qn.copy()
    for _ in range(iters):
        settle(rig, ctrl, g, seconds)
        q_act = np.array([rig.d.qpos[a] for a in rig.qadr[:5]])
        tcp_now = rig.d.site_xpos[rig.site].copy()
        if np.linalg.norm(tcp_now - target) < 0.0005:
            break
        ctrl = ctrl + (qn - q_act)
    tcp_now = rig.d.site_xpos[rig.site].copy()
    return ctrl, {"target_z_above_mat_mm": round((target[2] - MAT_Z) * 1000, 2),
                  "tcp_z_above_mat_mm": round((tcp_now[2] - MAT_Z) * 1000, 2),
                  "tcp_err_mm": round(float(np.linalg.norm(tcp_now - target)) * 1000, 3),
                  "ik_pos_mm": round(perr, 4), "ik_orient_resid": round(oerr, 5)}


def brick_jaw_forces(rig):
    """Count, total and max normal force (N) of brick-jaw contacts (mj_contactForce, frame normal first)."""
    m, d = rig.m, rig.d
    f = np.zeros(6)
    n, total, fmax = 0, 0.0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])),
                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2]))}
        if "brick" in names and names & JAW_BODIES:
            mujoco.mj_contactForce(m, d, i, f)
            n += 1
            total += abs(f[0])
            fmax = max(fmax, abs(f[0]))
    return {"contacts": n, "normal_sum_N": round(total, 4), "normal_max_N": round(fmax, 4)}


def run_sequence(xy=BRICK_XY, yaw=0.0, brick=True):
    """Hover, descend to 26 mm, descend to 8 mm, close (1.0 s), lift 45 mm (sort_any.grasp heights)."""
    m = build_model(xy if brick else None, yaw)
    rig = Rig(m)
    d = rig.d
    bx, by = xy
    g_open, g_closed = gripper_ctrl(25.0), gripper_ctrl(0.0)
    stages = {}
    q = np.zeros(5)
    settle(rig, q, g_open, 1.0)
    q, stages["hover"] = goto(rig, [bx, by, MAT_Z + HOVER_DZ], q, g_open)
    # centre the jaw midpoint (not the TCP site, which sits over the fixed jaw) over the brick
    fid, mid_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "pad_fixed"), mujoco.mj_name2id(
        m, mujoco.mjtObj.mjOBJ_GEOM, "pad_moving")
    if fid >= 0 and mid_id >= 0 and brick:
        mid = 0.5 * (d.geom_xpos[fid] + d.geom_xpos[mid_id])
        bx, by = bx + (xy[0] - mid[0]), by + (xy[1] - mid[1])
        q, stages["hover_jaw_centred"] = goto(rig, [bx, by, MAT_Z + HOVER_DZ], q, g_open)
    q, stages["descend_26mm"] = goto(rig, [bx, by, MAT_Z + TIP_TARGET], q, g_open)
    q, stages["descend_8mm"] = goto(rig, [bx, by, MAT_Z + TIP_FINAL], q, g_open)
    z0 = float(d.xpos[rig.brick][2]) if brick else None
    settle(rig, q, g_closed, 1.0)
    out = {"brick": brick, "xy_m": list(xy), "yaw_rad": yaw, "stages": stages,
           "reading_after_close": round(gripper_reading(rig), 4),
           "closed_contacts": brick_jaw_forces(rig) if brick else None}
    q, stages["lift_45mm"] = goto(rig, [bx, by, MAT_Z + TIP_FINAL + LIFT], q, g_closed, seconds=1.5)
    out["gripper_rad_after_lift"] = round(float(d.qpos[rig.qadr[5]]), 5)
    out["reading_after_lift"] = round(gripper_reading(rig), 4)
    if brick:
        bz = float(d.xpos[rig.brick][2])
        ref = 0.5 * (d.geom_xpos[fid] + d.geom_xpos[mid_id]) if (fid >= 0 and mid_id >= 0) else d.site_xpos[rig.site]
        dxy = float(np.linalg.norm(d.xpos[rig.brick][:2] - ref[:2]))
        forces = brick_jaw_forces(rig)
        out.update({"brick_lift_mm": round((bz - z0) * 1000, 2), "tcp_brick_xy_mm": round(dxy * 1000, 2),
                    "lift_contacts": forces})
        out["held"] = bool((bz - z0) > 0.03 and dxy < 0.02 and forces["contacts"] > 0)
    else:
        out["held"] = False
    return out


def geom_by_mesh(m, mesh_name):
    for g in range(m.ngeom):
        if m.geom_group[g] == 3 and m.geom_dataid[g] >= 0:
            if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, int(m.geom_dataid[g])) == mesh_name:
                return g
    raise KeyError(mesh_name)


def jaw_gap_table(readings):
    """Hull-to-hull distance (mm) between the fixed jaw mesh and the moving jaw mesh at each reading. Negative means
    the hulls overlap. The moving jaw centre is logged so a constant distance can be checked against motion."""
    m = build_model(None)
    rig = Rig(m)
    d = rig.d
    gf = geom_by_mesh(m, JAW_FIXED_MESH)
    gm = geom_by_mesh(m, JAW_MOVING_MESH)
    rows = []
    for n in readings:
        q = units.gripper_norm_to_rad(n, LAW)
        d.qpos[rig.qadr[5]] = q
        mujoco.mj_forward(m, d)
        fromto = np.zeros(6)
        dist = mujoco.mj_geomDistance(m, d, gf, gm, 0.5, fromto)
        rows.append({"reading": float(n), "q_rad": round(float(q), 5), "hull_gap_mm": round(dist * 1000, 2),
                     "moving_jaw_centre_mm": [round(float(x) * 1000, 2) for x in d.geom_xpos[gm]]})
    return rows


def _g1_errors(path=None):
    s = json.load(open(G1))["samples"]
    jn = np.array([x["joints_norm"] for x in s], dtype=float)
    kp = np.array([x["kin_tcp_mm"] for x in s], dtype=float)
    kr = np.array([x["kin_R"] for x in s], dtype=float).reshape(-1, 3, 3)
    arm = tcp.ArmModel(path or SRC)
    p, rot = tcp.tcp_batch(arm, jn)
    pos, ori = tcp.pose_errors(p, rot, kp, kr)
    return float(np.max(pos)), float(np.max(ori)), len(s)


def seed_case(seed):
    rng = np.random.default_rng(seed)
    xy = (BRICK_XY[0] + rng.uniform(-0.0015, 0.0015), BRICK_XY[1] + rng.uniform(-0.0015, 0.0015))
    return xy, rng.uniform(-5, 5) * math.pi / 180.0


# ---- tests ----

def test_g1_candidate_fixed_sample():
    pos, ori, n = _g1_errors()
    assert pos < 0.5, f"G1 max position error {pos} mm over {n} samples"


def test_empty_close_reads_empty_value():
    r = run_sequence(brick=False)
    assert 0.50 <= r["reading_after_close"] <= 0.60, r


def test_brick_grasp_hold_nominal():
    r = run_sequence()
    assert r["held"], r


def test_contact_stability_10_seeds():
    rows = [run_sequence(*seed_case(s)) for s in range(10)]
    held = sum(1 for r in rows if r["held"])
    assert held == 10, f"held {held}/10"


def test_jaw_gap_is_monotone():
    rows = jaw_gap_table([0.52, 1.57, 2.29, 24.6])
    gaps = [r["hull_gap_mm"] for r in rows]
    assert all(b > a for a, b in zip(gaps, gaps[1:])), rows


def collect():
    res = {"src": SRC, "empty": run_sequence(brick=False), "nominal": run_sequence(), "seeds": [],
           "jaw_gap": jaw_gap_table([0.52, 1.0, 1.57, 2.29, 5.0, 10.0, 15.0, 20.0, 24.6])}
    pos, ori, n = _g1_errors()
    res["g1"] = {"n": n, "max_pos_mm": pos, "max_ori_deg": ori}
    for s in range(10):
        r = run_sequence(*seed_case(s))
        res["seeds"].append({"seed": s, "held": r["held"], "brick_lift_mm": r.get("brick_lift_mm"),
                             "reading_after_lift": r["reading_after_lift"],
                             "lift_contacts": r.get("lift_contacts")})
    return res


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except AssertionError as exc:
                fails += 1
                print("FAIL", name, str(exc)[:800])
    res = collect()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote", OUT, "failures:", fails)
    sys.exit(1 if fails else 0)
