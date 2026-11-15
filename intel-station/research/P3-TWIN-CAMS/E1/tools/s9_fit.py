#!/usr/bin/env python3
"""P3-TWIN-CAMS E1 step 9: top and close camera fits from still-window jaw tips, leave-one-episode-out (LOEO), gate G3.

Provenance: nothing from R/twin is used at run time. The jaw geometry (out/jaw_geometry.json, step 3) was computed
at about 19:40Z from the Menagerie STL bytes that were then in R/twin/vendor; R/twin was deleted at 19:48Z, so this
script reads the saved geometry numbers, not the meshes (the file records the mesh md5s).
Observation model
- 3D features (base frame): fixed-jaw tip = lowest vertex of the wrist_roll_follower mesh (gripper frame, saved);
  moving-jaw tip = lowest vertex of the moving_jaw mesh at the rest pose (gripper frame, saved), mapped back into the
  moving-jaw link frame and rotated about the gripper joint axis by theta(g) = a + b*g (g = gripper normalized value
  of the window; a and b fitted). Both shifted by delta along the gripper z axis.
- FK: URDF chain base_link -> gripper_link (pick_v2 Chain, vectorised and checked against Chain.fk).
- Arm joints: rad = kin convention (CAL: sign +1, zero at mid-range), plus optional offsets (variant B only).
- 2D observations: bottom-most red pixel of each jaw blob (out/detect_<cam>.json, tip = [x, y]).
- Pairing per frame: two observations are matched to the two features in the order of smaller error; a one-jaw
  frame uses the feature with the smaller error.
- Camera: pinhole with R_base_cam (rotvec, columns = camera axes), centre C (m), fx, fy, cx, cy, plumb-bob k1, k2.
- Fit: multi-start (random look-at cameras x gripper-law seeds), stage 1 (camera, fx, fy, delta, a, b; principal
  point and distortion fixed), stage 2 (everything free), soft_l1 loss with f_scale 3 px.
- G3 error = pixel distance per observation, pooled over all held-out observations of the LOEO folds.
Run: envs/anomalib-env/bin/python -I tools/s9_fit.py <top|close> <A|B>   (A: no joint offsets; B: + 5 offsets)
"""
import json
import math
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

R = "/home/intel/intel-junction-2026"
E1 = f"{R}/work/P3-TWIN-CAMS/E1"
URDF = (f"{R}/repos/edge-developer-kit-reference-scripts/samples/ai/edge-ai-demo-studio/workers/services/"
        "robotics-ai/platforms/so101/urdf/so101.urdf")
sys.path.insert(0, f"{R}/app")
from pick_v2.kinematics import Chain, FollowerCalibration  # noqa: E402

CAL = FollowerCalibration.from_studio_json(f"{R}/app/motion/project_robots.json", "SO101_Follower")
CH = Chain(URDF, tip_link="gripper_link")
GR = CAL.joints["gripper"]
K_G = (GR.range_max - GR.range_min) / 100.0 * 2.0 * math.pi / 4095.0    # rad per gripper unit (calibration)
G = json.load(open(f"{E1}/out/jaw_geometry.json"))
PF = np.asarray(G["fixed_jaw_tip_gripper_mm"], float) / 1000.0
T_GJ = np.asarray(G["gripper_joint"]["origin_xyz_rpy_matrix"], float)     # gripper_link -> gripper joint, rest
AX = np.asarray(G["gripper_joint"]["axis"], float)
AX = AX / np.linalg.norm(AX)
PM_G0 = np.asarray(G["moving_jaw_tip_gripper_mm_at_theta0"], float) / 1000.0
PM_LINK = T_GJ[:3, :3].T @ (PM_G0 - T_GJ[:3, 3])                         # moving-jaw tip in its link frame
EZ = np.array([0.0, 0.0, 1.0])
JOINTS = CH.joints


def rot_axis(axis, th):
    th = np.asarray(th, float).reshape(-1)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    s = np.sin(th)[:, None, None]
    c = np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1.0 - c) * (K @ K)[None]


def fk_batch(Q):
    Q = np.asarray(Q, float)
    N = len(Q)
    T = np.tile(np.eye(4), (N, 1, 1))
    i = 0
    for j in JOINTS:
        T = T @ j.origin
        if j.kind == "revolute":
            Tr = np.tile(np.eye(4), (N, 1, 1))
            Tr[:, :3, :3] = rot_axis(j.axis, Q[:, i])
            T = T @ Tr
            i += 1
    return T


def moving_tip_grip(theta):
    v = np.einsum("nij,j->ni", rot_axis(AX, theta), PM_LINK)
    return (T_GJ[:3, :3] @ v.T).T + T_GJ[:3, 3]


def features(Tbg, theta, delta):
    N = len(Tbg)
    pf = np.broadcast_to(PF + delta * EZ, (N, 3))
    pm = moving_tip_grip(theta) + delta * EZ
    Rb, tb = Tbg[:, :3, :3], Tbg[:, :3, 3]
    return np.einsum("nij,nj->ni", Rb, pf) + tb, np.einsum("nij,nj->ni", Rb, pm) + tb


def project(P, x):
    Rc = Rotation.from_rotvec(x[0:3]).as_matrix()
    pc = (P - x[3:6]) @ Rc
    z = pc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    xn = pc[:, 0] / zs
    yn = pc[:, 1] / zs
    r2 = xn * xn + yn * yn
    d = 1.0 + x[10] * r2 + x[11] * r2 * r2
    uv = np.stack([x[6] * xn * d + x[8], x[7] * yn * d + x[9]], axis=1)
    uv[z <= 1e-3] = 1e4
    return uv


def predict(x, D):
    off = x[15:20]
    Tbg = fk_batch(D["Q"] + off[None, :]) if np.any(off != 0.0) else D["Tbg0"]
    theta = x[13] + x[14] * D["g"]
    Pf, Pm = features(Tbg, theta, x[12])
    return project(Pf, x), project(Pm, x)


def pair(uf, um, D):
    A, B, two = D["A"], D["B"], D["two"]
    e1 = np.hstack([uf - A, um - B])
    e2 = np.hstack([uf - B, um - A])
    use1 = np.sum(e1 * e1, axis=1) <= np.sum(e2 * e2, axis=1)
    e_two = np.where(use1[:, None], e1, e2)[two]
    sf, sm = uf - A, um - A
    usef = np.sum(sf * sf, axis=1) <= np.sum(sm * sm, axis=1)
    e_one = np.where(usef[:, None], sf, sm)[~two]
    return e_two, e_one


def residuals(x, D):
    uf, um = predict(x, D)
    e_two, e_one = pair(uf, um, D)
    prior_off = 3.0 * x[15:20] / math.radians(3.0)
    prior_g = np.array([3.0 * ((x[14] / K_G) - 1.0) / 0.5])
    return np.concatenate([e_two.ravel(), e_one.ravel(), prior_off, prior_g])


def distances(x, D):
    uf, um = predict(x, D)
    e_two, e_one = pair(uf, um, D)
    dt = np.stack([np.hypot(e_two[:, 0], e_two[:, 1]), np.hypot(e_two[:, 2], e_two[:, 3])], axis=1)
    do = np.hypot(e_one[:, 0], e_one[:, 1])
    return dt, do


LB = np.array([-np.pi] * 3 + [-1.5, -1.5, 0.05] + [300, 300, 160, 120, -1.0, -1.0, -0.04, -np.pi, -0.08] + [-0.2] * 5)
UB = np.array([np.pi] * 3 + [1.5, 1.5, 2.5] + [1500, 1500, 480, 360, 1.0, 1.0, 0.04, np.pi, 0.08] + [0.2] * 5)
FREE1 = [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14]
FREE2 = list(range(15))
FREE2B = list(range(20))
A_GRID = [-0.17, 0.0, 1.0]
B_GRID = [0.02, -0.02]


def run_ls(x0, free, D):
    x0 = np.clip(np.asarray(x0, float), LB + 1e-9, UB - 1e-9)
    idx = np.asarray(free)

    def fun(p):
        x = x0.copy()
        x[idx] = p
        return residuals(x, D)

    r = least_squares(fun, x0[idx], bounds=(LB[idx], UB[idx]), loss="soft_l1", f_scale=3.0, x_scale="jac",
                      max_nfev=400)
    x = x0.copy()
    x[idx] = r.x
    dt, do = distances(x, D)
    return x, float(np.median(np.concatenate([dt.ravel(), do])))


def look_at(C, c0, hint):
    z = c0 - C
    z = z / np.linalg.norm(z)
    y = hint - np.dot(hint, z) * z
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def start_pose(rng, c0):
    while True:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        C = c0 + rng.uniform(0.3, 1.2) * d
        if C[2] > 0.05:
            break
    Rc = look_at(C, c0, rng.normal(size=3))
    return np.r_[Rotation.from_matrix(Rc).as_rotvec(), C]


def fit_camera(D, n_cam, seed, offsets, top_k=4):
    rng = np.random.default_rng(seed)
    Pf0, _ = features(D["Tbg0"], np.zeros(len(D["g"])), 0.0)
    c0 = Pf0.mean(axis=0)
    cands = []
    for _ in range(n_cam):
        pose = start_pose(rng, c0)
        f0 = rng.uniform(450.0, 800.0)
        for a in A_GRID:
            for b in B_GRID:
                x0 = np.zeros(20)
                x0[0:6] = pose
                x0[6] = x0[7] = f0
                x0[8], x0[9] = 320.0, 240.0
                x0[13], x0[14] = a, b
                x1, m1 = run_ls(x0, FREE1, D)
                cands.append((m1, x1))
    cands.sort(key=lambda t: t[0])
    res = []
    for m1, x1 in cands[:top_k]:
        x2, m2 = run_ls(x1, FREE2B if offsets else FREE2, D)
        res.append((m2, x2, m1))
    res.sort(key=lambda t: t[0])
    starts = [{"stage2_train_median_px": round(m2, 3), "stage1_train_median_px": round(m1, 3)} for m2, _, m1 in res]
    return res[0][1], res[0][0], starts


def subset(D, m):
    N = len(m)
    return {k: (v[m] if isinstance(v, np.ndarray) and v.shape[0] == N else v) for k, v in D.items()}


def stats(v):
    v = np.asarray(v, float).ravel()
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "median_px": round(float(np.median(v)), 3),
            "p90_px": round(float(np.percentile(v, 90)), 3), "p95_px": round(float(np.percentile(v, 95)), 3),
            "rms_px": round(float(np.sqrt(np.mean(v * v))), 3), "max_px": round(float(np.max(v)), 3)}


def describe(x):
    Rc = Rotation.from_rotvec(x[0:3]).as_matrix()
    tilt = math.degrees(math.acos(float(np.clip(-Rc[2, 2], -1.0, 1.0))))
    return {"R_base_cam_columns_are_camera_axes": np.round(Rc, 6).tolist(), "rotvec": np.round(x[0:3], 6).tolist(),
            "C_base_m": np.round(x[3:6], 5).tolist(), "fx": round(float(x[6]), 3), "fy": round(float(x[7]), 3),
            "cx": round(float(x[8]), 3), "cy": round(float(x[9]), 3), "k1": round(float(x[10]), 6),
            "k2": round(float(x[11]), 6), "delta_m": round(float(x[12]), 6), "delta_mm": round(float(x[12]) * 1000, 3),
            "gripper_a_rad": round(float(x[13]), 6), "gripper_b_rad_per_unit": round(float(x[14]), 6),
            "gripper_b_over_calibration": round(float(x[14] / K_G), 4),
            "joint_offsets_rad": np.round(x[15:20], 6).tolist(),
            "optical_axis_tilt_from_straight_down_deg": round(tilt, 2)}


def load(cam):
    rows = [r for r in json.load(open(f"{E1}/out/detect_{cam}.json")) if r.get("jaws")]
    A = np.array([r["jaws"][0]["tip"] for r in rows], float)
    two = np.array([len(r["jaws"]) == 2 for r in rows])
    B = np.array([r["jaws"][1]["tip"] if len(r["jaws"]) == 2 else r["jaws"][0]["tip"] for r in rows], float)
    P = np.array([r["pose"] for r in rows], float)
    Q = CAL.sign * CAL.k * P[:, :5] + CAL.zero
    D = {"A": A, "B": B, "two": two, "g": P[:, 5], "Q": Q, "pose": P,
         "ep": np.array([str(r["episode"]) for r in rows]), "win": np.array([int(r["win"]) for r in rows])}
    D["Tbg0"] = fk_batch(Q)
    return D


def main(cam, tag):
    offsets = tag == "B"
    n_cam_loeo = 8
    n_cam_full = 40
    D = load(cam)
    for i in range(3):
        ref = CH.fk(D["Q"][i])
        assert np.max(np.abs(ref - D["Tbg0"][i])) < 1e-9, "vectorised FK differs from Chain.fk"
    assert np.allclose(moving_tip_grip(np.zeros(1))[0], PM_G0, atol=1e-6), "moving tip reconstruction mismatch"
    print(cam, tag, "frames", len(D["g"]), "two-jaw", int(D["two"].sum()), "episodes", len(set(D["ep"].tolist())),
          "| fixed tip mm", np.round(PF * 1000, 2).tolist(), "| moving tip at theta0 mm",
          np.round(PM_G0 * 1000, 2).tolist(), flush=True)
    t0 = time.time()
    eps = sorted(set(D["ep"].tolist()))
    per, dt_all, do_all, rows = {}, [], [], []
    for k, e in enumerate(eps):
        te = D["ep"] == e
        Dtr, Dte = subset(D, ~te), subset(D, te)
        x, med_tr, starts = fit_camera(Dtr, n_cam_loeo, seed=100 + k, offsets=offsets)
        dt, do = distances(x, Dte)
        per[e] = {"windows": int(te.sum()), "obs": int(dt.size + do.size), "train_median_px": round(med_tr, 3),
                  "test": stats(np.concatenate([dt.ravel(), do])), "fit_starts": starts,
                  "params_fold": describe(x)}
        dt_all.append(dt.ravel())
        do_all.append(do)
        for w, ee, d in zip(Dte["win"][Dte["two"]], Dte["ep"][Dte["two"]], dt):
            rows.append({"ep": str(ee), "win": int(w), "kind": "two", "d_fixed_px": round(float(d[0]), 3),
                         "d_moving_px": round(float(d[1]), 3)})
        for w, ee, d in zip(Dte["win"][~Dte["two"]], Dte["ep"][~Dte["two"]], do):
            rows.append({"ep": str(ee), "win": int(w), "kind": "one", "d_px": round(float(d), 3)})
        print(cam, tag, "fold", e, "test median", per[e]["test"].get("median_px"), "n", per[e]["test"]["n"],
              "train median", per[e]["train_median_px"], "t", round(time.time() - t0), flush=True)
    dt_pool = np.concatenate(dt_all)
    do_pool = np.concatenate(do_all)
    pooled = stats(np.concatenate([dt_pool, do_pool]))
    g3 = {"median_le_3px": bool(pooled["median_px"] <= 3.0), "p90_le_6px": bool(pooled["p90_px"] <= 6.0)}
    g3["pass"] = bool(g3["median_le_3px"] and g3["p90_le_6px"])
    x_full, med_full, starts_full = fit_camera(D, n_cam_full, seed=7, offsets=offsets, top_k=6)
    dt_in, do_in = distances(x_full, D)
    out = {"camera": cam, "variant": tag,
           "joint_convention": "kin (CAL sign +1, zero at mid-range)" +
           (" + 5 free joint offsets (prior 3 deg = 3 px)" if offsets else ""),
           "features": {"fixed_tip_gripper_mm": np.round(PF * 1000, 3).tolist(),
                        "moving_tip_at_rest_gripper_frame_mm": np.round(PM_G0 * 1000, 3).tolist(),
                        "geometry_source": "out/jaw_geometry.json (step 3; mesh md5s recorded there)",
                        "observation": "bottom-most red pixel of each jaw blob (detect_<cam>.json tip)"},
           "frames": int(len(D["g"])), "two_jaw_frames": int(D["two"].sum()), "episodes": len(eps),
           "observations": int(dt_pool.size + do_pool.size),
           "G3_loeo_pooled_all_observations": pooled, "G3_loeo_two_jaw_observations": stats(dt_pool),
           "G3_loeo_one_jaw_observations": stats(do_pool), "G3": g3,
           "full_fit": {"params": describe(x_full), "train_median_px": round(med_full, 3),
                        "in_sample": stats(np.concatenate([dt_in.ravel(), do_in])), "starts": starts_full},
           "loeo_per_episode": per, "loeo_rows": rows}
    with open(f"{E1}/out/fit_{cam}_{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    np.save(f"{E1}/out/fit_{cam}_{tag}_xfull.npy", x_full)
    print(cam, tag, "LOEO pooled", pooled, "G3", g3, flush=True)
    print(cam, tag, "in-sample", out["full_fit"]["in_sample"], "train median", round(med_full, 3), flush=True)
    print(cam, tag, "full params", json.dumps(out["full_fit"]["params"]), flush=True)
    print(cam, tag, "done in", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
