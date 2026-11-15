# V1 verifier: fit top-view pixel -> table XY from verified grasp pairs, LOO error, known-pose check, offline pick validation.
import json, math, sys
import numpy as np
sys.path.insert(0, "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1/tools")
from extract import load_urdf, chain_to, fk, q_of, URDF, TCP, CAL, BODY  # noqa: E402

OUT = "/home/intel/intel-junction-2026/work/P2-R8-PICK/V1"
# Manual labels: centre of the grasped brick in the START top view (visual, zoom sheets), presence-checked at 1 s after grasp.
# Accepted set and reasons for exclusion are written to calib.json.
LABELS = {0: (402, 170), 1: (419, 161), 2: (389, 152), 3: (405, 136), 5: (402, 172), 6: (361, 221), 7: (388, 141), 14: (423, 209)}
EXCLUDED = {4: "two candidates (yellow 400,128 and pink 383,190) both vanish at lift; cannot assign",
            11: "white 2x2 and red 2x2 both partly hidden at lift; label not verifiable",
            12: "white 2x4 and orange 2x2 both vanish at lift; occlusion by arm, label not verifiable",
            15: "yellow and white candidates partly occluded by cables at lift; label not verifiable",
            9: "gripper closed on nothing (level 0.98)", 8: "start holding brick, no grasp in episode",
            10: "start holding brick, no grasp in episode", 13: "gripper closes only to about 11, grasp missed"}

rows = json.load(open(OUT + "/episodes.json"))
R = {r["ep"]: r for r in rows["rows"] if r["ds"] == "6d6"}
joints = load_urdf(URDF)
chain = chain_to(joints, TCP)


def fk5(u5):
    return fk(joints, chain, q_of(np.asarray(u5, float)))


def fit_h(px, xy):
    P = np.array([[(u - 320) / 320.0, (v - 240) / 240.0] for u, v in px])
    Q = np.array([[X / 0.2, Y / 0.2] for X, Y in xy])
    A = []
    for (u, v), (X, Y) in zip(P, Q):
        A.append([u, v, 1, 0, 0, 0, -X * u, -X * v, -X])
        A.append([0, 0, 0, u, v, 1, -Y * u, -Y * v, -Y])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def apply_h(H, uv):
    u, v = (uv[0] - 320) / 320.0, (uv[1] - 240) / 240.0
    w = H @ np.array([u, v, 1.0])
    return np.array([w[0] / w[2] * 0.2, w[1] / w[2] * 0.2])


def inv_h(H, xy):
    Hi = np.linalg.inv(H)
    w = Hi @ np.array([xy[0] / 0.2, xy[1] / 0.2, 1.0])
    return np.array([w[0] / w[2] * 320 + 320, w[1] / w[2] * 240 + 240])


def inv_affine(M, xy):
    A = M[:2].T
    return np.linalg.solve(A, np.array(xy) - M[2])


def fit_affine(px, xy):
    A = np.array([[u, v, 1.0] for u, v in px])
    M, *_ = np.linalg.lstsq(A, np.array(xy), rcond=None)
    return M


def apply_affine(M, uv):
    return np.array([uv[0], uv[1], 1.0]) @ M


def ik5(u0, p_t, R_t, iters=80):
    u = np.array(u0, float)

    def resid(x):
        T = fk5(x)
        Rr = T[:3, :3]
        eR = 0.5 * (np.cross(Rr[:, 0], R_t[:, 0]) + np.cross(Rr[:, 1], R_t[:, 1]) + np.cross(Rr[:, 2], R_t[:, 2]))
        return np.r_[T[:3, 3] - p_t, 0.1 * eR]

    lam = 1e-4
    for _ in range(iters):
        r = resid(u)
        J = np.zeros((6, 5))
        for k in range(5):
            du = np.zeros(5)
            du[k] = 1e-4
            J[:, k] = (resid(u + du) - r) / 1e-4
        step = np.linalg.solve(J.T @ J + lam * np.eye(5), -J.T @ r)
        u = u + step
        if np.linalg.norm(step) < 1e-8:
            break
    rr = resid(u)
    return u, float(np.linalg.norm(rr[:3]))


def deg_of(u5):
    return [float(u5[i]) * (CAL[n]["range_max"] - CAL[n]["range_min"]) / 200.0 * 360.0 / 4095.0 for i, n in enumerate(BODY)]


def main():
    eps = sorted(LABELS)
    px = [LABELS[e] for e in eps]
    xy = [R[e]["tcp_grasp_m"][:2] for e in eps]
    z = [R[e]["tcp_grasp_m"][2] for e in eps]
    res_h, res_a, loo = [], [], []
    for k, e in enumerate(eps):
        idx = [i for i in range(len(eps)) if i != k]
        H = fit_h([px[i] for i in idx], [xy[i] for i in idx])
        M = fit_affine([px[i] for i in idx], [xy[i] for i in idx])
        ph = apply_h(H, px[k])
        pa = apply_affine(M, px[k])
        eh = 1000 * np.linalg.norm(ph - np.array(xy[k]))
        ea = 1000 * np.linalg.norm(pa - np.array(xy[k]))
        res_h.append(eh)
        res_a.append(ea)
        # offline pick validation (held out): plan XY from start-frame label, IK with the human approach orientation
        Th = fk5(R[e]["state_grasp"][:5])
        u_plan, ik_pos_err = ik5(R[e]["state_grasp"][:5], np.r_[pa, z[k]], Th[:3, :3])
        dj = np.array(deg_of(u_plan)) - np.array(deg_of(R[e]["state_grasp"][:5]))
        loo.append({"ep": e, "label_px": px[k], "human_tcp_xy_mm": [round(v * 1000, 1) for v in xy[k]],
                    "planned_xy_mm_affine": [round(v * 1000, 1) for v in pa], "xy_err_mm_homography": round(eh, 2),
                    "xy_err_mm_affine": round(ea, 2), "ik_pos_err_mm": round(ik_pos_err * 1000, 2),
                    "joint_err_deg_abs_max": round(float(np.max(np.abs(dj))), 2),
                    "joint_err_deg_norm": round(float(np.linalg.norm(dj)), 2),
                    "human_norm_joints": [round(v, 2) for v in R[e]["state_grasp"][:5]],
                    "ik_norm_joints": [round(float(v), 2) for v in u_plan]})
    H_all = fit_h(px, xy)
    M_all = fit_affine(px, xy)
    resid_all = [1000 * np.linalg.norm(apply_h(H_all, px[i]) - np.array(xy[i])) for i in range(len(eps))]
    plates = {"blue_ep7_release_facts": "blue", "blue_wp0_gm_fast": "blue", "green_ep8_release_facts": "green",
              "green_ep9_release_facts": "green", "green_ep10_release_facts": "green", "green_wp0_gm_fast": "green",
              "home_facts_13:24": "home"}
    known = []
    for k, tag in plates.items():
        xy_k = np.array(rows["known_poses"][k]["tcp_m"][:2])
        known.append({"pose": k, "tcp_xy_mm": [round(v * 1000, 1) for v in xy_k],
                      "pixel_homography": [round(float(v), 1) for v in inv_h(H_all, xy_k)],
                      "pixel_affine": [round(float(v), 1) for v in inv_affine(M_all, xy_k)], "class": tag})
    calib = {"units": {"pixel": "top view 640x480 (realsense 243622061421)", "xy": "m, robot base frame (URDF)"},
             "fk": {"urdf": URDF, "tcp": TCP, "normalisation": "rad = u*(range_max-range_min)/200*2*pi/4095 (physicalai _conversion.py)"},
             "accepted_eps": eps, "labels_px": {str(e): LABELS[e] for e in eps},
             "excluded": {str(k): v for k, v in EXCLUDED.items()},
             "homography_pixel_to_xy": H_all.round(8).tolist(),
             "affine_pixel_to_xy": M_all.round(8).tolist(),
             "loo_xy_err_mm_homography": {"median": round(float(np.median(res_h)), 2), "max": round(float(np.max(res_h)), 2), "per_ep": {str(e): round(float(v), 2) for e, v in zip(eps, res_h)}},
             "loo_xy_err_mm_affine": {"median": round(float(np.median(res_a)), 2), "max": round(float(np.max(res_a)), 2), "per_ep": {str(e): round(float(v), 2) for e, v in zip(eps, res_a)}},
             "in_sample_xy_err_mm_homography": [round(float(v), 2) for v in resid_all],
             "known_poses": known, "pairs": loo}
    json.dump(calib, open(OUT + "/calib.json", "w"), indent=1)
    print("accepted", eps)
    print("LOO homography mm median %.2f max %.2f" % (np.median(res_h), np.max(res_h)))
    print("LOO affine mm median %.2f max %.2f" % (np.median(res_a), np.max(res_a)))
    print("in-sample H mm", [round(float(v), 1) for v in resid_all])
    for p in loo:
        print(p["ep"], "label", p["label_px"], "human", p["human_tcp_xy_mm"], "plan", p["planned_xy_mm_affine"], "errH", p["xy_err_mm_homography"], "errA", p["xy_err_mm_affine"], "ikpos", p["ik_pos_err_mm"], "jdeg", p["joint_err_deg_abs_max"])
    for kp in known:
        print("KNOWN", kp["pose"], kp["tcp_xy_mm"], "-> px", kp["pixel_homography"])


if __name__ == "__main__":
    main()
