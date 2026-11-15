"""Offline feasibility of a wrist-camera survey of a brick put down at the inspection station: for each side face,
the reachable top-down pose (differential IK from the station hover) that shows it best to the wrist camera
(projected area, viewing angle, jaw-mask overlap, image margin). No robot motion. Uses kin + eyeinhand/wristcam."""
import itertools
import math
import os
import sys
import time

import cv2
import numpy as np

APP = "/home/intel/intel-junction-2026/app"
sys.path[:0] = [f"{APP}/tools", f"{APP}/eyeinhand"]
import kin  # noqa: E402
import wristcam as wc  # noqa: E402
import sort_any as sa  # noqa: E402  (SPOT_HOVER and the put-down arithmetic)
import pick_any as pa  # noqa: E402

W, H = 640, 480
JAW = cv2.imread(f"{APP}/tools/wrist_jaw_mask.png", 0)
JAW = (JAW > 0) if JAW is not None else np.zeros((H, W), bool)
S, HB = 31.8 / 2, 19.2  # DUPLO 2x2 nominal half-footprint and height (mm); the runtime uses the measured footprint


def brick(c, yaw):
    R2 = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    z0, z1 = kin.MAT_Z, kin.MAT_Z + HB
    xy = [c + R2 @ np.array(v) for v in ((S, -S), (S, S), (-S, S), (-S, -S))]
    bot = [np.array([p[0], p[1], z0]) for p in xy]
    top = [np.array([p[0], p[1], z1]) for p in xy]
    faces = {"top": (top, np.array([0, 0, 1.0]))}
    for k in range(4):
        a, b = k, (k + 1) % 4
        n2 = R2 @ np.array([(1, 0), (0, 1), (-1, 0), (0, -1)][k], float)
        faces[f"side{k}"] = ([bot[a], bot[b], top[b], top[a]], np.array([n2[0], n2[1], 0.0]))
    return faces


def view(q, faces):
    Rbc, cb = wc.camera_pose(q)
    out = {}
    for name, (pts, n) in faces.items():
        ctr = np.mean(pts, axis=0)
        to_cam = cb - ctr
        cosang = float(n @ to_cam / np.linalg.norm(to_cam))
        uv = np.array([wc.project(q, p) for p in pts])
        inside = bool(np.all((uv[:, 0] > 15) & (uv[:, 0] < W - 15) & (uv[:, 1] > 15) & (uv[:, 1] < H - 15)))
        area = 0.5 * abs(np.dot(uv[:, 0], np.roll(uv[:, 1], 1)) - np.dot(uv[:, 1], np.roll(uv[:, 0], 1)))
        jaw = 0.0
        if inside and area > 1:
            m = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(m, uv.astype(np.int32), 1)
            jaw = float((JAW & (m > 0)).sum()) / max(1, int(m.sum()))
        out[name] = {"cos": round(cosang, 3), "area": round(float(area)), "inside": inside, "jaw": round(jaw, 3),
                     "uv": uv.round(1).tolist()}
    return out


t0 = time.time()
hi = kin.ik_rel(sa.SPOT_HOVER, [0, 0, 35.0])
qd = kin.ik_rel(hi, [0, 0, -(35.0 + pa.HOVER_DZ - pa.TIP_FINAL - 1.0)])
put = np.array(kin.fk(qd)[0][:2])
yaw = kin.psi_of(qd)
print("station put xy", put.round(1), "tool yaw deg", round(math.degrees(yaw), 1), "q_put", qd)
faces = brick(put, yaw)
qh = list(sa.SPOT_HOVER)
print("hover q", qh, "tcp", np.round(kin.fk(qh)[0], 1))
v0 = view(qh, faces)
print("AT HOVER:", {k: (v["cos"], v["area"], v["inside"], v["jaw"]) for k, v in v0.items()})
best = {}
grid = itertools.product(range(-105, 106, 15), range(-105, 106, 15), (0, 20, 40, 70), (-60, -30, 0, 30, 60))
n_ok = 0
for dx, dy, dz, dyaw in grid:
    q = kin.ik_rel(qh, [dx, dy, dz], math.radians(dyaw))
    if q is None or abs(q[4]) > 95:
        continue
    n_ok += 1
    v = view(q, faces)
    for name, r in v.items():
        if not r["inside"] or r["cos"] < 0.35 or r["jaw"] > 0.05:
            continue
        top_ok = v["top"]["inside"]
        score = r["area"] * r["cos"]
        if name != "top" and not top_ok:
            score *= 0.8
        if score > best.get(name, (0,))[0]:
            best[name] = (round(score), dx, dy, dz, dyaw, q, r["cos"], r["area"], r["jaw"])
print("reachable candidates", n_ok, "time", round(time.time() - t0, 1), "s")
for name in ["top", "side0", "side1", "side2", "side3"]:
    print(name, best.get(name, "NOT VISIBLE"))
