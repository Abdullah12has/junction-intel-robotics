#!/usr/bin/env python3
"""pick_v1 live tip-visible calibration (V1 candidate; not run on hardware by V1).

Visits a grid of poses at ONE fixed height above the mat, reads the red jaw centroid in the top
view at each pose, and fits jointly:
  G        plane homography, table XY (scaled by 0.2 m) -> top-view pixels (8 DOF)
  offsets  zero offsets for shoulder_lift, elbow_flex, wrist_flex (rad)
  tool     jaw-centroid offset from gripper_frame_link, x and y in the gripper frame (m)
shoulder_pan and wrist_roll get no fitted offset: a constant pan rotation is absorbed by G, and the
roll is held constant during the grid. A leave-one-out gate (max table error, mm) and plausibility
checks decide PASS or FAIL.

Motion goes only through app/motion/so101_motion.py (claim_devices, open_follower_held, MotionPlayer).
Default --bus fake uses FakeFollower and FakeCamera: no serial port and no camera.
Real motion: --bus real --confirm-motion --cal-z METRES --port /dev/serial/by-id/... (speed <= 20).

Exit codes: 0 PASS, 1 selftest failed, 2 stopped (arm held), 3 device owned by another process,
4 configuration error, 5 hardware error, 6 gate FAIL (output written, do not use).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
import types
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path("/home/intel/intel-junction-2026")
MOTION_DIR = REPO / "app" / "motion"
URDF_PATH = REPO / "repos/physicalai/packages/physicalai-so101-plugin/urdf/so101-follower/so101_follower.urdf"
URDF_MD5 = "308645ed050642dbbfc21814588a286c"
ROBOT_JSON = MOTION_DIR / "project_robots.json"
FOLLOWER_SERIAL = "5B3D040939"
OUT_DIR = Path(__file__).resolve().parent / "logs"

TIP_LINK = "gripper_frame_link"
BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ROI = (40, 30, 600, 450)                # x0, y0, x1, y1 in the 640x480 top view; covers the grid
RED_HUE_MAX, RED_HUE_MIN = 8.0, 172.0   # OpenCV hue units 0..179: within 8 of 0
RED_S_MIN, RED_V_MIN = 110.0, 40.0
BLOB_MIN_PX, BLOB_MAX_PX = 40, 2000
AMBIGUOUS_RATIO = 0.3
XY_SCALE = 0.2                          # plane coordinates are divided by this (metres)
IMG_C = (320.0, 240.0)
IMG_S = (320.0, 240.0)
LOO_GATE_MM = 10.0
MIN_POSES = 20
OFFSET_LIMIT_RAD = math.radians(10.0)
TOOL_LIMIT_M = 0.08
HOVER_M = 0.04
SETTLE_S = 0.6
RATE_HZ = 30.0
SPEED_DEFAULT, SPEED_CAP = 15.0, 20.0
TOLERANCE_DEFAULT = 6.0
UNIT_LIMIT = 95.0
GRIPPER_OPEN_MIN = 8.0
FLOOR_MARGIN_M = 0.005
CAL_Z_DRYRUN_M = 0.010
PLATE_CENTRES_M = ((0.1854, -0.0898), (0.1973, 0.2283))   # URDF-frame release centres, dataset 6d6bdce3 (blue n=3, green n=2); MJCF geometry.json is 2 cm off
PLATE_EXCLUDE_M = 0.115
BASE_EXCLUDE_M = 0.12
G_TRUE = np.array([[0.0, 0.625, 0.0], [0.8333, 0.0, -0.8333], [1e-4, -5e-5, 1.0]])   # synthetic truth, 1000 px/m
TRUTH_OFFSETS = (0.02, -0.015, 0.025)
TRUTH_TOOL = (0.02, -0.01)
PRIOR_PX = 1.5                        # MAP prior residuals, pixel-equivalent scale
OFFSET_PRIOR_RAD = math.radians(5.0)  # weak prior on lift, elbow, wrist zero offsets
TOOL_PRIOR_M = 0.03                   # weak prior on the jaw-centroid tool offset


class ConfigError(RuntimeError):
    """Bad configuration or a missing precondition; nothing was moved."""


class StoppedError(RuntimeError):
    """A move or a safety check stopped the calibration; the arm is held."""


def _floats(text, default):
    return np.array([float(x) for x in (default if text is None else text).split()], dtype=float)


def check_urdf():
    digest = hashlib.md5(URDF_PATH.read_bytes()).hexdigest()
    if digest != URDF_MD5:
        raise ConfigError(f"URDF md5 {digest} is not {URDF_MD5}")


def load_chain():
    root = ET.parse(URDF_PATH).getroot()
    by_child = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        if parent is None or child is None:
            continue
        origin, axis = joint.find("origin"), joint.find("axis")
        by_child[child.get("link")] = {
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": parent.get("link"),
            "xyz": _floats(origin.get("xyz") if origin is not None else None, "0 0 0"),
            "rpy": _floats(origin.get("rpy") if origin is not None else None, "0 0 0"),
            "axis": _floats(axis.get("xyz") if axis is not None else None, "0 0 1"),
        }
    chain, link = [], TIP_LINK
    while link in by_child:
        chain.append(by_child[link])
        link = by_child[link]["parent"]
    chain.reverse()
    names = {joint["name"] for joint in chain}
    missing = [name for name in BODY if name not in names]
    if missing:
        raise ConfigError(f"URDF chain to {TIP_LINK} lacks {missing}")
    return chain


def load_ranges():
    found = []

    def walk(node):
        if isinstance(node, dict):
            if "calibration" in node and FOLLOWER_SERIAL in json.dumps(node):
                found.append(node["calibration"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(json.loads(ROBOT_JSON.read_text()))
    if not found:
        raise ConfigError("follower calibration not found in project_robots.json")
    cal = found[0]
    return {name: (float(cal[name]["range_min"]), float(cal[name]["range_max"])) for name in BODY}


def rad_from_units(units, rng):
    rmin, rmax = rng
    return units * (rmax - rmin) / 200.0 * 2.0 * math.pi / 4095.0


def _rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _axis_rotations(axis, angles):
    x, y, z = axis / np.linalg.norm(axis)
    c, s = np.cos(angles), np.sin(angles)
    k = 1.0 - c
    rot = np.empty((len(angles), 3, 3))
    rot[:, 0, 0] = x * x * k + c
    rot[:, 0, 1] = x * y * k - z * s
    rot[:, 0, 2] = x * z * k + y * s
    rot[:, 1, 0] = y * x * k + z * s
    rot[:, 1, 1] = y * y * k + c
    rot[:, 1, 2] = y * z * k - x * s
    rot[:, 2, 0] = z * x * k - y * s
    rot[:, 2, 1] = z * y * k + x * s
    rot[:, 2, 2] = z * z * k + c
    return rot


def fk_gripper(chain, angles):
    """angles: joint name -> (N,) radians. Returns (N,4,4) pose of gripper_frame_link in the base frame."""
    n = len(next(iter(angles.values())))
    T = np.tile(np.eye(4), (n, 1, 1))
    for joint in chain:
        Tj = np.eye(4)
        Tj[:3, :3] = _rpy_matrix(joint["rpy"])
        Tj[:3, 3] = joint["xyz"]
        T = T @ Tj
        if joint["type"] in ("revolute", "continuous"):
            Tq = np.tile(np.eye(4), (n, 1, 1))
            Tq[:, :3, :3] = _axis_rotations(joint["axis"], angles.get(joint["name"], np.zeros(n)))
            T = T @ Tq
    return T


def body_angles(units, ranges, offsets=(0.0, 0.0, 0.0), pan_offset=0.0):
    out = {}
    for i, name in enumerate(BODY):
        out[name] = rad_from_units(units[:, i], ranges[name])
    out["shoulder_lift"] = out["shoulder_lift"] + offsets[0]
    out["elbow_flex"] = out["elbow_flex"] + offsets[1]
    out["wrist_flex"] = out["wrist_flex"] + offsets[2]
    out["shoulder_pan"] = out["shoulder_pan"] + pan_offset
    return out


def tip_xyz(chain, ranges, units, offsets=(0.0, 0.0, 0.0), tool=(0.0, 0.0), pan_offset=0.0):
    """Modelled jaw-centroid position (N,3) in the base frame for normalized body units (N,5)."""
    units = np.atleast_2d(np.asarray(units, dtype=float))
    T = fk_gripper(chain, body_angles(units, ranges, offsets, pan_offset))
    tool3 = np.array([tool[0], tool[1], 0.0])
    return T[:, :3, 3] + np.einsum("nij,j->ni", T[:, :3, :3], tool3)


def ik_tool_down(chain, ranges, target, roll_units, start_units=None):
    """Nominal IK for gripper_frame_link at target with its z axis pointing down; roll fixed at roll_units.

    Returns (units(5), position error mm, tilt residual). Joints solved: pan, lift, elbow, wrist_flex.
    """
    free = [0, 1, 2, 3]
    base = np.zeros(5) if start_units is None else np.array(start_units, dtype=float)
    base[4] = roll_units
    tgt = np.asarray(target, dtype=float)
    down = np.array([0.0, 0.0, -1.0])

    def resid(v):
        u = base.copy()
        u[free] = v
        T = fk_gripper(chain, body_angles(u[None, :], ranges))[0]
        return np.r_[T[:3, 3] - tgt, 0.1 * (T[:3, 2] - down)]

    v = base[free].copy()
    r = resid(v)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(60):
        J = np.empty((6, 4))
        for k in range(4):
            dv = np.zeros(4)
            dv[k] = 0.05
            J[:, k] = (resid(v + dv) - r) / 0.05
        A = J.T @ J
        g = J.T @ r
        step = np.linalg.solve(A + lam * np.eye(4), -g)
        v_new = v + step
        r_new = resid(v_new)
        cost_new = float(r_new @ r_new)
        if cost_new < cost:
            v, r, cost = v_new, r_new, cost_new
            lam = max(lam / 3.0, 1e-9)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            lam *= 4.0
            if lam > 1e8:
                break
    u = base.copy()
    u[free] = v
    return u, 1000.0 * float(np.linalg.norm(r[:3])), float(np.linalg.norm(r[3:]) / 0.1)


def candidate_points():
    pts = []
    for radius in np.linspace(0.12, 0.28, 9):
        for az in np.linspace(-60.0, 60.0, 13):
            x = radius * math.cos(math.radians(az))
            y = radius * math.sin(math.radians(az))
            if math.hypot(x, y) < BASE_EXCLUDE_M:
                continue
            if any(math.hypot(x - cx, y - cy) < PLATE_EXCLUDE_M for cx, cy in PLATE_CENTRES_M):
                continue
            pts.append((x, y))
    return np.array(pts)


def select_points(pts, n):
    """Farthest-point sampling: n poses spread over the candidate set (all of them if n is larger)."""
    if n >= len(pts):
        return pts
    chosen = [int(np.argmax(np.hypot(pts[:, 0], pts[:, 1])))]
    dist = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    while len(chosen) < n:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(pts - pts[nxt], axis=1))
    return pts[chosen]


def plan_targets(chain, ranges, cal_z, pts, roll_units):
    plan, last = [], None
    for x, y in pts:
        starts = [np.zeros(5)] + ([last] if last is not None else [])
        best = None
        for start in starts:
            u, pos_mm, tilt = ik_tool_down(chain, ranges, [x, y, cal_z], roll_units, start)
            if best is None or pos_mm < best[1]:
                best = (u, pos_mm, tilt)
        u_cal, pos_mm, tilt = best
        if pos_mm > 1.0:
            continue
        u_hov, pos_hov, _ = ik_tool_down(chain, ranges, [x, y, cal_z + HOVER_M], roll_units, u_cal)
        if pos_hov > 1.0 or np.any(np.abs(u_cal[:4]) > UNIT_LIMIT) or np.any(np.abs(u_hov[:4]) > UNIT_LIMIT):
            continue
        last = u_cal
        plan.append({"xy": [float(x), float(y)], "u_cal": u_cal, "u_hover": u_hov,
                     "ik_pos_mm": pos_mm, "tilt": tilt})
    return plan


def order_nn(plan):
    """Greedy nearest-neighbour order so the arm path between poses stays short."""
    if not plan:
        return plan
    remaining = list(range(len(plan)))
    order = [remaining.pop(0)]
    while remaining:
        last = np.array(plan[order[-1]]["xy"])
        nxt = min(remaining, key=lambda i: np.linalg.norm(np.array(plan[i]["xy"]) - last))
        remaining.remove(nxt)
        order.append(nxt)
    return [plan[i] for i in order]


def unpack(p):
    G = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]])
    return G, p[8:11], p[11:13]


def pack(G, offsets, tool):
    G = np.asarray(G, dtype=float)
    G = G / G[2, 2]
    return np.r_[G.ravel()[:8], np.asarray(offsets, dtype=float), np.asarray(tool, dtype=float)]


def project(G, xy):
    S = np.c_[xy[:, 0] / XY_SCALE, xy[:, 1] / XY_SCALE, np.ones(len(xy))]
    W = S @ G.T
    return np.c_[IMG_C[0] + IMG_S[0] * (W[:, 0] / W[:, 2]), IMG_C[1] + IMG_S[1] * (W[:, 1] / W[:, 2])]


def residual(p, chain, ranges, units, px):
    G, offsets, tool = unpack(p)
    xy = tip_xyz(chain, ranges, units, offsets, tool)[:, :2]
    return (project(G, xy) - px).ravel()


def prior_rows(p):
    _, offsets, tool = unpack(p)
    return np.r_[PRIOR_PX * offsets / OFFSET_PRIOR_RAD, PRIOR_PX * tool / TOOL_PRIOR_M]


def full_residual(p, chain, ranges, units, px):
    return np.r_[residual(p, chain, ranges, units, px), prior_rows(p)]


def init_params(chain, ranges, units, px):
    """DLT for G using nominal positions (no offsets, no tool offset)."""
    xy = tip_xyz(chain, ranges, units)[:, :2]
    S = np.c_[xy / XY_SCALE, np.ones(len(xy))]
    up = (px[:, 0] - IMG_C[0]) / IMG_S[0]
    vp = (px[:, 1] - IMG_C[1]) / IMG_S[1]
    rows = []
    for s, a, b in zip(S, up, vp):
        rows.append(np.r_[s, np.zeros(3), -a * s])
        rows.append(np.r_[np.zeros(3), s, -b * s])
    _, _, vt = np.linalg.svd(np.array(rows))
    h = vt[-1].reshape(3, 3)
    G = h / h[2, 2]
    return np.r_[G.ravel()[:8], np.zeros(5)]


def _jacobian(fun, p, r0):
    J = np.empty((len(r0), len(p)))
    for k in range(len(p)):
        h = 1e-6 * max(1.0, abs(p[k]))
        dp = np.zeros(len(p))
        dp[k] = h
        J[:, k] = (fun(p + dp) - r0) / h
    return J


def levenberg_marquardt(p0, fun, max_iter=80):
    p = np.asarray(p0, dtype=float).copy()
    r = fun(p)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(max_iter):
        J = _jacobian(fun, p, r)
        A = J.T @ J
        g = J.T @ r
        scale = np.diag(np.diag(A)) + 1e-12 * np.eye(len(p))
        accepted, tiny = False, False
        for _ in range(15):
            try:
                step = -np.linalg.solve(A + lam * scale, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(A + lam * scale, g, rcond=None)[0]
            p_new = p + step
            r_new = fun(p_new)
            cost_new = float(r_new @ r_new)
            if np.isfinite(cost_new) and cost_new < cost:
                accepted = True
                tiny = (cost - cost_new) <= 1e-12 * max(1.0, cost)
                p, r, cost = p_new, r_new, cost_new
                lam = max(lam / 4.0, 1e-10)
                break
            lam *= 4.0
        if not accepted or tiny:
            break
    return p, r


def fit_and_gate(chain, ranges, units, px):
    units = np.asarray(units, dtype=float)
    px = np.asarray(px, dtype=float)
    n = len(units)
    p0 = init_params(chain, ranges, units, px)
    p, r = levenberg_marquardt(p0, lambda q: full_residual(q, chain, ranges, units, px))
    rd = r[:2 * n]
    rms = float(np.sqrt(np.mean(rd ** 2)))
    loo = np.empty(n)
    for k in range(n):
        keep = np.arange(n) != k
        pk, _ = levenberg_marquardt(
            p, lambda q, keep=keep: full_residual(q, chain, ranges, units[keep], px[keep]), max_iter=40)
        Gk, offsets_k, tool_k = unpack(pk)
        s = np.linalg.solve(Gk, np.array([(px[k, 0] - IMG_C[0]) / IMG_S[0],
                                          (px[k, 1] - IMG_C[1]) / IMG_S[1], 1.0]))
        xy_meas = XY_SCALE * s[:2] / s[2]
        xy_model = tip_xyz(chain, ranges, units[k:k + 1], offsets_k, tool_k)[0, :2]
        loo[k] = 1000.0 * float(np.linalg.norm(xy_meas - xy_model))
    G, offsets, tool = unpack(p)
    xy_all = tip_xyz(chain, ranges, units, offsets, tool)[:, :2]
    pxs = project(G, xy_all)
    x0, y0, x1, y1 = ROI
    centre = xy_all.mean(axis=0)
    p_c = project(G, centre[None])[0]
    p_dx = project(G, (centre + np.array([0.01, 0.0]))[None])[0]
    p_dy = project(G, (centre + np.array([0.0, 0.01]))[None])[0]
    base_px = project(G, np.zeros((1, 2)))[0]
    jac = _jacobian(lambda q: full_residual(q, chain, ranges, units, px), p, r)
    sv = np.linalg.svd(jac, compute_uv=False)
    checks = {
        "min_poses": n >= MIN_POSES,
        "loo_max_mm_at_most_10": bool(loo.max() <= LOO_GATE_MM),
        "in_sample_rms_px_at_most_3": bool(rms <= 3.0),
        "offsets_within_10_deg": bool(np.all(np.abs(offsets) <= OFFSET_LIMIT_RAD)),
        "tool_offset_within_8_cm": bool(np.linalg.norm(tool) <= TOOL_LIMIT_M),
        "image_orientation_u_grows_with_Y_v_grows_with_X": bool(p_dy[0] > p_c[0] and p_dx[1] > p_c[1]),
        "grid_inside_roi": bool(np.all((pxs[:, 0] > x0) & (pxs[:, 0] < x1) & (pxs[:, 1] > y0) & (pxs[:, 1] < y1))),
        "base_origin_top_centre_assumption": bool(160.0 < base_px[0] < 480.0 and base_px[1] < 200.0),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "n_poses": int(n),
        "params": {
            "G": G.tolist(),
            "offsets_rad": [float(v) for v in offsets],
            "offsets_deg": [float(v) for v in np.degrees(offsets)],
            "tool_offset_gripper_m": [float(v) for v in tool],
            "shoulder_pan_offset": "not fitted: a constant rotation about the base is absorbed by G",
        },
        "in_sample_rms_px": rms,
        "in_sample_max_px": float(np.max(np.abs(rd))),
        "loo_mm": {"median": float(np.median(loo)), "rms": float(np.sqrt(np.mean(loo ** 2))),
                   "max": float(loo.max())},
        "jacobian_singular_values": [float(v) for v in sv],
        "checks": checks,
    }


def rgb_to_hsv_cv(img):
    f = img.astype(np.float64)
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(axis=2), f.min(axis=2)
    d = mx - mn
    safe = np.where(d > 0, d, 1.0)
    hue = np.where(mx == r, 60.0 * np.mod((g - b) / safe, 6.0),
                   np.where(mx == g, 60.0 * ((b - r) / safe + 2.0), 60.0 * ((r - g) / safe + 4.0)))
    hue = np.where(d > 0, hue, 0.0) / 2.0
    sat = np.where(mx > 0, d / np.where(mx > 0, mx, 1.0) * 255.0, 0.0)
    return hue, sat, mx


def _components(mask):
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        seen[y, x] = True
        stack, pts = [(int(y), int(x))], []
        while stack:
            cy, cx = stack.pop()
            pts.append((cy, cx))
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(np.array(pts))
    return comps


def detect_red_tip(img_rgb):
    """Centroid (u, v, area) of the red jaw in the top view, or (None, reason)."""
    x0, y0, x1, y1 = ROI
    sub = img_rgb[y0:y1, x0:x1]
    hue, sat, val = rgb_to_hsv_cv(sub)
    mask = ((hue <= RED_HUE_MAX) | (hue >= RED_HUE_MIN)) & (sat >= RED_S_MIN) & (val >= RED_V_MIN)
    comps = sorted(_components(mask), key=len, reverse=True)
    if not comps:
        return None, "no red pixels in ROI"
    big = comps[0]
    if not BLOB_MIN_PX <= len(big) <= BLOB_MAX_PX:
        return None, f"red blob area {len(big)} px outside {BLOB_MIN_PX}..{BLOB_MAX_PX}"
    if len(comps) > 1 and len(comps[1]) >= AMBIGUOUS_RATIO * len(big):
        return None, "second red blob too large (ambiguous)"
    rows, cols = big[:, 0], big[:, 1]
    h, w = mask.shape
    if rows.min() == 0 or cols.min() == 0 or rows.max() == h - 1 or cols.max() == w - 1:
        return None, "red blob touches the ROI border"
    return (x0 + float(cols.mean()), y0 + float(rows.mean()), int(len(big))), "ok"


class LiveCameraCv2:
    def __init__(self, source):
        try:
            import cv2
        except ImportError as error:
            raise ConfigError(f"cv2 is needed for the live camera ({error})") from error
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"camera {source} did not open")

    def capture(self):
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return np.ascontiguousarray(bgr[:, :, ::-1])

    def close(self):
        self.cap.release()


class LiveCamera:
    """Top camera through ffmpeg v4l2, one RGB frame per capture (the motion-env has no cv2)."""

    def __init__(self, source, size=(640, 480)):
        if shutil.which("ffmpeg") is None:
            raise ConfigError("ffmpeg is not on PATH; it is needed for the live camera")
        self.source = str(source)
        self.size = size

    def capture(self):
        width, height = self.size
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2",
               "-video_size", f"{width}x{height}", "-i", self.source,
               "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.run(cmd, capture_output=True, timeout=20)
        if proc.returncode != 0 or len(proc.stdout) != width * height * 3:
            raise RuntimeError(f"camera capture failed: {proc.stderr.decode(errors='replace')[:200]}")
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(height, width, 3).copy()

    def close(self):
        return None


class FakeFollower:
    """Dry-run stand-in for the SO101 driver: goals apply at once, read-back adds encoder noise."""

    max_speed = 279.0

    def __init__(self, start_units, rng, noise_units=0.2):
        self.pos = np.array(start_units, dtype=float)
        self.rng = rng
        self.noise = noise_units

    def send_action(self, setpoint):
        self.pos = np.asarray(setpoint, dtype=float).copy()

    def get_observation(self):
        noise = self.rng.normal(0.0, self.noise, 6)
        noise[5] = 0.0
        return types.SimpleNamespace(joint_positions=self.pos + noise)

    def _read_joint_positions(self):
        return np.zeros(6, dtype=np.int32)

    def _write_joint_positions(self, raw):
        return None

    def disconnect(self):
        return None


class FakeCamera:
    """Draws the red jaw disc where the truth model puts it (1.5 px pixel noise, 3 % missed frames)."""

    def __init__(self, driver, chain, ranges, truth_p, rng):
        self.driver, self.chain, self.ranges, self.rng = driver, chain, ranges, rng
        self.G, self.offsets, self.tool = unpack(truth_p)
        self.yy, self.xx = np.mgrid[0:480, 0:640]

    def capture(self):
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        if self.rng.random() < 0.03:
            return img
        xy = tip_xyz(self.chain, self.ranges, self.driver.pos[None, :5], self.offsets, self.tool)[:, :2]
        uv = project(self.G, xy)[0] + self.rng.normal(0.0, 1.5, 2)
        disc = (self.xx - uv[0]) ** 2 + (self.yy - uv[1]) ** 2 <= 36.0
        img[disc] = (220, 30, 30)
        return img


def truth_params():
    return pack(G_TRUE, TRUTH_OFFSETS, TRUTH_TOOL)


def acquire(player, camera, chain, ranges, plan, cal_z, speed, records, missing):
    start = player.current()
    if start[5] < GRIPPER_OPEN_MIN:
        raise ConfigError(f"gripper reads {start[5]:.1f}; open it above {GRIPPER_OPEN_MIN} before calibration")
    for k, item in enumerate(plan):
        goal = player.current()
        goal[:5] = item["u_hover"]
        res = player.move_to(goal, max_speed=speed)
        if not res.arrived:
            raise StoppedError(f"pose {k} hover not reached (max error {res.max_error:.2f} units)")
        goal[:5] = item["u_cal"]
        res = player.move_to(goal, max_speed=speed)
        if not res.arrived:
            raise StoppedError(f"pose {k} calibration height not reached (max error {res.max_error:.2f} units)")
        read = player.current()
        z_nom = float(tip_xyz(chain, ranges, read[None, :5])[0, 2])
        if z_nom < cal_z - FLOOR_MARGIN_M:
            raise StoppedError(f"pose {k}: nominal tip z {z_nom:.4f} m is below the floor "
                               f"{cal_z - FLOOR_MARGIN_M:.4f} m")
        det, why = None, "no frame"
        for _ in range(2):
            det, why = detect_red_tip(camera.capture())
            if det is not None:
                break
            time.sleep(0.5)
        if det is None:
            missing.append({"k": k, "reason": why})
            continue
        records.append({"k": k, "xy_nominal": item["xy"], "units_read": [float(v) for v in read[:5]],
                        "u": det[0], "v": det[1], "area": det[2]})


def import_motion():
    if str(MOTION_DIR) not in sys.path:
        sys.path.insert(0, str(MOTION_DIR))
    try:
        import so101_motion
    except ImportError as error:
        raise ConfigError(f"so101_motion import failed ({error}); use the motion-env interpreter") from error
    return so101_motion


def write_output(args, cal_z, plan, records, missing, result):
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    doc = {
        "bus": args.bus,
        "utc": stamp,
        "cal_z_m": cal_z,
        "speed": args.speed,
        "tolerance": args.tolerance,
        "poses_requested": args.poses,
        "poses_planned_xy": [item["xy"] for item in plan],
        "records": records,
        "missing": missing,
        "fit": result,
        "note": ("fake bus: synthetic driver and camera, the result shows the code path only"
                 if args.bus == "fake" else "real bus"),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"live_calib_{args.bus}_{stamp}.json"
    path.write_text(json.dumps(doc, indent=1))
    return path


def run(args):
    if not 0.0 < args.speed <= SPEED_CAP:
        raise ConfigError(f"--speed must be in (0, {SPEED_CAP}] units/s")
    if args.bus == "real" and args.cal_z is None:
        raise ConfigError("--cal-z (TCP height in the base frame, metres) is required on the real bus")
    cal_z = CAL_Z_DRYRUN_M if args.cal_z is None else float(args.cal_z)
    check_urdf()
    chain = load_chain()
    ranges = load_ranges()
    pts = select_points(candidate_points(), args.poses)
    if args.bus == "real" and not args.confirm_motion:
        plan = order_nn(plan_targets(chain, ranges, cal_z, pts, 0.0))
        print(f"dry run: {len(plan)} poses planned at cal_z {cal_z:.4f} m, speed {args.speed}, "
              f"tolerance {args.tolerance}; no port or camera opened; add --confirm-motion to move the arm")
        return 0
    sm = import_motion()
    records, missing, plan = [], [], []
    try:
        if args.bus == "fake":
            rng = np.random.default_rng(args.seed)
            driver = FakeFollower(start_units=[0.0, 0.0, 0.0, 0.0, 0.0, 50.0], rng=rng)
            camera = FakeCamera(driver, chain, ranges, truth_params(), rng)
            try:
                player = sm.MotionPlayer(driver, max_speed=args.speed, tolerance=args.tolerance,
                                         settle_s=SETTLE_S, rate_hz=RATE_HZ)
                plan = order_nn(plan_targets(chain, ranges, cal_z, pts, 0.0))
                acquire(player, camera, chain, ranges, plan, cal_z, args.speed, records, missing)
            finally:
                driver.disconnect()
        else:
            if args.port != sm.FOLLOWER_PORT_BY_ID:
                raise ConfigError(f"--port must be {sm.FOLLOWER_PORT_BY_ID}")
            port, calibration = sm.resolve_arm(studio_export=ROBOT_JSON, name="SO101 Follower",
                                               port=args.port, calibration_path=None, follower=True)
            camera = LiveCamera(args.camera)
            try:
                with sm.claim_devices(sm.device_ids_for(port), ports=(port,)):
                    driver = sm.open_follower_held(port, calibration, baudrate=sm.DEFAULT_BAUDRATE)
                    try:
                        player = sm.MotionPlayer(driver, max_speed=args.speed, tolerance=args.tolerance,
                                                 settle_s=SETTLE_S, rate_hz=RATE_HZ)
                        roll = float(player.current()[4])
                        plan = order_nn(plan_targets(chain, ranges, cal_z, pts, roll))
                        acquire(player, camera, chain, ranges, plan, cal_z, args.speed, records, missing)
                    finally:
                        driver.disconnect()
            finally:
                camera.close()
    except StoppedError as error:
        write_output(args, cal_z, plan, records, missing, {"status": "STOPPED", "reason": str(error)})
        raise
    if len(records) >= MIN_POSES:
        result = fit_and_gate(chain, ranges, [r["units_read"] for r in records],
                              [[r["u"], r["v"]] for r in records])
    else:
        result = {"status": "FAIL", "failed_checks": ["min_poses"], "n_poses": len(records)}
    path = write_output(args, cal_z, plan, records, missing, result)
    print(f"{result['status']} {path}")
    return 0 if result["status"] == "PASS" else 6


def selftest():
    check_urdf()
    chain = load_chain()
    ranges = load_ranges()
    print("chain joints:", [joint["name"] for joint in chain])
    cands = candidate_points()
    plan_all = plan_targets(chain, ranges, CAL_Z_DRYRUN_M, cands, 0.0)
    print(f"candidate points {len(cands)}; IK-feasible at cal_z {len(plan_all)}; "
          f"max IK position error {max(p['ik_pos_mm'] for p in plan_all):.3f} mm; "
          f"max tilt residual {max(p['tilt'] for p in plan_all):.3f}")
    if len(plan_all) < MIN_POSES:
        print("SELFTEST FAIL: too few IK-feasible poses")
        return 1
    truth_o, truth_d = np.array(TRUTH_OFFSETS), np.array(TRUTH_TOOL)
    main_ok = True
    for label, n in (("40", 40), ("60", 60)):
        plan = plan_targets(chain, ranges, CAL_Z_DRYRUN_M, select_points(cands, n), 0.0)
        units = np.array([t["u_cal"] for t in plan])
        passes, loo_max, off_err, tool_err = 0, [], [], []
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            read = units + rng.normal(0.0, 0.2, units.shape) * np.array([1.0, 1.0, 1.0, 1.0, 0.0])
            xy = tip_xyz(chain, ranges, read, truth_o, truth_d)[:, :2]
            px = project(G_TRUE, xy) + rng.normal(0.0, 1.5, (len(read), 2))
            res = fit_and_gate(chain, ranges, read, px)
            passes += res["status"] == "PASS"
            loo_max.append(res["loo_mm"]["max"])
            off_err.append(float(np.max(np.abs(np.degrees(np.array(res["params"]["offsets_rad"]) - truth_o)))))
            tool_err.append(1000.0 * float(np.linalg.norm(np.array(res["params"]["tool_offset_gripper_m"]) - truth_d)))
        print(f"synthetic N={len(plan)} (requested {label}): PASS {passes}/6; LOO max median "
              f"{np.median(loo_max):.2f} mm (worst {np.max(loo_max):.2f}); offset error max "
              f"{np.max(off_err):.3f} deg; tool error max {np.max(tool_err):.2f} mm")
        main_ok = main_ok and passes >= 5
    plan = plan_targets(chain, ranges, CAL_Z_DRYRUN_M, select_points(cands, 40), 0.0)
    units = np.array([t["u_cal"] for t in plan])
    rng = np.random.default_rng(7)
    read = units + rng.normal(0.0, 0.2, units.shape) * np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    xy = tip_xyz(chain, ranges, read, truth_o, truth_d)[:, :2]
    px = project(G_TRUE, xy) + rng.normal(0.0, 1.5, (len(read), 2))
    mirror = px.copy()
    mirror[:, 0] = 640.0 - mirror[:, 0]
    res_m = fit_and_gate(chain, ranges, read, mirror)
    print("negative, mirrored image (expect FAIL):", res_m["status"], res_m["failed_checks"])
    xy_p = tip_xyz(chain, ranges, read, truth_o, truth_d, pan_offset=math.radians(2.0))[:, :2]
    px_p = project(G_TRUE, xy_p) + rng.normal(0.0, 1.5, (len(read), 2))
    res_p = fit_and_gate(chain, ranges, read, px_p)
    print("negative, 2 deg pan offset (absorbed by G; expect PASS, offsets unchanged):", res_p["status"],
          np.round(np.degrees(res_p["params"]["offsets_rad"]), 2))
    xy_b = tip_xyz(chain, ranges, read, (math.radians(15.0), 0.0, 0.0), truth_d)[:, :2]
    px_b = project(G_TRUE, xy_b) + rng.normal(0.0, 1.5, (len(read), 2))
    res_b = fit_and_gate(chain, ranges, read, px_b)
    print("negative, 15 deg lift offset (expect FAIL on the offset bound):", res_b["status"], res_b["failed_checks"])
    ok = main_ok and res_m["status"] == "FAIL" and res_b["status"] == "FAIL" and res_p["status"] == "PASS"
    print("SELFTEST PASS" if ok else "SELFTEST FAIL")
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="pick_v1 live tip-visible calibration (V1 candidate).")
    parser.add_argument("--bus", choices=("fake", "real"), default="fake")
    parser.add_argument("--selftest", action="store_true", help="synthetic checks of the fit and gate; no motion")
    parser.add_argument("--poses", type=int, default=40, help="requested number of poses at one height")
    parser.add_argument("--cal-z", type=float, default=None,
                        help="TCP height in the base frame, metres; required on the real bus")
    parser.add_argument("--speed", type=float, default=SPEED_DEFAULT, help="units per second, at most 20")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_DEFAULT, help="allowed read-back error, units")
    parser.add_argument("--port", default=None, help="follower port; must be the by-id path")
    parser.add_argument("--camera", default="/dev/video8", help="top camera device path or index")
    parser.add_argument("--confirm-motion", action="store_true", help="allow the real arm to move")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    try:
        return selftest() if args.selftest else run(args)
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 4
    except StoppedError as error:
        print(f"stopped: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # map motion-layer exception names to exit codes
        name = type(error).__name__
        if name == "DeviceOwnedError":
            print(f"device busy: {error}", file=sys.stderr)
            return 3
        if name == "MotionStalledError":
            print(f"stalled: {error}", file=sys.stderr)
            return 2
        if name == "MotionError":
            print(f"motion configuration error: {error}", file=sys.stderr)
            return 4
        if isinstance(error, (ConnectionError, OSError, RuntimeError)):
            print(f"hardware error: {error}", file=sys.stderr)
            return 5
        raise


if __name__ == "__main__":
    raise SystemExit(main())
