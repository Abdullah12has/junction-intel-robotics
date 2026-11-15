"""Tip-visible calibration of the SO-101 follower from the top camera, at one fixed height.

Flow: plan a pose grid at one height above the mat, move to each pose with the motion layer
(MotionPlayer, speed 15 to 20 units/s, tolerance 6 units), find the red jaw tip in the top view,
then fit a plane-to-image homography together with per-joint zero offsets. The gate is a
leave-one-pose-out error of at most 10 mm on the table.

Identifiability: a constant shoulder_pan offset is a rigid rotation about the pan axis, and a free
homography absorbs it, so tip pixels alone do not fix it. Anchors (table XY in base_link metres with
the pixel where that point appears, e.g. plate centres measured with a ruler) pin the homography in
the base frame. The fit reports, per offset, how much of its Jacobian column lies outside the
homography span; a near-zero share means that offset is not identifiable from the data given.

The live path moves the arm only through so101_motion (open_follower_held, MotionPlayer) and turns
torque off at the end, after the operator supports the arm (I-43 rule 5). The dry run uses fake_bus
and never opens a port or a camera.

Commands (run from the app folder that contains pick_v2):
  python3 -m pick_v2.tip_calib dry-run --urdf URDF --out DIR
  python3 -m pick_v2.tip_calib live --urdf URDF --out DIR --plate X,Y,R [--plate ...] --confirm-motion
  python3 -m pick_v2.tip_calib fit --urdf URDF --samples FILE --out DIR [--anchors FILE]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .kinematics import PAN_AXIS_XY, Chain, FollowerCalibration, solve_top_down, tcp_xyz
from .params import BASE_KEEPOUT_M, HOME_NORM, IK_POS_TOL_M, IK_ROT_TOL_RAD, JOINT_MARGIN_UNITS, TABLE_Z_BASE_M
from .table_calib import apply_homography, fit_homography

SPEED_MIN_UNITS_S = 15.0
SPEED_MAX_UNITS_S = 20.0
TOLERANCE_UNITS = 6.0
LOO_GATE_MM = 10.0
N_BODY = 5
OFFSET_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
ROBOT_NAME = "SO101 Follower"
PORT_SERIAL = "5B3D040939"  # follower serial; the port must be the by-id path (LAPTOP.md port rule)
TOP_CAMERA_SERIAL = "243622061421"  # RealSense named 'top' in the studio dataset info.json
MOTION_DIR = Path(__file__).resolve().parents[1] / "motion"  # R/app/motion when this package is R/app/pick_v2
URDF_DEFAULT = (
    "/home/intel/intel-junction-2026/repos/edge-developer-kit-reference-scripts/samples/ai/"
    "edge-ai-demo-studio/workers/services/robotics-ai/platforms/so101/urdf/so101.urdf"
)


@dataclass(frozen=True)
class GridConfig:
    z_above_table_m: float = 0.06
    psi_rad: float = math.pi
    n_poses: int = 40
    x_range_m: tuple = (0.12, 0.28)
    y_range_m: tuple = (-0.12, 0.12)
    step_m: float = 0.02
    plate_margin_m: float = 0.03
    speed_units_s: float = 15.0
    tolerance_units: float = TOLERANCE_UNITS
    settle_s: float = 0.6
    margin_units: float = JOINT_MARGIN_UNITS
    loo_gate_mm: float = LOO_GATE_MM
    min_samples: int = 20
    anchor_weight: float = 5.0

    def __post_init__(self) -> None:
        if not SPEED_MIN_UNITS_S <= self.speed_units_s <= SPEED_MAX_UNITS_S:
            raise ValueError(f"speed {self.speed_units_s} is outside {SPEED_MIN_UNITS_S:.0f} to {SPEED_MAX_UNITS_S:.0f} units/s")
        if self.tolerance_units != TOLERANCE_UNITS:
            raise ValueError(f"tolerance is fixed at {TOLERANCE_UNITS:.0f} units for this routine")
        if not 20 <= self.n_poses <= 99:
            raise ValueError("n_poses must be between 20 and 99")
        if self.loo_gate_mm > LOO_GATE_MM:
            raise ValueError("the leave-one-out gate may not exceed 10 mm")


@dataclass(frozen=True)
class CalPose:
    index: int
    xy: tuple
    n_body: tuple


@dataclass
class Sample:
    index: int
    xy_cmd: list
    n_cmd: list
    n_meas: list
    uv: Optional[list]
    max_error: float
    status: str


def _arange(lo: float, hi: float, step: float) -> np.ndarray:
    return np.arange(lo, hi + 1e-9, step)


def _spread(cands: list, k: int) -> list:
    """Farthest-point sampling of k candidates on the table, starting from the one nearest the centre."""
    if len(cands) <= k:
        return list(cands)
    pts = np.array([[c[0], c[1]] for c in cands])
    first = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    chosen = [first]
    d = np.linalg.norm(pts - pts[first], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(pts - pts[nxt], axis=1))
    return [cands[i] for i in chosen]


def _greedy_path(items: list, start_xy: np.ndarray) -> list:
    left, path, cur = list(items), [], np.asarray(start_xy, dtype=float)
    while left:
        j = int(np.argmin([math.hypot(c[0] - cur[0], c[1] - cur[1]) for c in left]))
        c = left.pop(j)
        path.append(c)
        cur = np.array([c[0], c[1]])
    return path


def plan_poses(chain: Chain, cal: FollowerCalibration, cfg: GridConfig, plates: Sequence = ()) -> list[CalPose]:
    """Top-down IK poses on one table height, away from the base and the plate discs, at most cfg.n_poses."""
    lo, hi = cal.rad_bounds(chain, cfg.margin_units)
    q_ref = cal.rad_from_norm(np.asarray(HOME_NORM, dtype=float))
    z = TABLE_Z_BASE_M + cfg.z_above_table_m
    cands = []
    for x in _arange(*cfg.x_range_m, cfg.step_m):
        for y in _arange(*cfg.y_range_m, cfg.step_m):
            if math.hypot(x - PAN_AXIS_XY[0], y - PAN_AXIS_XY[1]) < BASE_KEEPOUT_M:
                continue
            if any(math.hypot(x - px, y - py) < pr + cfg.plate_margin_m for px, py, pr in plates):
                continue
            res = solve_top_down(chain, cal, float(x), float(y), z, cfg.psi_rad, lo, hi, q_ref)
            if not res.ok(IK_POS_TOL_M, IK_ROT_TOL_RAD):
                continue
            cands.append((float(x), float(y), cal.norm_from_rad(res.q)[:N_BODY]))
    home_xy = tcp_xyz(chain, cal, np.asarray(HOME_NORM, dtype=float))[:2]
    ordered = _greedy_path(_spread(cands, cfg.n_poses), home_xy)
    return [CalPose(i, (x, y), tuple(float(v) for v in n)) for i, (x, y, n) in enumerate(ordered)]


def _red_mask(rgb: np.ndarray) -> np.ndarray:
    """Red mask with the OpenCV HSV thresholds (hue within 8 of 0 on the 0-179 scale, S >= 120, V >= 80), numpy only."""
    x = rgb.astype(np.float64) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    mx, mn = np.max(x, axis=-1), np.min(x, axis=-1)
    d = mx - mn
    safe = np.maximum(d, 1e-12)
    hue_deg = np.where(mx == r, np.mod((g - b) / safe, 6.0),
                       np.where(mx == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0)) * 60.0
    hue_deg = np.where(d > 0, hue_deg, 0.0)
    s255 = np.where(mx > 0, d / np.maximum(mx, 1e-12) * 255.0, 0.0)
    v255 = mx * 255.0
    h_cv = np.round(hue_deg / 2.0)  # OpenCV stores 8-bit hue as whole units on the 0-179 scale
    return ((h_cv <= 8) | (h_cv >= 170)) & (s255 >= 120) & (v255 >= 80)


def _red_components(rgb: np.ndarray) -> list:
    """8-connected components of the red mask as lists of (row, col); numpy and Python only."""
    ys, xs = np.nonzero(_red_mask(rgb))
    pts = set(zip(ys.tolist(), xs.tolist()))
    seen, comps = set(), []
    for start in pts:
        if start in seen:
            continue
        seen.add(start)
        stack, comp = [start], []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pts and q not in seen:
                        seen.add(q)
                        stack.append(q)
        comps.append(comp)
    return comps


def detect_tip(rgb: np.ndarray, min_area: int = 20, max_area: int = 6000) -> Optional[tuple[float, float]]:
    """Red jaw tip in the top view: centroid of the largest red blob, or None if absent or ambiguous.

    Same thresholds as locate.detect_red_tip (hue within 8 of 0, saturation >= 120, value >= 80), but
    two red blobs of comparable size return None instead of a mean position.
    """
    comps = _red_components(rgb)
    if not comps:
        return None
    comps.sort(key=len, reverse=True)
    big = len(comps[0])
    if big < min_area or big > max_area:
        return None
    if len(comps) > 1 and len(comps[1]) > 0.3 * big:
        return None
    pts = np.asarray(comps[0], dtype=float)
    return float(pts[:, 1].mean()), float(pts[:, 0].mean())


def run_grid(player, poses: Sequence[CalPose], capture_rgb: Callable[[], np.ndarray], cfg: GridConfig,
             log: Callable[[str], None] = print) -> list[Sample]:
    """Move to each pose through MotionPlayer; keep the read-back pose and the tip pixel when it arrived."""
    out: list[Sample] = []
    for pose in poses:
        gripper = float(player.current()[5])
        goal = np.concatenate([np.asarray(pose.n_body, dtype=float), [gripper]])
        res = player.move_to(goal, max_speed=cfg.speed_units_s)
        if not res.arrived:
            out.append(Sample(pose.index, list(pose.xy), list(pose.n_body), [], None, float(res.max_error), "not_arrived"))
            log(f"pose {pose.index}: not arrived ({res.max_error:.2f} units from goal), dropped")
            continue
        time.sleep(cfg.settle_s)
        uv = detect_tip(capture_rgb())
        meas = np.asarray(player.current(), dtype=float)[:N_BODY]
        status = "ok" if uv is not None else "no_detection"
        out.append(Sample(pose.index, list(pose.xy), list(pose.n_body), meas.tolist(),
                          None if uv is None else [uv[0], uv[1]], float(res.max_error), status))
        log(f"pose {pose.index}: {status}")
    return out


def _unpack(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.append(p[:8], 1.0).reshape(3, 3), p[8:8 + N_BODY]


def _tip_xy(chain: Chain, cal: FollowerCalibration, n_meas: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.array([tcp_xyz(chain, cal, n + offsets)[:2] for n in n_meas])


class _Problem:
    def __init__(self, chain, cal, n_meas, uv, ax, au, anchor_weight):
        self.chain, self.cal = chain, cal
        self.n_meas, self.uv = n_meas, uv
        self.ax, self.au, self.w = ax, au, anchor_weight

    def residuals(self, p: np.ndarray) -> np.ndarray:
        H, b = _unpack(p)
        parts = [(apply_homography(H, _tip_xy(self.chain, self.cal, self.n_meas, b)) - self.uv).ravel()]
        if len(self.ax):
            parts.append(self.w * (apply_homography(H, self.ax) - self.au).ravel())
        return np.concatenate(parts)


def _jacobian(f: Callable[[np.ndarray], np.ndarray], p: np.ndarray, r0: np.ndarray) -> np.ndarray:
    J = np.empty((r0.size, p.size))
    for j in range(p.size):
        h = 1e-6 * max(1.0, abs(p[j])) if j < 8 else 1e-3
        dp = np.zeros_like(p)
        dp[j] = h
        J[:, j] = (f(p + dp) - r0) / h
    return J


def _solve(prob: _Problem, iters: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg-Marquardt on [homography (8), per-joint zero offsets (5)]."""
    xy0 = _tip_xy(prob.chain, prob.cal, prob.n_meas, np.zeros(N_BODY))
    p = np.concatenate([fit_homography(xy0, prob.uv).ravel()[:8], np.zeros(N_BODY)])
    r = prob.residuals(p)
    cost, lam, step = float(r @ r), 1e-3, np.zeros_like(p)
    for _ in range(iters):
        J = _jacobian(prob.residuals, p, r)
        A, g = J.T @ J, J.T @ r
        accepted = False
        for _ in range(15):
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            p_try = p + step
            r_try = prob.residuals(p_try)
            c_try = float(r_try @ r_try)
            if c_try < cost:
                p, r, cost, lam, accepted = p_try, r_try, c_try, max(lam / 3.0, 1e-12), True
                break
            lam *= 4.0
        if not accepted or np.linalg.norm(step) < 1e-10:
            break
    return p, r


def _identifiability(prob: _Problem, p: np.ndarray) -> dict:
    r0 = prob.residuals(p)
    J = _jacobian(prob.residuals, p, r0)
    JH = J[:, :8]
    out = {}
    for j, name in enumerate(OFFSET_NAMES):
        col = J[:, 8 + j]
        coef = np.linalg.lstsq(JH, col, rcond=None)[0]
        outside = float(np.linalg.norm(col - JH @ coef) / max(np.linalg.norm(col), 1e-12))
        out[name] = {"outside_homography_span": outside, "identifiable": bool(outside > 1e-3)}
    return out


def _loo_mm(chain, cal, n_meas, uv, ax, au, anchor_weight) -> np.ndarray:
    errs = []
    for i in range(len(uv)):
        keep = np.arange(len(uv)) != i
        p, _ = _solve(_Problem(chain, cal, n_meas[keep], uv[keep], ax, au, anchor_weight))
        H, b = _unpack(p)
        xy_pred = tcp_xyz(chain, cal, n_meas[i] + b)[:2]
        xy_meas = apply_homography(np.linalg.inv(H), uv[i:i + 1])[0]
        errs.append(1000.0 * float(np.linalg.norm(xy_meas - xy_pred)))
    return np.array(errs)


def calibrate(chain: Chain, cal: FollowerCalibration, samples: Sequence[Sample], *,
              anchors_xy=(), anchors_uv=(), cfg: GridConfig = GridConfig()) -> dict:
    """Fit homography and zero offsets on the usable samples, then apply the leave-one-out gate."""
    ok = [s for s in samples if s.status == "ok"]
    if len(ok) < cfg.min_samples:
        return {"status": "FAILED", "reason": f"{len(ok)} usable samples, need {cfg.min_samples}", "usable": len(ok)}
    n_meas = np.array([s.n_meas for s in ok], dtype=float)
    uv = np.array([s.uv for s in ok], dtype=float)
    ax = np.asarray(anchors_xy, dtype=float).reshape(-1, 2)
    au = np.asarray(anchors_uv, dtype=float).reshape(-1, 2)
    try:
        prob = _Problem(chain, cal, n_meas, uv, ax, au, cfg.anchor_weight)
        p, r = _solve(prob)
        H, b = _unpack(p)
        errs = _loo_mm(chain, cal, n_meas, uv, ax, au, cfg.anchor_weight)
        ident = _identifiability(prob, p)
    except (np.linalg.LinAlgError, ValueError) as exc:
        return {"status": "FAILED", "reason": f"fit did not converge: {exc}", "usable": len(ok)}
    tip_px = np.linalg.norm(r[:2 * len(ok)].reshape(-1, 2), axis=1)
    rms = float(np.sqrt(np.mean(errs ** 2)))
    tip_z = np.array([tcp_xyz(chain, cal, n + b)[2] for n in n_meas])
    return {
        "status": "PASS" if rms <= cfg.loo_gate_mm else "FAIL",
        "usable": len(ok),
        "anchors": int(len(ax)),
        "homography_table_to_px": H.tolist(),
        "offsets_normalized_units": dict(zip(OFFSET_NAMES, b.tolist())),
        "zero_shift_rad": dict(zip(OFFSET_NAMES, (cal.sign * cal.k * b).tolist())),
        "in_sample_px_rms": float(np.sqrt(np.mean(tip_px ** 2))),
        "in_sample_px_max": float(tip_px.max()),
        "loo_rms_mm": rms,
        "loo_median_mm": float(np.median(errs)),
        "loo_max_mm": float(errs.max()),
        "loo_gate_mm": cfg.loo_gate_mm,
        "loo_per_sample_mm": errs.tolist(),
        "tip_z_m_mean": float(tip_z.mean()),
        "tip_z_m_std": float(tip_z.std()),
        "identifiability": ident,
        "scope": ("XY pixel consistency at the single calibration height only. The gate does not validate the "
                  "per-joint offsets: offsets that move the tip only in z are not observed by a top camera at one "
                  "height (dry run: gate PASS with wrong offsets). Do not apply the offsets to grasp-height IK "
                  "without a second height or an independent check."),
    }


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_new(out_dir: Path, stem: str, payload: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}_{_utc()}.json"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


GEOMETRY_BUDGET_FALLBACK_M = 0.032


def load_priors(path: Optional[str]) -> dict:
    """Plate and pixel priors from the sim2 hand-off (geometry.json). Its frame is the MJCF base, not the URDF chain used here."""
    if not path:
        return {}
    g = json.loads(Path(path).read_text(encoding="utf-8"))
    plates = {name: {"xy_m": [float(v) for v in p["centre_xy_m"]], "n": int(p["n"]),
                     "sd_xy_m": [float(v) for v in p["sd_xy_m"][:2]]}
              for name, p in g["plates_in_robot_base_frame_m"].items()}
    budget = float(g.get("uncertainty_horizontal_m", {}).get("budget_1sigma_m", GEOMETRY_BUDGET_FALLBACK_M))
    px = g.get("top_view_pixels_640x480", {}).get("green_plate_real_top_camera")
    return {"source": str(path), "frame": g.get("frame", ""), "plates": plates, "budget_m": budget,
            "green_uv_px": None if px is None else [float(px[0]), float(px[1])]}


def plates_from_priors(priors: dict, radius_m: float = 0.06) -> list:
    """Keep-out discs: plate radius plus the horizontal budget (the prior positions are not verified)."""
    return [(p["xy_m"][0], p["xy_m"][1], radius_m + priors["budget_m"]) for p in priors.get("plates", {}).values()]


def prior_check(H: np.ndarray, priors: dict) -> dict:
    """Map the measured green-plate pixel to the table with a fit that did not use it, then compare with the prior."""
    if not priors or not priors.get("green_uv_px") or "green" not in priors.get("plates", {}):
        return {"status": "not checked"}
    xy_fit = apply_homography(np.linalg.inv(H), np.array([priors["green_uv_px"]], dtype=float))[0]
    prior_xy = np.asarray(priors["plates"]["green"]["xy_m"], dtype=float)
    d_mm = 1000.0 * float(np.linalg.norm(xy_fit - prior_xy))
    return {"green_pixel_table_xy_m": [float(v) for v in xy_fit], "green_prior_xy_m": prior_xy.tolist(),
            "difference_mm": d_mm, "budget_mm": 1000.0 * priors["budget_m"],
            "within_budget": bool(d_mm <= 1000.0 * priors["budget_m"]),
            "note": "green pixel is an approximate mask centroid (about 15 px); the prior is in the MJCF base frame"}


def finish(*, mode, out_dir, chain, cal, cfg, poses, samples, plates, anchor_sets, extra, priors=None) -> dict:
    results = {name: calibrate(chain, cal, samples, anchors_xy=ax, anchors_uv=au, cfg=cfg)
               for name, (ax, au) in anchor_sets.items()}
    for res in results.values():
        if priors and res.get("homography_table_to_px"):
            res["prior_check"] = prior_check(np.asarray(res["homography_table_to_px"], dtype=float), priors)
    if priors:
        extra = {**extra, "priors_source": priors.get("source"), "priors_frame": priors.get("frame")}
    payload = {"mode": mode, "utc": _utc(), "config": asdict(cfg), "plates": [list(p) for p in plates],
               "poses": [asdict(p) for p in poses], "samples": [asdict(s) for s in samples],
               "results": results, **extra}
    path = _write_new(Path(out_dir), f"tip_calib_{mode}", payload)
    return {"path": str(path), "results": results}


def run_session(*, mode, chain, cal, player, capture_rgb, plates, cfg, log=print):
    poses = plan_poses(chain, cal, cfg, plates)
    log(f"[{mode}] {len(poses)} poses at z {TABLE_Z_BASE_M + cfg.z_above_table_m:.3f} m, speed {cfg.speed_units_s:g}, tolerance {cfg.tolerance_units:g}")
    samples = run_grid(player, poses, capture_rgb, cfg, log)
    return poses, samples


def import_motion_layer(motion_dir: Path):
    """Import so101_motion from R/app/motion (the only place that touches the arm)."""
    if str(motion_dir) not in sys.path:
        sys.path.insert(0, str(motion_dir))
    import so101_motion  # noqa: PLC0415

    return so101_motion


def realsense_top_capture(serial: str, width: int = 640, height: int = 480, fps: int = 30):
    """RGB frames from the top RealSense. Not run in this session (no camera access); Opus verifies it."""
    import pyrealsense2 as rs  # noqa: PLC0415

    pipe, config = rs.pipeline(), rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    pipe.start(config)

    def grab() -> np.ndarray:
        for _ in range(5):
            frames = pipe.wait_for_frames()
        return np.asanyarray(frames.get_color_frame().get_data()).copy()

    return grab, pipe.stop


def _parse_plates(items: Sequence[str]) -> list[tuple[float, float, float]]:
    out = []
    for item in items:
        x, y, r = (float(v) for v in item.split(","))
        out.append((x, y, r))
    return out


def _load_anchors(path: Optional[str]) -> tuple[list, list]:
    if not path:
        return [], []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [a["xy_m"] for a in data], [a["uv_px"] for a in data]


def dry_run(args, cfg: GridConfig) -> dict:
    from . import fake_bus as fb  # noqa: PLC0415

    motion = import_motion_layer(Path(args.motion_dir))
    chain = Chain(args.urdf)
    cal = FollowerCalibration.from_studio_json(args.robots, "SO101_Follower")
    plates = fb.DRY_PLATES
    follower = fb.FakeFollower(np.concatenate([np.asarray(HOME_NORM, dtype=float), [30.0]]))
    camera = fb.FakeTopCamera(follower, chain, cal, fb.H_TRUE, fb.B_TRUE, noise_px=0.5, seed=0)
    player = motion.MotionPlayer(follower, max_speed=cfg.speed_units_s, tolerance=TOLERANCE_UNITS)
    poses, samples = run_session(mode="dry_run", chain=chain, cal=cal, player=player,
                                 capture_rgb=camera, plates=plates, cfg=cfg)
    plate_xy = [[p[0], p[1]] for p in plates]
    plate_uv = apply_homography(fb.H_TRUE, np.array(plate_xy)).tolist()
    return finish(mode="dry_run", out_dir=args.out, chain=chain, cal=cal, cfg=cfg, poses=poses,
                  samples=samples, plates=plates,
                  anchor_sets={"tip_only": ([], []), "plate_anchors": (plate_xy, plate_uv)},
                  extra={"fake_bus": True, "port_opened": False, "camera_opened": False})


def live(args, cfg: GridConfig, plates, anchors, priors=None) -> dict:
    motion = import_motion_layer(Path(args.motion_dir))
    chain = Chain(args.urdf)
    cal = FollowerCalibration.from_studio_json(args.robots, "SO101_Follower")
    # The follower port is always the by-id path; the export's connection_string (/dev/ttyACM1) is stale.
    port, calibration = motion.resolve_arm(studio_export=args.robots, name=args.robot_name, follower=True)
    if not (port.startswith("/dev/serial/by-id/") and PORT_SERIAL in port):
        raise SystemExit(f"refusing: follower port {port!r} is not the by-id path for serial {PORT_SERIAL}")
    grab, stop_camera = realsense_top_capture(args.camera_serial)
    driver = None
    torque_on = False
    try:
        answer = input("No hands or people in the workspace (I-43 rule 3)? Type YES to start motion: ")
        if answer.strip() != "YES":
            raise SystemExit("not confirmed: no port opened, no motion")
        driver = motion.open_follower_held(port, calibration, unit="normalized")
        torque_on = True
        player = motion.MotionPlayer(driver, max_speed=cfg.speed_units_s, tolerance=TOLERANCE_UNITS)
        poses, samples = run_session(mode="live", chain=chain, cal=cal, player=player,
                                     capture_rgb=grab, plates=plates, cfg=cfg)
        home = np.concatenate([np.asarray(HOME_NORM, dtype=float), [float(player.current()[5])]])
        player.move_to(home, max_speed=cfg.speed_units_s)
        input("Support the arm by hand, then press Enter to switch torque off: ")
        driver.set_torque(enabled=False)
        torque_on = False
        # disconnect() re-enables torque while torque_on_disconnect is True (plugin so101.py:457-470).
        driver.torque_on_disconnect = False
        return finish(mode="live", out_dir=args.out, chain=chain, cal=cal, cfg=cfg, poses=poses,
                      samples=samples, plates=plates,
                      anchor_sets={"tip_only": ([], []), "with_anchors": anchors},
                      extra={"port": port, "torque_off_at_end": True}, priors=priors)
    except BaseException:
        if torque_on:
            print("error: torque is still on and the arm is holding; support it by hand before releasing", file=sys.stderr)
        raise
    finally:
        with contextlib.suppress(Exception):
            stop_camera()
        if driver is not None:
            with contextlib.suppress(Exception):
                driver.disconnect()


def refit(args, cfg: GridConfig) -> dict:
    chain = Chain(args.urdf)
    cal = FollowerCalibration.from_studio_json(args.robots, "SO101_Follower")
    data = json.loads(Path(args.samples).read_text(encoding="utf-8"))
    samples = [Sample(**s) for s in data["samples"]]
    ax, au = _load_anchors(args.anchors)
    res = calibrate(chain, cal, samples, anchors_xy=ax, anchors_uv=au, cfg=cfg)
    path = _write_new(Path(args.out), "tip_calib_refit", {"source": args.samples, "result": res})
    return {"path": str(path), "results": {"refit": res}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pick_v2.tip_calib")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--urdf", default=URDF_DEFAULT)
        p.add_argument("--robots", default=str(MOTION_DIR / "project_robots.json"))
        p.add_argument("--robot-name", default=ROBOT_NAME)
        p.add_argument("--motion-dir", default=str(MOTION_DIR))
        p.add_argument("--out", required=True)
        p.add_argument("--speed", type=float, default=15.0, help="units/s, allowed 15 to 20")
        p.add_argument("--n-poses", type=int, default=40)
        p.add_argument("--geometry", default=None, help="sim2 geometry.json: plate priors and the green pixel check")
        p.add_argument("--plate", action="append", default=[], metavar="X,Y,R",
                       help="plate disc centre and radius in base_link metres; repeat per plate")
        p.add_argument("--anchors", default=None, help="JSON list of {\"xy_m\": [x, y], \"uv_px\": [u, v]}")

    common(sub.add_parser("dry-run"))
    live_p = sub.add_parser("live")
    common(live_p)
    live_p.add_argument("--confirm-motion", action="store_true")
    live_p.add_argument("--camera-serial", default=TOP_CAMERA_SERIAL)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--urdf", default=URDF_DEFAULT)
    fit_p.add_argument("--robots", default=str(MOTION_DIR / "project_robots.json"))
    fit_p.add_argument("--samples", required=True)
    fit_p.add_argument("--anchors", default=None)
    fit_p.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "fit":
        out = refit(args, GridConfig())
    else:
        cfg = GridConfig(speed_units_s=args.speed, n_poses=args.n_poses)
        if args.cmd == "live":
            if not args.confirm_motion:
                ap.error("live motion needs --confirm-motion")
            priors = load_priors(args.geometry)
            plates = _parse_plates(args.plate) or plates_from_priors(priors)
            if not plates:
                ap.error("live motion needs --plate X,Y,R or --geometry (plate discs)")
            if not sys.stdin.isatty():
                ap.error("live motion needs an interactive terminal for the torque-off step")
            ax, au = _load_anchors(args.anchors)
            out = live(args, cfg, plates, (ax, au), priors)
        else:
            out = dry_run(args, cfg)
    print(json.dumps({"path": out["path"], "status": {k: v.get("status") for k, v in out["results"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
