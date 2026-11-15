"""Top-view calibration: pixel <-> table-plane mapping from correspondences (DLT), residual report, active grid.

Two models are fitted from the same correspondences:
- fit_homography: a plane-to-image homography for points that lie on one plane (for example the TCP at one height).
- fit_camera: a 3x4 projection matrix from 3-D TCP positions (FK) to pixels. plane_homography(P, z) then gives the
  pixel <-> (x, y) mapping for any table height z, which is what a brick top needs (parallax-correct).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


def _norm_2d(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = float(np.sqrt(2.0)) / max(d, 1e-12)
    return np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])


def _norm_3d(pts: np.ndarray) -> np.ndarray:
    c = pts.mean(axis=0)
    d = float(np.mean(np.linalg.norm(pts - c, axis=1)))
    s = float(np.sqrt(3.0)) / max(d, 1e-12)
    T = np.eye(4)
    T[0, 0] = T[1, 1] = T[2, 2] = s
    T[:3, 3] = -s * c
    return T


def fit_homography(src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> np.ndarray:
    """3x3 H with dst ~ H @ src (normalized DLT). Needs at least 4 non-degenerate correspondences."""
    s = np.asarray(src, dtype=float)
    d = np.asarray(dst, dtype=float)
    if len(s) < 4 or len(s) != len(d):
        raise ValueError("need at least 4 matching point pairs")
    Ts, Td = _norm_2d(s), _norm_2d(d)
    sh = np.column_stack([s, np.ones(len(s))]) @ Ts.T
    dh = np.column_stack([d, np.ones(len(d))]) @ Td.T
    A = []
    for (x, y, _), (u, v, _) in zip(sh, dh):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    Hn = vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def apply_homography(H: np.ndarray, pts: Sequence[Sequence[float]]) -> np.ndarray:
    p = np.column_stack([np.asarray(pts, dtype=float), np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def fit_camera(xyz: Sequence[Sequence[float]], uv: Sequence[Sequence[float]]) -> np.ndarray:
    """3x4 projection matrix P with uv ~ P @ [x, y, z, 1] (normalized DLT). Needs at least 6 non-coplanar points."""
    X = np.asarray(xyz, dtype=float)
    U = np.asarray(uv, dtype=float)
    if len(X) < 6 or len(X) != len(U):
        raise ValueError("need at least 6 matching 3-D/2-D pairs")
    T3 = _norm_3d(X)
    Tuv = _norm_2d(U)
    Xn = (T3 @ np.column_stack([X, np.ones(len(X))]).T).T
    Un = (Tuv @ np.column_stack([U, np.ones(len(U))]).T).T
    A = []
    for (x, y, z, _), (u, v, _) in zip(Xn, Un):
        A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    _, _, vt = np.linalg.svd(np.asarray(A))
    Pn = vt[-1].reshape(3, 4)
    P = np.linalg.inv(Tuv) @ Pn @ T3
    return P / np.linalg.norm(P[2, :3])


def project_camera(P: np.ndarray, xyz: Sequence[Sequence[float]]) -> np.ndarray:
    X = np.column_stack([np.asarray(xyz, dtype=float), np.ones(len(xyz))]) @ P.T
    return X[:, :2] / X[:, 2:3]


def plane_homography(P: np.ndarray, z: float) -> np.ndarray:
    """H with [u, v, w] ~ H @ [x, y, 1] for points on the plane at height z."""
    return np.column_stack([P[:, 0], P[:, 1], z * P[:, 2] + P[:, 3]])


def pixel_to_plane(P: np.ndarray, uv: Sequence[float], z: float) -> np.ndarray:
    Hi = np.linalg.inv(plane_homography(P, z))
    p = Hi @ np.array([uv[0], uv[1], 1.0])
    return p[:2] / p[2]


def pixel_dir_to_plane(P: np.ndarray, uv: Sequence[float], direction_px: Sequence[float], z: float) -> np.ndarray:
    """World (x, y) unit direction that a pixel direction maps to on the plane at height z."""
    d = np.asarray(direction_px, dtype=float)
    d = d / max(np.linalg.norm(d), 1e-12)
    eps = 1.0
    p0 = pixel_to_plane(P, uv, z)
    p1 = pixel_to_plane(P, (uv[0] + eps * d[0], uv[1] + eps * d[1]), z)
    v = p1 - p0
    return v / max(np.linalg.norm(v), 1e-12)


@dataclass(frozen=True)
class Residuals:
    rms_px: float
    mean_px: float
    max_px: float
    n: int


def residuals(pred_uv: np.ndarray, meas_uv: np.ndarray) -> Residuals:
    e = np.linalg.norm(np.asarray(pred_uv) - np.asarray(meas_uv), axis=1)
    return Residuals(rms_px=float(np.sqrt(np.mean(e * e))), mean_px=float(np.mean(e)), max_px=float(np.max(e)), n=len(e))


def active_grid(xs: Iterable[float], ys: Iterable[float], z: float) -> np.ndarray:
    """Hover targets (x, y, z) for the active self-calibration; reachability is filtered by the caller."""
    return np.array([(float(x), float(y), float(z)) for x in xs for y in ys])


def collect_correspondences(
    targets: Sequence[Sequence[float]],
    move_fn: Callable[[Sequence[float]], bool],
    detect_fn: Callable[[], Optional[Sequence[float]]],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Visit each target, detect the red TCP in the top view and keep (TCP xyz from FK, detected pixel) pairs.

    move_fn(target) moves the arm (the caller converts xyz to joints with IK and to normalized units) and returns
    True when the arm arrived. detect_fn() returns (u, v) of the TCP or None. The TCP xyz kept is the target, which
    is the FK position the arm was commanded to reach (its residual is checked by the caller).
    """
    xyz, uv, skipped = [], [], 0
    for t in targets:
        if not move_fn(t):
            skipped += 1
            continue
        px = detect_fn()
        if px is None:
            skipped += 1
            continue
        xyz.append(tuple(t))
        uv.append(tuple(px))
    return np.asarray(xyz, dtype=float), np.asarray(uv, dtype=float), skipped
