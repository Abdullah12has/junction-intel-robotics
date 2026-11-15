"""P2-B-DET2 (E3) per-brick inspection API, D-068 version. CPU only (OpenCV + NumPy); no camera, robot or Studio access.

scan(top_rgb, close_rgb=None) -> list of dicts, one per segmented brick per view. The top view gives the footprint
shape; the close view gives dark grey tape. Views are NOT fused into one brick (no cross-camera calibration).
Routing (I-52 / OD-10, fail-safe): CONFIDENT_DEFECT -> GREEN; every other verdict -> INSPECT. Nothing routes to BLUE.
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import det2  # noqa: E402

TOP_KEY, CLOSE_KEY = "243622061421", "243722064316"
FRAME_HW = (480, 640)
TOP_CONF = 0.5             # shape_conf needed for a top-view verdict
TAPE_FLAG = 0.05           # dark-grey fraction that flags a close brick; fitted on the Opus close labels (in-sample)
TAPE_CONFIDENT = 0.10      # above this, an unmerged and untouched close brick is CONFIDENT_DEFECT
CLOSE_MERGED_AREA = 25000  # single close bricks seen so far are <= ~18000 px
HAND_MIN_AREA = 4000
CONF_DEFECT, CONF_GOOD, UNC = "CONFIDENT_DEFECT", "CONFIDENT_GOOD", "UNCERTAIN"
GOOD, DEFECTIVE, UNCERTAIN = "GOOD", "DEFECTIVE", "UNCERTAIN"
_CACHE = {}


def _model():
    if "m" not in _CACHE:
        _CACHE["m"] = {"views": {v: det2.load_model(v) for v in ("top", "close")}, "cal": det2.load_cal()}
    return _CACHE["m"]


def view_of(camera_key):
    return "top" if camera_key and TOP_KEY in str(camera_key) else "close"


def route_of(verdict):
    return "GREEN" if verdict == CONF_DEFECT else "INSPECT"


def route(bricks):
    """GREEN if any record is CONFIDENT_DEFECT; otherwise INSPECT (uncertain, confident-good, no brick, refused)."""
    return "GREEN" if any(b.get("route") == "GREEN" for b in bricks) else "INSPECT"


def hands_present(bgr, roi=None):
    """Skin-coloured blob of at least HAND_MIN_AREA px, optionally restricted to the mat ROI."""
    hsv = det2.hsv_of(bgr)
    skin = (hsv[..., 0] <= 20) & (hsv[..., 1] >= 40) & (hsv[..., 1] <= 170) & (hsv[..., 2] >= 120)
    if roi is not None:
        skin &= roi
    skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(skin, 8)
    return bool(n > 1 and st[1:, cv2.CC_STAT_AREA].max() >= HAND_MIN_AREA)


def _verdict(rec, view):
    reasons = []
    touch = bool(rec["touch_edge"])
    mixed = bool(rec.get("mixed_colour"))
    if touch:
        reasons.append("touches_edge")
    if mixed:
        reasons.append("mixed_colour_blob")
    shape, sconf, tape, tconf = None, None, None, None
    if view == "top":
        shape, sconf = rec.get("shape", "unknown"), float(rec.get("shape_conf", 0.0))
        if shape == "unknown":
            reasons.append("shape_unknown")
        if touch or mixed:
            verdict = UNC
        elif shape == "not_2x2" and sconf >= TOP_CONF:
            verdict = CONF_DEFECT
            reasons.append("not_2x2")
        elif shape == "2x2" and sconf >= TOP_CONF:
            verdict = CONF_GOOD
        else:
            verdict = UNC
    else:
        dg = float(rec["tape_feats"]["dark_grey"])
        tape = dg
        tconf = round(min(1.0, max(0.0, (dg - TAPE_FLAG) / (TAPE_CONFIDENT - TAPE_FLAG))), 3)
        merged = rec["area"] > CLOSE_MERGED_AREA
        if merged:
            reasons.append("merged_blob")
        if mixed:
            reasons.append("mixed_colour_info")  # informational in the close view: light bricks have bright highlights
        if dg >= TAPE_FLAG:
            reasons.append("tape_flag")
        if touch or merged:
            verdict = UNC
        elif dg >= TAPE_CONFIDENT:
            verdict = CONF_DEFECT
            reasons.append("tape_confident")
        else:
            verdict = UNC
            reasons.append("shape_not_assessed_in_close")
    x, y, w, h = rec["bbox"]
    return {"id": f"scan_{view}_{rec['k']}", "view": view, "colour": rec["colour"],
            "top_bbox": rec["bbox"] if view == "top" else None, "bbox": rec["bbox"],
            "centre_px": rec["centre_px"], "angle_deg": rec["angle_deg"], "footprint_px": rec["footprint_px"],
            "area": rec["area"], "shape": shape, "shape_conf": None if sconf is None else round(sconf, 3),
            "tape_score": None if tape is None else round(tape, 4), "tape_conf": tconf,
            "verdict": verdict, "confident": verdict in (CONF_DEFECT, CONF_GOOD), "route": route_of(verdict),
            "reasons": reasons, "crop_paths": []}


def _no_brick(view, reason):
    return {"id": f"scan_{view}_{reason}", "view": view, "colour": None, "top_bbox": None, "bbox": None,
            "centre_px": None, "angle_deg": None, "footprint_px": None, "area": 0, "shape": None,
            "shape_conf": None, "tape_score": None, "tape_conf": None, "verdict": UNC, "confident": False,
            "route": "INSPECT", "reasons": [reason], "crop_paths": []}


def scan(top_rgb=None, close_rgb=None):
    """All bricks in the given views. A view with a hand present is refused; an empty view gives one UNCERTAIN entry."""
    m = _model()
    out = []
    for view, img in (("top", top_rgb), ("close", close_rgb)):
        if img is None:
            continue
        if img.shape[:2] != FRAME_HW:
            raise ValueError(f"expects {FRAME_HW[1]}x{FRAME_HW[0]} frames, got {img.shape[1]}x{img.shape[0]}")
        roi = m["views"]["top"]["roi"] if view == "top" else None
        if view == "top" and hands_present(img, roi):
            out.append(_no_brick(view, "hands_present"))
            continue
        recs = det2.analyse(img, view, m["views"][view], a_ref=m["cal"]["a_ref_top_px"])
        if not recs:
            out.append(_no_brick(view, "no_brick"))
            continue
        out.extend(_verdict(r, view) for r in recs)
    return out


def find_target(top_rgb, colour="red"):
    """Largest unmerged, untouched top-view 2x2 of the given colour, or None (for 'move the red brick' texts)."""
    cands = [b for b in scan(top_rgb=top_rgb) if b.get("colour") == colour and b.get("shape") == "2x2"
             and "touches_edge" not in b["reasons"] and "mixed_colour_blob" not in b["reasons"]]
    return max(cands, key=lambda b: b["area"]) if cands else None


class Inspector:
    """LOOP-CONTRACT section 3 surface (inspect_rgb, locate_brick). Rule backend on CPU; the anomalib backend is not in this pass."""

    def __init__(self, device="CPU"):
        self.device = device
        self.m = _model()

    def _recs(self, rgb, camera_key):
        view = view_of(camera_key)
        roi = self.m["views"]["top"]["roi"] if view == "top" else None
        if view == "top" and hands_present(rgb, roi):
            return view, None, "hands_present"
        recs = det2.analyse(rgb, view, self.m["views"][view], a_ref=self.m["cal"]["a_ref_top_px"])
        if not recs:
            return view, None, "no_brick"
        return view, recs, None

    def locate_brick(self, rgb, camera_key=None):
        _, recs, _ = self._recs(rgb, camera_key)
        if recs is None:
            return None
        return max(recs, key=lambda r: r["area"])["bbox"]

    def inspect_rgb(self, rgb, camera_key=None):
        t0 = time.perf_counter()
        view, recs, why = self._recs(rgb, camera_key)
        if recs is None:
            res = {"label": UNCERTAIN, "route": "INSPECT", "score": None, "reasons": [why], "crop": None, "geometry": {}}
        elif len(recs) > 1:
            res = {"label": UNCERTAIN, "route": "INSPECT", "score": None, "reasons": ["multiple_bricks"],
                   "crop": None, "geometry": {}}
        else:
            v = _verdict(recs[0], view)
            x, y, w, h = recs[0]["bbox"]
            res = {"label": {CONF_DEFECT: DEFECTIVE, CONF_GOOD: GOOD}.get(v["verdict"], UNCERTAIN),
                   "route": v["route"], "score": v["tape_score"], "reasons": v["reasons"],
                   "crop": rgb[y:y + h, x:x + w].copy(),
                   "geometry": {"view": view, "shape": v["shape"], "footprint_px": v["footprint_px"], "area": v["area"]}}
        res.update({"threshold": {"tape_flag": TAPE_FLAG, "tape_confident": TAPE_CONFIDENT},
                    "device": self.device, "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2)})
        return res


if __name__ == "__main__":
    print("use scan_once.py --top PATH [--close PATH] --out DIR")
