#!/usr/bin/env python3
"""Depth grab for the two RealSense D415 cameras.

Reads depth (gray16le) and the left IR image through plain V4L2 (kernel uvcvideo driver). It does not use
librealsense and does not detach any driver, so the monitor's colour streams (video8 top, video14 close)
keep running. D400 default depth unit is 1 mm, so a raw value is millimetres.

Outputs in --out:
  <cam>_depth.npy   float32 mm, per-pixel median of the grabbed frames, NaN = no depth
  <cam>_fill.npy    fraction of frames with valid depth per pixel
  <cam>_std.npy     per-pixel std of valid depth (mm)
  <cam>_ir.png      left IR frame (pixel-aligned with depth)
  <cam>_color.jpg   live colour frame copied at grab time (NOT pixel-aligned with depth)
  <cam>_meta.json   node, mode, frame count, timestamps
"""
import argparse
import json
import os
import shutil
import subprocess
import time
import warnings

import numpy as np

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import APP, R  # noqa: E402  (roots from this file's location)
CAMS = {
    "top": {"depth": "/dev/video4", "ir": "/dev/video6", "live": "run/live_top.jpg"},
    "close": {"depth": "/dev/video10", "ir": "/dev/video12", "live": "run/live_close.jpg"},
}
MODES = [(848, 480, 15), (848, 480, 30), (640, 480, 15), (640, 480, 30), (848, 480, 6), (640, 480, 6), (480, 270, 15)]


def ff(node, fmt, w, h, fps, n):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-input_format", fmt,
           "-video_size", f"{w}x{h}", "-framerate", str(fps), "-i", node, "-frames:v", str(n),
           "-f", "rawvideo", "-pix_fmt", fmt, "-"]
    p = subprocess.run(cmd, capture_output=True, timeout=45)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(p.stderr.decode(errors="replace").strip()[-200:])
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", choices=list(CAMS), required=True)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--skip", type=int, default=15, help="frames dropped while depth auto-exposure settles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="", help="WxH@fps to force a mode")
    ap.add_argument("--no-ir", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    c = CAMS[a.cam]
    t0 = time.time()
    try:
        shutil.copy(os.path.join(R, c["live"]), os.path.join(a.out, f"{a.cam}_color.jpg"))
    except OSError:
        pass
    modes = MODES
    if a.mode:
        wh, fps = a.mode.split("@")
        w, h = wh.split("x")
        modes = [(int(w), int(h), int(fps))]
    errs = []
    for (w, h, fps) in modes:
        try:
            raw = ff(c["depth"], "gray16le", w, h, fps, a.frames + a.skip)
            break
        except Exception as e:  # mode not allowed on this link (top camera runs at USB2 speed)
            errs.append(f"{w}x{h}@{fps}: {e}")
    else:
        raise SystemExit("depth grab failed: " + " | ".join(errs))
    st = np.frombuffer(raw, "<u2")
    k = st.size // (w * h)
    st = st[: k * w * h].reshape(k, h, w)[min(a.skip, max(k - 5, 0)):].astype(np.float32)
    st[st == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fill = np.mean(np.isfinite(st), axis=0).astype(np.float32)
        depth = np.nanmedian(st, axis=0).astype(np.float32)
        std = np.nanstd(st, axis=0).astype(np.float32)
    depth[fill < 0.3] = np.nan
    np.save(os.path.join(a.out, f"{a.cam}_depth.npy"), depth)
    np.save(os.path.join(a.out, f"{a.cam}_fill.npy"), fill)
    np.save(os.path.join(a.out, f"{a.cam}_std.npy"), std)
    ir_ok = False
    if not a.no_ir:
        try:
            ir = ff(c["ir"], "gray", w, h, fps, 6)
            im = np.frombuffer(ir, np.uint8)
            n = im.size // (w * h)
            import cv2
            cv2.imwrite(os.path.join(a.out, f"{a.cam}_ir.png"), im[(n - 1) * w * h: n * w * h].reshape(h, w))
            ir_ok = True
        except Exception as e:
            errs.append(f"ir: {e}")
    meta = {"cam": a.cam, "node": c["depth"], "mode": [w, h, fps], "frames_used": int(st.shape[0]),
            "t_start_unix": t0, "t_end_unix": time.time(), "valid_frac": float(np.isfinite(depth).mean()),
            "depth_unit_mm": 1.0, "ir": ir_ok, "errors": errs}
    json.dump(meta, open(os.path.join(a.out, f"{a.cam}_meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
